from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import Positions, SOURCE_COMMIT, load_model, to_device


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clock_band(row):
    remaining = row["remaining_seconds"]
    return "under_10s" if remaining < 10 else "10_to_30s" if remaining < 30 else "30s_plus"


def timing_baseline(rows):
    values = defaultdict(list)
    for row in rows:
        elapsed = row.get("elapsed_seconds")
        if elapsed is not None and elapsed >= 0:
            values["all"].append(elapsed)
            values[row["time_control"] + "/" + clock_band(row)].append(elapsed)
    return {key: float(np.median(value)) for key, value in values.items()}


def loss_for(model, batch, timing_weight):
    move, timing, _ = model(batch["tokens"], batch["self_elos"], batch["oppo_elos"])
    move = move.masked_fill(~batch["legal"], -torch.inf)
    move_loss = F.cross_entropy(move, batch["target"])
    valid = batch["time_valid"]
    time_loss = F.cross_entropy(timing[valid], batch["time_target"][valid]) if valid.any() else timing.sum() * 0
    return move_loss + timing_weight * time_loss


def evaluate(model, dataset, device="cpu", batch_size=32, empirical=None):
    model.eval()
    groups = defaultdict(list)
    offset = 0
    with torch.no_grad():
        for raw in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            batch = to_device(raw, device)
            move, timing, seconds = model(batch["tokens"], batch["self_elos"], batch["oppo_elos"])
            move = move.masked_fill(~batch["legal"], -torch.inf)
            nll = F.cross_entropy(move, batch["target"], reduction="none")
            top = move.topk(3, dim=-1).indices
            for i in range(len(nll)):
                row = dataset.rows[offset + i]
                elapsed = row.get("elapsed_seconds")
                item = {"nll": nll[i].item(), "top1": int(top[i, 0] == batch["target"][i]),
                        "top3": int((top[i] == batch["target"][i]).any())}
                if elapsed is not None:
                    item["absolute_time_error"] = abs(seconds[i].item() - elapsed)
                    item["long_think"] = elapsed >= 99.9
                    if empirical and "all" in empirical:
                        estimate = empirical.get(row["time_control"] + "/" + clock_band(row), empirical["all"])
                        item["baseline_time_error"] = abs(estimate - elapsed)
                groups["all"].append(item)
                groups["control/" + row["time_control"]].append(item)
                groups["clock/" + clock_band(row)].append(item)
                groups["color/" + str(row["player_color"])].append(item)
                if row["ply"] > 20:
                    groups["after_move_10"].append(item)
                if row.get("position_seen_in_train") is False:
                    groups["unseen_position"].append(item)
            offset += len(nll)
    result = {}
    for name, items in groups.items():
        times = [x["absolute_time_error"] for x in items if "absolute_time_error" in x]
        baseline = [x["baseline_time_error"] for x in items if "baseline_time_error" in x]
        result[name] = {
            "positions": len(items), "move_nll": float(np.mean([x["nll"] for x in items])),
            "top1": float(np.mean([x["top1"] for x in items])),
            "top3": float(np.mean([x["top3"] for x in items])),
            "timing_positions": len(times), "timing_mae_seconds": float(np.mean(times)) if times else None,
            "timing_median_absolute_error_seconds": float(np.median(times)) if times else None,
            "timing_p90_absolute_error_seconds": float(np.quantile(times, .9)) if times else None,
            "empirical_timing_mae_seconds": float(np.mean(baseline)) if baseline else None,
            "outside_timing_range": sum(x.get("long_think", False) for x in items),
        }
    return result


def select_parameters(model, final_blocks=0, *, policy_only=False):
    if not 0 <= final_blocks <= len(model.transformer.layers):
        raise ValueError("Invalid final block count")
    if policy_only and final_blocks:
        raise ValueError("Policy-only training cannot adapt transformer blocks")
    prefixes = ["proj_sq_from.", "proj_sq_to.", "promo_bias_proj."]
    if not policy_only:
        prefixes += ["fc_ponder.", "fc_ponder_hid.", "last_ln."]
    if final_blocks:
        prefixes += [f"transformer.layers.{i}." for i in range(len(model.transformer.layers) - final_blocks, len(model.transformer.layers))]
        prefixes.append("transformer.norm.")
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(any(name.startswith(prefix) for prefix in prefixes))
        if parameter.requires_grad:
            selected.append(parameter)
    return selected


def train(data: Path, checkpoint: Path, output: Path, *, device="cuda", epochs=8,
          batch_size=16, learning_rate=1e-5, timing_weight=.2, patience=2, final_blocks=0,
          seed=42, smoke=False, policy_only=False):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Use --device cpu --smoke for a readiness check only.")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or timing_weight < 0 or patience < 1:
        raise ValueError("Invalid training settings")
    if policy_only and (final_blocks or timing_weight):
        raise ValueError("Policy-only training requires final_blocks=0 and timing_weight=0")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Run directory is not empty; choose a new --output to preserve previous runs")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    training, validation = Positions(data / "train.jsonl"), Positions(data / "val.jsonl")
    if set(r["game_id"] for r in training.rows) & set(r["game_id"] for r in validation.rows):
        raise ValueError("Train/validation game overlap")
    if smoke:
        training.rows = training.rows[:4]
        validation.rows = validation.rows[:4]
        epochs, batch_size = 1, 2
    model = load_model(checkpoint, device)
    params = select_parameters(model, final_blocks, policy_only=policy_only)
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=.01)
    empirical = timing_baseline(training.rows)
    print("Evaluating unchanged checkpoint on validation games...", flush=True)
    run_start = time.perf_counter()
    baseline = evaluate(model, validation, device, batch_size, empirical)
    baseline_seconds = time.perf_counter() - run_start
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(device=device, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                    timing_weight=timing_weight, patience=patience, final_blocks=final_blocks, seed=seed, smoke=smoke,
                    policy_only=policy_only)
    manifest = {"settings": settings, "source_commit": SOURCE_COMMIT, "checkpoint_sha256": file_hash(checkpoint),
                "data_sha256": {name: file_hash(data / name) for name in ("train.jsonl", "val.jsonl", "test.jsonl")},
                "trainable_parameters": sum(p.numel() for p in params), "baseline": baseline, "epochs": [],
                "selection_metric": "validation move_nll", "selected_epoch": 0, "test_evaluated": False,
                "baseline_seconds": baseline_seconds}
    best, stale = baseline["all"]["move_nll"], 0

    def save(epoch):
        temporary = output / "best.pt.tmp"
        torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "source_commit": SOURCE_COMMIT, "settings": settings, "epoch": epoch,
                    "personalized": epoch > 0 and not smoke, "smoke_only": smoke}, temporary)
        temporary.replace(output / "best.pt")

    def record():
        temporary = output / "run.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output / "run.json")

    save(0)
    record()
    loader = DataLoader(training, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    updates, skipped_updates = 0, 0
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        # The policy projections are linear; keep the frozen backbone free of dropout noise.
        model.train(not policy_only)
        losses = []
        for batch_index, raw in enumerate(loader, 1):
            batch = to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = loss_for(model, batch, timing_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # CUDA GradScaler must be allowed to skip an overflowing FP16 update.
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=not use_amp)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if use_amp and scaler.get_scale() < previous_scale:
                skipped_updates += 1
            else:
                updates += 1
            losses.append(loss.item())
            if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(loader):
                elapsed = time.perf_counter() - epoch_start
                remaining = elapsed / batch_index * (len(loader) - batch_index)
                print(f"epoch {epoch}/{epochs} batch {batch_index}/{len(loader)}: "
                      f"loss {loss.item():.4f}, {elapsed:.1f}s elapsed, ~{remaining:.0f}s to validation", flush=True)
        training_seconds = time.perf_counter() - epoch_start
        metrics = evaluate(model, validation, device, batch_size, empirical)
        manifest["epochs"].append({"epoch": epoch, "training_loss": float(np.mean(losses)), "validation": metrics,
                                   "training_seconds": training_seconds, "total_seconds": time.perf_counter() - epoch_start,
                                   "positions_per_second": len(training) / training_seconds})
        if metrics["all"]["move_nll"] < best:
            best, stale = metrics["all"]["move_nll"], 0
            manifest["selected_epoch"] = epoch
            save(epoch)
        else:
            stale += 1
        manifest["optimizer_steps"] = updates
        manifest["skipped_amp_updates"] = skipped_updates
        manifest["elapsed_seconds"] = time.perf_counter() - run_start
        record()
        print(f"epoch {epoch}: validation move NLL {metrics['all']['move_nll']:.4f}; best {best:.4f}", flush=True)
        if stale >= patience:
            break
    return manifest
