from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def emit(value, output=None):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare and train your personal Maia3 chess model")
    parser.add_argument("--threads", type=int, default=4, help="CPU torch threads")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("platform", choices=["lichess", "chess.com"])
    fetch.add_argument("username")
    fetch.add_argument("--output", type=Path, default=Path("data/raw"))
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--pgn", type=Path, nargs="+", required=True)
    prepare.add_argument("--players", nargs="+", required=True)
    prepare.add_argument("--output", type=Path, default=Path("data/processed"))
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, default=Path("data/processed"))
    training.add_argument("--checkpoint", type=Path, default=Path("models/maia3-23m-ponder.pt"))
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    training.add_argument("--epochs", type=int, default=8)
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--learning-rate", type=float, default=1e-5)
    training.add_argument("--timing-weight", type=float, default=.2)
    training.add_argument("--patience", type=int, default=2)
    training.add_argument("--final-blocks", type=int, default=0)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--smoke", action="store_true")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--positions", type=Path, required=True)
    evaluation.add_argument("--training-positions", type=Path, default=Path("data/processed/train.jsonl"))
    evaluation.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    evaluation.add_argument("--batch-size", type=int, default=32)
    evaluation.add_argument("--output", type=Path, required=True)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", type=Path, default=Path("models/maia3-23m-ponder.pt"))
    prediction.add_argument("--position", type=Path, required=True, help="JSON row with current FEN, history, ratings and clocks")
    prediction.add_argument("--output", type=Path)
    exporting = commands.add_parser("export")
    exporting.add_argument("--checkpoint", type=Path, required=True)
    exporting.add_argument("--output", type=Path, required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--data", type=Path, default=Path("data/processed"))
    doctor.add_argument("--checkpoint", type=Path, default=Path("models/maia3-23m-ponder.pt"))
    doctor.add_argument("--output", type=Path, default=Path("reports/readiness.json"))
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.command == "fetch":
        from .fetch import fetch_games
        emit(fetch_games(args.platform, args.username, args.output))
    elif args.command == "prepare":
        from .data import prepare_pgn
        report = prepare_pgn(args.pgn, args.players, args.output)
        emit({k: v for k, v in report.items() if k not in {"policy", "splits"}})
    elif args.command == "train":
        from .training import train
        options = vars(args).copy()
        options.pop("command")
        options.pop("threads")
        report = train(**options)
        emit({"selected_epoch": report["selected_epoch"], "optimizer_steps": report["optimizer_steps"],
              "smoke_only": args.smoke, "report": str(args.output / "run.json")})
    elif args.command == "evaluate":
        from .model import Positions, load_model
        from .training import evaluate, file_hash, timing_baseline
        rows = Positions(args.positions)
        training_rows = Positions(args.training_positions).rows
        train_positions = {" ".join(row["fen"].split()[:4]) for row in training_rows}
        for row in rows.rows:
            row["position_seen_in_train"] = " ".join(row["fen"].split()[:4]) in train_positions
        report = evaluate(load_model(args.checkpoint, args.device), rows, args.device, args.batch_size,
                          timing_baseline(training_rows))
        emit({"checkpoint_sha256": file_hash(args.checkpoint), "positions_sha256": file_hash(args.positions),
              "metrics": report}, args.output)
    elif args.command == "predict":
        from .model import load_model, predict
        emit(predict(load_model(args.checkpoint), json.loads(args.position.read_text())), args.output)
    elif args.command == "export":
        from .export import export_model
        emit(export_model(args.checkpoint, args.output))
    elif args.command == "doctor":
        from .model import Positions, load_model, predict
        from .training import file_hash
        checks, errors = {}, []
        sets = []
        for split in ("train", "val", "test"):
            try:
                dataset = Positions(args.data / f"{split}.jsonl")
                sets.append({r["game_id"] for r in dataset.rows})
                checks[split] = {"positions": len(dataset), "games": len(sets[-1]),
                                 "timing_labels": sum(r.get("elapsed_seconds") is not None for r in dataset.rows)}
            except (ValueError, OSError) as error:
                errors.append(str(error))
        if len(sets) == 3 and any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            errors.append("Game leakage between data splits")
        try:
            model = load_model(args.checkpoint)
            row = Positions(args.data / "train.jsonl").rows[0]
            result = predict(model, row)
            checks["checkpoint"] = {"sha256": file_hash(args.checkpoint), "parameters": sum(p.numel() for p in model.parameters()),
                                    "strict_load": True, "legal_moves": len(result["moves"]), "expected_seconds": result["expected_seconds"]}
        except Exception as error:
            errors.append(str(error))
        report = {"pipeline_ready": not errors, "cuda_available": torch.cuda.is_available(), "checks": checks,
                  "errors": errors, "training_status": "Not inspected by doctor; consult the selected run.json",
                  "note": "CPU load/inference and dataset checks. GPU capacity and training quality are not measured."}
        emit(report, args.output)
        if errors:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
