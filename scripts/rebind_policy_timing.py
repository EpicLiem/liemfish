"""Rebind a timing head after verifying a move-projection-only checkpoint change."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from chess_self.timing_data import feature_names
from chess_self.timing_model import load_timing_head

POLICY_KEYS = {"proj_sq_from.weight", "proj_sq_to.weight", "promo_bias_proj.weight"}
SAFE_GROUPS = {"clock", "phase", "chess", "history_summary", "history_window"}


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def state(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    result = payload.get("model_state_dict", payload)
    if not isinstance(result, dict) or not POLICY_KEYS <= set(result):
        raise ValueError("Checkpoint does not contain the expected policy state")
    if any(not isinstance(value, torch.Tensor) for value in result.values()):
        raise ValueError("Checkpoint state must contain only tensors")
    return result


def rebind(original_checkpoint, checkpoint, head, output, *, allow_policy_surprisal_change=False):
    original_checkpoint, checkpoint, head, output = map(Path, (original_checkpoint, checkpoint, head, output))
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    original_sha = digest(original_checkpoint)
    candidate_sha = digest(checkpoint)
    head_sha = digest(head)
    original, candidate = state(original_checkpoint), state(checkpoint)
    if original.keys() != candidate.keys():
        raise ValueError("Checkpoint state keys differ")
    changed = []
    for name, value in original.items():
        other = candidate[name]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError(f"Checkpoint shape or dtype changed: {name}")
        if not torch.isfinite(value).all() or not torch.isfinite(other).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
        if not torch.equal(value, other):
            if name not in POLICY_KEYS:
                raise ValueError(f"Nonpolicy checkpoint tensor changed: {name}")
            changed.append(name)
    timing_head, payload = load_timing_head(head, base_checkpoint_sha256=original_sha)
    groups = payload.get("extra", {}).get("feature_groups")
    if not isinstance(groups, list) or any(group not in SAFE_GROUPS for group in groups):
        raise ValueError("Unsupported or policy-dependent timing feature groups")
    if payload["feature_names"] != feature_names(groups):
        raise ValueError("Timing feature names do not match declared groups")
    policy_surprisal_changed = bool(changed) and timing_head.config.move_mode != "none"
    if policy_surprisal_changed and not allow_policy_surprisal_change:
        raise ValueError("Move-conditioned head uses policy surprisal; pass --allow-policy-surprisal-change and validate timing")
    provenance = {
        "original_head_sha256": head_sha,
        "original_checkpoint_sha256": original_sha,
        "checkpoint_sha256": candidate_sha,
        "changed_policy_tensors": sorted(changed),
        "nonpolicy_tensors_exact": True,
        "timing_weights_unchanged": True,
        "policy_surprisal_changed": policy_surprisal_changed,
        "timing_validation_required": policy_surprisal_changed,
    }
    payload["base_checkpoint_sha256"] = candidate_sha
    payload["extra"] = {**payload.get("extra", {}), "policy_only_rebind": provenance}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing artifacts even if another process races this run.
    with output.open("xb") as stream:
        torch.save(payload, stream)
    return {**provenance, "output": str(output), "output_sha256": digest(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-policy-surprisal-change", action="store_true")
    print(json.dumps(rebind(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
