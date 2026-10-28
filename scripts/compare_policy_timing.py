"""Score unchanged timing weights with old/new policies on identical recorded moves."""
import argparse
import json
from pathlib import Path

import torch

from chess_self.model import Positions, load_model
from chess_self.timing_cache import digest
from chess_self.timing_data import feature_names
from chess_self.timing_experiments import evaluate_head
from chess_self.timing_metrics import compare_scorecards
from chess_self.timing_model import MaiaFeatures, load_timing_head


def compare(positions, original_checkpoint, checkpoint, original_head, head, output,
            *, device="cpu", batch_size=32, crps_points=256):
    positions, output = Path(positions), Path(output)
    if positions.name == "test.jsonl":
        raise ValueError("Reserved test data is not accepted")
    if output.exists():
        raise FileExistsError(output)
    rows = Positions(positions)
    if any(row.get("schema_version") != "timing-data-v2" for row in rows.rows):
        raise ValueError("Timing comparison requires timing-data-v2 development rows")
    excluded = sum(row["remaining_seconds"] <= 0 for row in rows.rows)
    rows.rows = [row for row in rows.rows if row["remaining_seconds"] > 0]
    if not rows.rows:
        raise ValueError("No positive-clock development rows")
    scores = {}
    for name, policy_path, head_path in (("original", original_checkpoint, original_head),
                                        ("candidate", checkpoint, head)):
        model = MaiaFeatures(load_model(Path(policy_path), device)).eval()
        try:
            timing, payload = load_timing_head(Path(head_path), device=device,
                                              base_checkpoint_sha256=digest(Path(policy_path)))
            groups = payload["extra"]["feature_groups"]
            if feature_names(groups) != payload["feature_names"]:
                raise ValueError("Head feature ordering mismatch")
            scores[name] = evaluate_head(timing, rows, groups, device=device, batch_size=batch_size,
                                         extractor=model, conditional=True, crps_points=crps_points)
        finally:
            model.close()
    result = {"timing_information": "human_move_conditional_diagnostic", "test_evaluated": False,
              "excluded_nonpositive_clock_rows": excluded,
              "inputs": {str(path): digest(Path(path)) for path in
                         (positions, original_checkpoint, checkpoint, original_head, head)},
              "scores": scores,
              "candidate_minus_original": compare_scorecards(scores["candidate"], scores["original"])}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("positions", "original-checkpoint", "checkpoint", "original-head", "head", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--crps-points", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1 or args.crps_points < 1:
        parser.error("Batch size and CRPS points must be positive")
    torch.set_num_threads(2)
    result = compare(**vars(args))
    print(json.dumps({name: report["groups"]["all"] for name, report in result["scores"].items()}, indent=2))


if __name__ == "__main__":
    main()
