"""Build a chronological recent-play development split without reading reserved test rows."""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
from statistics import median


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def position_key(row):
    return " ".join(row["fen"].split()[:4])


def describe(rows):
    games = {row["game_id"]: row for row in rows}
    controls = {}
    for control in sorted({row["time_control"] for row in rows}):
        selected = [row for row in games.values() if row["time_control"] == control]
        controls[control] = {
            "games": len(selected),
            "moves": sum(row["time_control"] == control for row in rows),
            "median_self_rating_per_game": median(row["elo_self"] for row in selected),
            "median_opponent_rating_per_game": median(row["elo_oppo"] for row in selected),
        }
    return {"games": len(games), "moves": len(rows), "game_ids": sorted(games),
            "first_date": min(row["date"] for row in rows),
            "last_date": max(row["date"] for row in rows),
            "time_controls": controls}


def prepare(source, output, *, since="2024-01-01", validation_since="2026-01-01"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if date.fromisoformat(since) >= date.fromisoformat(validation_since):
        raise ValueError("since must precede validation_since")
    if output == source or source.is_relative_to(output):
        raise ValueError("Output must not replace the source directory or its ancestors")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a fresh output")
    audit_path = source / "audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    test_ids = set(audit["splits"]["test"]["game_ids"]) if audit is not None else None
    rows, game_dates, game_sources, seen = [], {}, {}, set()
    hashes = {}
    for split in ("train", "val"):
        path = source / f"{split}.jsonl"
        hashes[path.name] = digest(path)
        loaded_ids = set()
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            game_id = row["game_id"]
            day = row["date"]
            if date.fromisoformat(day).isoformat() != day:
                raise ValueError("Dates must use YYYY-MM-DD")
            if game_id in game_dates and game_dates[game_id] != day:
                raise ValueError(f"Conflicting dates for game {game_id}")
            if game_id in game_sources and game_sources[game_id] != split:
                raise ValueError(f"Source train/validation game overlap: {game_id}")
            key = (game_id, row["ply"])
            if key in seen:
                raise ValueError(f"Duplicate game/ply: {key}")
            if test_ids is not None and game_id in test_ids:
                raise ValueError(f"Reserved test game in development data: {game_id}")
            seen.add(key)
            loaded_ids.add(game_id)
            game_dates[game_id], game_sources[game_id] = day, split
            rows.append(row)
        if audit is not None and loaded_ids != set(audit["splits"][split]["game_ids"]):
            raise ValueError(f"Source {split} game IDs disagree with audit")
    selected = sorted((row for row in rows if row["date"] >= since),
                      key=lambda row: (row["date"], row["game_id"], row["ply"]))
    training = [row for row in selected if row["date"] < validation_since]
    validation = [row.copy() for row in selected if row["date"] >= validation_since]
    if not training or not validation:
        raise ValueError("Both recent train and validation must contain games")
    positions = {position_key(row) for row in training}
    for row in validation:
        row["position_seen_in_train"] = position_key(row) in positions
    report = {
        "schema_version": 1,
        "source_sha256": hashes,
        "source_audit_sha256": digest(audit_path) if audit is not None else None,
        "policy": {"since": since, "validation_since": validation_since,
                   "split": "Whole games ordered by date, game_id, ply; chronological date boundary.",
                   "former_validation_role": "retired; reused as development data",
                   "validation_pristine": False,
                   "validation_note": "Previously inspected in historical analysis and used for earlier model selection; not fitted as training by this split.",
                   "reserved_test_rows_read": False,
                   "reserved_test_disjoint_verified_from_audit": test_ids is not None,
                   "test_placeholder": "Empty test.jsonl only satisfies the training manifest hash contract; it is not an evaluation set.",
                   "rating_summary": "Per-game medians grouped by exact recorded time control; platforms are not inferred or normalized."},
        "excluded_older_games": len({row["game_id"] for row in rows if row["date"] < since}),
        "former_validation_games_in_training": sum(game_sources[game_id] == "val" for game_id in {row["game_id"] for row in training}),
        "splits": {"train": describe(training), "val": describe(validation)},
        "validation_seen_positions": sum(row["position_seen_in_train"] for row in validation),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (("train", training), ("val", validation), ("test", [])):
        (output / f"{name}.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in values))
    report["output_sha256"] = {name: digest(output / name) for name in ("train.jsonl", "val.jsonl", "test.jsonl")}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/recent-policy-v1"))
    parser.add_argument("--since", default="2024-01-01")
    parser.add_argument("--validation-since", default="2026-01-01")
    args = parser.parse_args()
    report = prepare(args.source, args.output, since=args.since, validation_since=args.validation_since)
    print(json.dumps({"output": str(args.output), "splits": {
        key: {field: value[field] for field in ("games", "moves", "first_date", "last_date")}
        for key, value in report["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
