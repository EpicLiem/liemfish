import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_recent_policy.py"
spec = importlib.util.spec_from_file_location("prepare_recent_policy", SCRIPT)
recent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recent)


def row(game, day, ply=1, fen="same w KQkq - 0 1"):
    return {"game_id": game, "date": day, "ply": ply, "fen": fen,
            "time_control": "600", "elo_self": 650, "elo_oppo": 700,
            "history_fens": [fen], "elapsed_seconds": None, "extra": "preserve"}


def source(tmp_path, training=None, validation=None):
    path = tmp_path / "source"
    path.mkdir()
    values = {"train": training or [row("old", "2023-12-31"), row("a", "2024-01-01")],
              "val": validation or [row("b", "2025-12-31"), row("c", "2026-01-01"), row("c", "2026-01-01", 3)]}
    for split, rows in values.items():
        (path / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    audit = {"splits": {split: {"game_ids": sorted({r["game_id"] for r in rows})} for split, rows in values.items()}}
    audit["splits"]["test"] = {"game_ids": ["reserved"]}
    (path / "audit.json").write_text(json.dumps(audit))
    # Invalid bytes ensure accidental reading/parsing of test rows cannot succeed.
    (path / "test.jsonl").write_bytes(b"\xffnever read this")
    return path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_whole_game_boundary_provenance_and_original_untouched(tmp_path):
    src = source(tmp_path)
    before = {p.name: p.read_bytes() for p in src.iterdir()}
    out = tmp_path / "out"
    report = recent.prepare(src, out)
    assert [r["game_id"] for r in read(out / "train.jsonl")] == ["a", "b"]
    val = read(out / "val.jsonl")
    assert [r["ply"] for r in val] == [1, 3]
    assert all(r["position_seen_in_train"] and r["extra"] == "preserve" for r in val)
    assert (out / "test.jsonl").read_bytes() == b""
    assert report["former_validation_games_in_training"] == 1
    assert report["policy"]["validation_pristine"] is False
    assert report["policy"]["reserved_test_disjoint_verified_from_audit"] is True
    assert report["excluded_older_games"] == 1
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before
    again = tmp_path / "again"
    recent.prepare(src, again)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == {p.name: p.read_bytes() for p in again.iterdir()}


@pytest.mark.parametrize("training,validation,message", [
    ([row("a", "2024-01-01"), row("a", "2024-01-01")], None, "Duplicate"),
    ([row("a", "2024-01-01"), row("a", "2025-01-01", 3)], None, "Conflicting dates"),
    ([row("a", "2024-01-01")], [row("a", "2024-01-01", 3), row("c", "2026-01-01")], "overlap"),
    ([row("reserved", "2024-01-01")], None, "Reserved test"),
    ([row("old", "2023-01-01")], [row("c", "2026-01-01")], "Both recent"),
])
def test_invalid_data_refused_without_output(tmp_path, training, validation, message):
    src = source(tmp_path, training, validation)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=message):
        recent.prepare(src, out)
    assert not out.exists()


def test_existing_output_preserved(tmp_path):
    src = source(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep").write_text("keep")
    with pytest.raises(ValueError, match="not empty"):
        recent.prepare(src, out)
    assert (out / "keep").read_text() == "keep"


def test_audit_mismatch_fails(tmp_path):
    src = source(tmp_path)
    audit = json.loads((src / "audit.json").read_text())
    audit["splits"]["train"]["game_ids"].append("missing")
    (src / "audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="disagree"):
        recent.prepare(src, tmp_path / "out")


def test_unseen_position_and_clock_counter_normalization(tmp_path):
    src = source(tmp_path, [row("a", "2024-01-01", fen="same w KQkq - 0 1")],
                 [row("b", "2026-01-01", fen="same w KQkq - 8 20"),
                  row("c", "2026-02-01", fen="different b - - 0 1")])
    out = tmp_path / "out"
    recent.prepare(src, out)
    assert [r["position_seen_in_train"] for r in read(out / "val.jsonl")] == [True, False]
