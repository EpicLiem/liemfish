import json
from pathlib import Path

import chess
import pytest

from chess_self.data import prepare_pgn


MOVES = "1. e4 {[%clk 0:01:39]} e5 {[%clk 0:01:38]} 2. Nf3 {[%clk 0:01:35]} Nc6 {[%clk 0:01:36]} 3. Bb5 {[%clk 0:01:30]} a6 {[%clk 0:01:32]} *"


def pgn(index=1, moves=MOVES, white="Me", black=None, **headers):
    values = {"Event": "test", "Date": f"2025.01.{index:02}", "White": white,
              "Black": black or f"Opponent{index}", "Result": "*", "WhiteElo": "1500",
              "BlackElo": "1600", "TimeControl": "100+2"}
    values.update(headers)
    return "\n".join(f'[{key} "{value}"]' for key, value in values.items()) + "\n\n" + moves + "\n\n"


def prepare(tmp_path, games=None, players=None):
    source = tmp_path / "games.pgn"
    source.write_text("".join(games or [pgn(i) for i in range(1, 4)]))
    output = tmp_path / "prepared"
    report = prepare_pgn([source], players or ["me"], output)
    rows = {split: [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
            for split in ("train", "val", "test")}
    assert json.loads((output / "audit.json").read_text()) == report
    return report, rows


@pytest.mark.parametrize("color", ["white", "black"])
def test_player_filter_and_pre_move_history(tmp_path, color):
    games = [pgn(i, white="Me" if color == "white" else f"Opponent{i}",
                 black=f"Opponent{i}" if color == "white" else "ME") for i in range(1, 4)]
    report, splits = prepare(tmp_path, games)
    rows = splits["train"]
    assert report["usable_moves"] == 9
    assert [row["ply"] for row in rows] == ([1, 3, 5] if color == "white" else [2, 4, 6])
    assert all(row["player_color"] == color for row in rows)
    assert rows[0]["elo_self"] == (1500 if color == "white" else 1600)
    assert rows[0]["elo_oppo"] == (1600 if color == "white" else 1500)
    board = chess.Board()
    history = []
    for move in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"):
        history.append(board.fen())
        row = next((row for row in rows if row["ply"] == len(history)), None)
        if row:
            assert row["fen"] == board.fen()
            assert row["history_fens"] == history[-8:]
            assert row["move_uci"] == move
        board.push_uci(move)


def test_same_player_increment_and_no_future_leakage(tmp_path):
    _, splits = prepare(tmp_path)
    first, second, third = splits["train"]
    assert first["remaining_seconds"] == 100
    assert first["elapsed_seconds"] is None
    assert first["timing_source"] == "unknown_first_move_convention"
    assert (second["remaining_seconds"], second["elapsed_seconds"]) == (99, 6)
    assert (third["remaining_seconds"], third["elapsed_seconds"]) == (95, 7)
    assert second["initial_seconds"] == 100
    assert second["increment_seconds"] == 2
    altered = MOVES.replace("0:01:35", "0:01:31")
    other = tmp_path / "other"
    other.mkdir()
    _, changed = prepare(other, [pgn(i, moves=altered) for i in range(1, 4)])
    before, after = second.copy(), changed["train"][1].copy()
    assert before.pop("elapsed_seconds") != after.pop("elapsed_seconds")
    assert before == after


def test_missing_clock_clears_bank_and_emt_does_not_fill(tmp_path):
    moves = MOVES.replace("{[%clk 0:01:35]}", "{[%emt 0:00:06]}")
    report, splits = prepare(tmp_path, [pgn(i, moves=moves) for i in range(1, 4)])
    assert [row["ply"] for row in splits["train"]] == [1, 3]
    assert splits["train"][1]["elapsed_seconds"] is None
    assert splits["train"][1]["remaining_seconds"] == 99
    assert report["rejected_rows"]["missing_pre_move_clock"] == 3


@pytest.mark.parametrize("clock,issue", [("0:01:42", "negative_elapsed"),
                                            ("0:00:01", "elapsed_over_budget"),
                                            ("-0:00:01", "malformed_clock"),
                                            ("0:03:00", "clock_above_possible_bank")])
def test_invalid_clock_masks_label_and_next_feature(tmp_path, clock, issue):
    moves = MOVES.replace("0:01:35", clock)
    report, splits = prepare(tmp_path, [pgn(i, moves=moves) for i in range(1, 4)])
    assert report["clock_issues"][issue] == 3
    assert [row["ply"] for row in splits["train"]] == [1, 3]
    assert splits["train"][1]["elapsed_seconds"] is None


def test_inconsistent_emt_masks_label(tmp_path):
    moves = MOVES.replace("[%clk 0:01:35]", "[%clk 0:01:35] [%emt 0:00:50]")
    report, splits = prepare(tmp_path, [pgn(i, moves=moves) for i in range(1, 4)])
    assert report["clock_issues"]["inconsistent_emt"] == 3
    assert splits["train"][1]["elapsed_seconds"] is None
    assert splits["train"][2]["remaining_seconds"] == 95


def test_chronological_full_game_split_dedup_and_overlap_audit(tmp_path):
    one, two = tmp_path / "one.pgn", tmp_path / "two.pgn"
    one.write_text("".join(pgn(i) for i in reversed(range(1, 11))))
    two.write_text(pgn(1, Event="another export") + pgn(10))
    output = tmp_path / "prepared"
    report = prepare_pgn([two, one], ["Me"], output)
    assert report["duplicate_games"] == 2
    assert report["usable_games"] == 10
    assert [report["splits"][split]["games"] for split in ("train", "val", "test")] == [8, 1, 1]
    ids = [set(report["splits"][split]["game_ids"]) for split in ("train", "val", "test")]
    assert not (ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2])
    assert report["splits"]["train"]["last_date"] == "2025-01-08"
    assert report["splits"]["test"]["first_date"] == "2025-01-10"
    assert report["repeated_positions_across_splits"] == {"train_val": 3, "train_test": 3, "val_test": 3}


@pytest.mark.parametrize("extra,reason", [({"Date": "????.??.??"}, "missing_or_invalid_date"),
                                            ({"TimeControl": "40/7200:3600"}, "unsupported_time_control"),
                                            ({"TimeControl": "100+2d"}, "unsupported_time_control"),
                                            ({"WhiteElo": "?"}, "missing_or_invalid_rating"),
                                            ({"BlackElo": ""}, "missing_or_invalid_rating"),
                                            ({"Variant": "Atomic"}, "unsupported_variant")])
def test_rejected_game_metadata(tmp_path, extra, reason):
    report, _ = prepare(tmp_path, [pgn(i) for i in range(1, 4)] + [pgn(4, **extra)])
    assert report["rejected_games"][reason] == 1


def test_invalid_pgn_is_rejected_without_partial_game(tmp_path):
    report, _ = prepare(tmp_path, [pgn(i) for i in range(1, 4)] + [pgn(4, moves="1. e4 e5 2. Bh6 *")])
    assert report["rejected_games"]["parser_error"] == 1
    assert report["usable_games"] == 3


def test_insufficient_games_writes_audit_and_removes_stale_splits(tmp_path):
    source = tmp_path / "games.pgn"
    source.write_text(pgn() + pgn())
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "train.jsonl").write_text("stale")
    with pytest.raises(ValueError, match="at least 3 unique usable games"):
        prepare_pgn([source], ["Me"], output)
    report = json.loads((output / "audit.json").read_text())
    assert report["status"] == "insufficient_games"
    assert report["duplicate_games"] == 1
    assert not (output / "train.jsonl").exists()


def test_integer_seconds_and_history_bound(tmp_path):
    moves = MOVES.replace(" *", " 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 *")
    # Keep clocks on later moves so the history window can be exercised.
    moves = moves.replace("Ba4", "Ba4 {[%clk 0:01:25]}").replace("O-O", "O-O {[%clk 0:01:20]}")
    _, splits = prepare(tmp_path, [pgn(i, moves=moves, TimeControl="100") for i in range(1, 4)])
    row = splits["train"][-1]
    assert row["ply"] == 11
    assert len(row["history_fens"]) == 8
    assert row["history_fens"][-1] == row["fen"]
    assert row["increment_seconds"] == 0
    assert splits["train"][1]["elapsed_seconds"] == 4


def test_empty_or_non_pgn_fails_clearly(tmp_path):
    source = tmp_path / "bad.pgn"
    source.write_text("this is not a PGN")
    with pytest.raises(ValueError, match="at least 3 unique usable games"):
        prepare_pgn([source], ["Me"], tmp_path / "prepared")


def test_missing_first_clock_drops_next_same_player_move(tmp_path):
    moves = MOVES.replace("{[%clk 0:01:39]}", "{[%emt 0:00:01]}")
    _, splits = prepare(tmp_path, [pgn(i, moves=moves) for i in range(1, 4)])
    rows = splits["train"]
    assert [row["ply"] for row in rows] == [1, 5]
    assert rows[0]["elapsed_seconds"] is None
    assert rows[1]["remaining_seconds"] == 95
    assert rows[1]["elapsed_seconds"] == 7


def test_both_players_and_black_clock_math(tmp_path):
    _, splits = prepare(tmp_path, players=["Me", "Opponent1", "Opponent2", "Opponent3"])
    rows = splits["train"]
    assert [row["ply"] for row in rows] == list(range(1, 7))
    assert rows[1]["remaining_seconds"] == 100
    assert rows[1]["elapsed_seconds"] is None
    assert (rows[3]["remaining_seconds"], rows[3]["elapsed_seconds"]) == (98, 4)
    assert (rows[5]["remaining_seconds"], rows[5]["elapsed_seconds"]) == (96, 6)


def test_custom_start_does_not_invent_first_pre_clock(tmp_path):
    start = chess.Board()
    start.push_uci("d2d4")
    moves = "1... d5 {[%clk 0:01:38]} 2. Nf3 {[%clk 0:01:35]} Nf6 {[%clk 0:01:36]} 3. e3 {[%clk 0:01:30]} *"
    _, splits = prepare(tmp_path, [pgn(i, moves=moves, SetUp="1", FEN=start.fen()) for i in range(1, 4)])
    assert len(splits["train"]) == 1
    row = splits["train"][0]
    assert row["ply"] == 4
    assert row["remaining_seconds"] == 95
    assert row["elapsed_seconds"] == 7
    assert row["history_fens"][0] == start.fen()


def test_emt_over_budget_is_masked_even_within_rounding_tolerance(tmp_path):
    moves = MOVES.replace("[%clk 0:01:35]", "[%clk 0:00:02] [%emt 0:01:39.5]")
    report, splits = prepare(tmp_path, [pgn(i, moves=moves) for i in range(1, 4)])
    assert report["clock_issues"]["emt_over_budget"] == 3
    assert splits["train"][1]["elapsed_seconds"] is None
