"""Local PGN preparation. Clock labels never become pre-move input features."""

from __future__ import annotations

from collections import Counter, deque
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re

import chess
import chess.pgn


class _QuietGameBuilder(chess.pgn.GameBuilder):
    def handle_error(self, error: Exception) -> None:
        self.game.errors.append(error)


def _time_control(value: str) -> tuple[float, float] | None:
    match = re.fullmatch(r"(\d+)(?:\+(\d+))?", value.strip())
    if not match or int(match[1]) <= 0:
        return None
    return float(match[1]), float(match[2] or 0)


def _rating(value: str | None) -> int | None:
    if value is None or not re.fullmatch(r"\d+", value.strip()):
        return None
    rating = int(value)
    return rating if rating > 0 else None


def _date(headers: chess.pgn.Headers) -> str | None:
    # UTCDate is preferred when present; an incomplete date is never guessed.
    value = headers.get("UTCDate", headers.get("Date", ""))
    try:
        return datetime.strptime(value, "%Y.%m.%d").date().isoformat()
    except ValueError:
        return None


def _annotation(node: chess.pgn.ChildNode, name: str) -> tuple[float | None, bool]:
    """Read one nonnegative clock/EMT value, rejecting malformed annotations."""
    tags = re.findall(r"\[%" + name + r"\b[^\]]*\]", node.comment, re.IGNORECASE)
    if not tags:
        return None, bool(re.search(r"\[%" + name + r"\b", node.comment, re.IGNORECASE))
    if len(tags) != 1:
        return None, True
    match = re.fullmatch(r"\[%" + name + r"\s+(\d+):([0-5]\d):([0-5]\d(?:\.\d+)?)\s*\]", tags[0], re.IGNORECASE)
    if not match:
        return None, True
    return int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3]), False


def _game_rows(game: chess.pgn.Game, targets: set[str], game_id: str,
               date: str, control: tuple[float, float], ratings: dict[bool, int],
               issues: Counter, rejected: Counter, coverage: Counter) -> list[dict]:
    initial, increment = control
    board = game.board()
    history = deque(maxlen=8)
    previous = {chess.WHITE: None, chess.BLACK: None}
    turns = Counter()
    normal_start = board.fen() == chess.Board().fen()
    rows = []
    for ply, node in enumerate(game.mainline(), 1):
        color = board.turn
        history.append(board.fen())
        first = turns[color] == 0
        before = initial if first and normal_start else previous[color]
        post, bad_clock = _annotation(node, "clk")
        emt, bad_emt = _annotation(node, "emt")
        turns[color] += 1
        if bad_clock:
            issues["malformed_clock"] += 1
        if bad_emt:
            issues["malformed_emt"] += 1
        if emt is not None and before is not None and emt > before:
            issues["emt_over_budget"] += 1
            bad_emt = True
        # The maximum possible bank includes only increments already earned.
        if post is not None and post > initial + turns[color] * increment:
            issues["clock_above_possible_bank"] += 1
            post, bad_clock = None, True
        elapsed = None
        source = "missing_pre_clock" if before is None else "missing_clock"
        if first:
            source = "unknown_first_move_convention"
        elif post is not None and before is not None:
            candidate = before + increment - post
            if candidate < 0 or candidate > before:
                issues["negative_elapsed" if candidate < 0 else "elapsed_over_budget"] += 1
                # An inconsistent observation cannot seed the next input clock.
                post, bad_clock = None, True
                source = "invalid_elapsed"
            elif bad_emt or (emt is not None and abs(emt - candidate) > 1.0):
                issues["inconsistent_emt"] += 1
                source = "inconsistent_emt"
            else:
                elapsed = candidate
                source = "clock_delta+emt" if emt is not None else "clock_delta"
        if bad_clock and source != "invalid_elapsed":
            source = "invalid_clock"
        # No forward fill: even a missing observation clears the same-player bank.
        previous[color] = post
        player = game.headers.get("White" if color else "Black", "").strip().casefold()
        if player in targets:
            coverage["target_moves"] += 1
            if post is not None:
                coverage["target_moves_with_valid_post_clock"] += 1
            if before is None:
                rejected["missing_pre_move_clock"] += 1
            else:
                rows.append({
                    "game_id": game_id, "date": date, "ply": ply,
                    "player_color": "white" if color else "black",
                    "fen": board.fen(), "history_fens": list(history),
                    "move_uci": node.move.uci(), "elo_self": ratings[color],
                    "elo_oppo": ratings[not color], "initial_seconds": initial,
                    "increment_seconds": increment, "remaining_seconds": before,
                    "elapsed_seconds": elapsed, "timing_source": source,
                    "time_control": game.headers["TimeControl"],
                })
        board.push(node.move)
    return rows


def prepare_pgn(paths: list[Path], players: list[str], output: Path) -> dict:
    """Write local JSONL splits and audit.json; raise ValueError if <3 usable games.

    Identity hashes canonical start FEN, mainline UCI moves and normalized players.
    The earliest date resolves duplicate date conflicts; the export with the most
    usable rows/labels supplies annotations. Splits are chronological by full game,
    with game_id breaking same-day ties. Repeated positions ignore FEN move counters.
    """
    targets = {name.strip().casefold() for name in players if name.strip()}
    if not targets:
        raise ValueError("At least one target player is required")
    if not paths:
        raise ValueError("At least one PGN path is required")
    output = Path(output)
    rejected_games, rejected_rows, clock_issues, coverage = Counter(), Counter(), Counter(), Counter()
    controls = Counter()
    games = {}
    parsed = duplicates = date_conflicts = 0
    for path in sorted(map(Path, paths)):
        with path.open(encoding="utf-8-sig") as handle:
            while True:
                try:
                    game = chess.pgn.read_game(handle, Visitor=_QuietGameBuilder)
                except (ValueError, IndexError):
                    rejected_games["parser_error"] += 1
                    # Recovery after an exception is ambiguous; stop this input.
                    break
                if game is None:
                    break
                parsed += 1
                if game.errors:
                    rejected_games["parser_error"] += 1
                    continue
                if game.headers.get("Variant", "Standard").casefold() not in {"standard", "chess", "normal"}:
                    rejected_games["unsupported_variant"] += 1
                    continue
                try:
                    board = game.board()
                except ValueError:
                    rejected_games["invalid_start_position"] += 1
                    continue
                if not board.is_valid():
                    rejected_games["invalid_start_position"] += 1
                    continue
                names = [game.headers.get(key, "").strip().casefold() for key in ("White", "Black")]
                if not targets.intersection(names):
                    rejected_games["no_target_player"] += 1
                    continue
                date = _date(game.headers)
                if date is None:
                    rejected_games["missing_or_invalid_date"] += 1
                    continue
                control = _time_control(game.headers.get("TimeControl", ""))
                if control is None:
                    rejected_games["unsupported_time_control"] += 1
                    continue
                ratings = {chess.WHITE: _rating(game.headers.get("WhiteElo")), chess.BLACK: _rating(game.headers.get("BlackElo"))}
                if any(value is None for value in ratings.values()):
                    rejected_games["missing_or_invalid_rating"] += 1
                    continue
                moves = list(game.mainline_moves())
                if not moves:
                    rejected_games["empty_game"] += 1
                    continue
                identity = [board.fen(), names, [move.uci() for move in moves]]
                game_id = sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
                issues, rejected, counts = Counter(), Counter(), Counter()
                rows = _game_rows(game, targets, game_id, date, control, ratings, issues, rejected, counts)
                candidate = {"date": date, "game_id": game_id, "rows": rows,
                             "issues": issues, "rejected": rejected, "coverage": counts,
                             "time_control": game.headers["TimeControl"]}
                if game_id in games:
                    duplicates += 1
                    previous = games[game_id]
                    if previous["date"] != date:
                        date_conflicts += 1
                    earliest = min(previous["date"], date)
                    def quality(item):
                        return len(item["rows"]), sum(row["elapsed_seconds"] is not None for row in item["rows"])
                    candidate = candidate if quality(candidate) > quality(previous) else previous
                    candidate["date"] = earliest
                    for row in candidate["rows"]:
                        row["date"] = earliest
                games[game_id] = candidate
    usable = []
    for game in games.values():
        clock_issues.update(game["issues"])
        rejected_rows.update(game["rejected"])
        coverage.update(game["coverage"])
        if game["rows"]:
            usable.append(game)
            controls[game["time_control"]] += 1
        else:
            rejected_games["no_usable_rows"] += 1
    usable.sort(key=lambda game: (game["date"], game["game_id"]))
    report = {
        "schema_version": 1, "status": "ready" if len(usable) >= 3 else "insufficient_games",
        "input_files": len(paths), "parsed_games": parsed, "duplicate_games": duplicates,
        "duplicate_date_conflicts": date_conflicts, "usable_games": len(usable),
        "usable_moves": sum(len(game["rows"]) for game in usable),
        "rejected_games": dict(rejected_games), "rejected_rows": dict(rejected_rows),
        "clock_issues": dict(clock_issues), "clock_coverage": dict(coverage),
        "time_controls": dict(controls), "splits": {}, "repeated_positions_across_splits": {},
        "policy": {
            "split": "Whole games ordered by UTCDate or Date, then game_id; 80/10/10 with at least one game per split.",
            "clock": "Only integer base seconds or base+increment Fischer controls; clocks are post-move. Same-player elapsed=previous_post+increment-current_post; pre-move=previous_post. Missing/invalid clocks clear the bank. First-move elapsed is masked; normal-start pre-move uses initial seconds.",
            "emt": "EMT only cross-checks clock deltas (1 second rounding tolerance); never supplies a missing pre-move clock or first-move label.",
            "rejections": "Reject parser errors, nonstandard variants, invalid positions, missing full dates, unsupported time controls, missing positive numeric ratings, and empty games. Rows require real pre-move clock features. Timing inconsistencies mask labels and invalid clocks cannot seed subsequent features.",
            "deduplication": "SHA256 of canonical starting FEN, mainline UCI sequence, and casefolded player names. Prefer duplicate export with most usable rows then timing labels; earliest date wins.",
            "counts": "Clock and row counts use selected unique exports; rejected_games counts rejected input records plus unique games without usable rows. Time controls count usable games.",
            "privacy": "Local files only; no network access or account lookup.",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    if len(usable) < 3:
        # Remove stale splits so a failed preparation cannot look training-ready.
        for split in ("train", "val", "test"):
            (output / f"{split}.jsonl").unlink(missing_ok=True)
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        raise ValueError(f"Need at least 3 unique usable games for nonempty train/val/test splits; found {len(usable)}. See {output / 'audit.json'}")
    n = len(usable)
    n_train = min(n - 2, max(1, int(n * 0.8)))
    n_val = max(1, int(n * 0.1))
    splits = {"train": usable[:n_train], "val": usable[n_train:n_train + n_val], "test": usable[n_train + n_val:]}
    positions = {}
    labeled = 0
    for split, selected in splits.items():
        rows = [row for game in selected for row in game["rows"]]
        labeled_here = sum(row["elapsed_seconds"] is not None for row in rows)
        labeled += labeled_here
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        positions[split] = {" ".join(row["fen"].split()[:4]) for row in rows}
        report["splits"][split] = {"games": len(selected), "moves": len(rows), "timing_labels": labeled_here,
                                    "game_ids": [game["game_id"] for game in selected],
                                    "first_date": selected[0]["date"], "last_date": selected[-1]["date"]}
    report["clock_coverage"].update({"usable_moves": report["usable_moves"], "timing_labels": labeled,
                                       "timing_label_fraction": labeled / report["usable_moves"]})
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        report["repeated_positions_across_splits"][f"{left}_{right}"] = len(positions[left] & positions[right])
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
