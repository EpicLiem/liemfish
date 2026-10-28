"""Versioned, causal timing features layered over the unchanged PGN benchmark."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import statistics

import chess
import chess.pgn

from .data import _QuietGameBuilder, _annotation, _date, _game_rows, _rating, _time_control

SCHEMA_VERSION = "timing-data-v2"
INTERVAL_ASSUMPTION = "independent nearest rounding of both clock readings; bounds only, not a known platform likelihood"
GROUPS = {
    "clock": ["log_base", "log_increment", "log_own_clock", "log_opponent_clock", "own_clock_known", "opponent_clock_known", "own_base_ratio", "opponent_base_ratio", "relative_clock_difference", "increment_clock_ratio"],
    "phase": ["move_number", "piece_fraction", "nonpawn_material_fraction"],
    "chess": ["in_check", "log_legal_count", "forced_reply", "capture_available", "previous_move_was_capture"],
    "history_summary": [f"{side}_{name}" for side in ("own", "opponent") for name in ("log_mean", "log_sd", "log_last", "last_known", "log_change", "valid_fraction")],
    "history_window": [f"{side}_{name}_{i}" for side in ("own", "opponent") for i in range(8) for name in ("log_time", "known")],
    "familiarity": ["log_prior_position_games", "log_prior_line_games", "first_unfamiliar_position", "familiarity_known"],
    "difficulty": ["policy_entropy", "top_two_gap", "opponent_surprisal", "opponent_surprisal_known"],
    "retrieval": ["nearest_timing_logmean", "log_neighbor_count"],
    "engine": ["score_disagreement", "move_disagreement", "known"],
}


def feature_names(groups: list[str]) -> list[str]:
    if len(groups) != len(set(groups)) or any(g not in GROUPS for g in groups):
        raise ValueError(f"Feature groups must be unique members of {tuple(GROUPS)}")
    return [f"{group}/{name}" for group in groups for name in GROUPS[group]]


def _number(value):
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def feature_vector(row: dict, groups: list[str]) -> list[float]:
    """Only named pre-move fields are read; labels, ending metadata and move_uci are excluded."""
    feature_names(groups)
    result = []
    board = chess.Board(row["fen"]) if any(g in groups for g in ("phase", "chess")) else None
    for group in groups:
        if group == "clock":
            base, inc = float(row["initial_seconds"]), float(row["increment_seconds"])
            own, opp = row.get("remaining_seconds"), row.get("opponent_remaining_seconds")
            ok, pk = _number(own), _number(opp)
            own, opp = float(own) if ok else 0., float(opp) if pk else 0.
            result += [math.log1p(base), math.log1p(inc), math.log1p(own), math.log1p(opp), float(ok), float(pk), own / max(base, 1), opp / max(base, 1), (own - opp) / max(base, 1) if ok and pk else 0., inc / max(own, 1)]
        elif group == "phase":
            material = sum(len(board.pieces(piece, color)) * value for color in chess.COLORS for piece, value in ((chess.KNIGHT, 3), (chess.BISHOP, 3), (chess.ROOK, 5), (chess.QUEEN, 9)))
            result += [board.fullmove_number / 100, len(board.piece_map()) / 32, material / 62]
        elif group == "chess":
            legal = list(board.legal_moves)
            result += [float(board.is_check()), math.log1p(len(legal)), float(len(legal) == 1), float(any(board.is_capture(move) for move in legal)), float(row.get("previous_move_was_capture", False))]
        elif group in {"history_summary", "history_window"}:
            for side in ("own", "opponent"):
                history = list(row.get(side + "_time_history", []))[-8:]
                if group == "history_window":
                    history = [None] * (8 - len(history)) + history
                    for value in history:
                        result += [math.log1p(value) if _number(value) else 0., float(_number(value))]
                else:
                    values = [float(v) for v in history if _number(v)]
                    last = history[-1] if history else None
                    change = (math.log1p(history[-1]) - math.log1p(history[-2])) if len(history) > 1 and _number(history[-1]) and _number(history[-2]) else 0.
                    result += [math.log1p(statistics.mean(values)) if values else 0., math.log1p(statistics.pstdev(values)) if values else 0., math.log1p(last) if _number(last) else 0., float(_number(last)), change, len(values) / 8]
        elif group == "familiarity":
            result += [math.log1p(row.get("prior_position_games", 0)), math.log1p(row.get("prior_line_games", 0)), float(row.get("first_unfamiliar_position", False)), float(row.get("familiarity_known", False))]
        elif group == "retrieval":
            result += [float(row.get("nearest_timing_logmean", 0.)), math.log1p(row.get("neighbor_count", 0))]
        elif group == "engine":
            score = row.get("engine_score_disagreement")
            result += [math.log1p(score) if _number(score) else 0., float(row.get("engine_move_disagreement", 0)), float(_number(score))]
        elif group == "difficulty":
            surprise = row.get("opponent_surprisal", row.get("opponent_move_surprisal"))
            result += [float(row.get("policy_entropy", 0.)), float(row.get("top_two_gap", row.get("policy_top_two_gap", 0.))), float(surprise) if _number(surprise) else 0., float(_number(surprise))]
    if not all(math.isfinite(v) for v in result):
        raise ValueError("Nonfinite timing feature")
    return result


def _hash(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _precision(node):
    value, bad = _annotation(node, "clk")
    if value is None or bad:
        return None
    match = re.search(r"\[%clk\s+\d+:[0-5]\d:([0-5]\d(?:\.\d+)?)\s*\]", node.comment, re.I)
    seconds = match[1]
    return 10. ** -len(seconds.partition(".")[2]) if "." in seconds else 1.


def _timestamp(headers):
    if headers.get("UTCDate") and headers.get("UTCTime"):
        try:
            dt = datetime.strptime(headers["UTCDate"] + " " + headers["UTCTime"], "%Y.%m.%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass
    return None


def _identity(game):
    names = [game.headers.get(k, "").strip().casefold() for k in ("White", "Black")]
    value = [game.board().fen(), names, [m.uci() for m in game.mainline_moves()]]
    return sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _enrich(game, rows, path):
    """Replay only completed prior turns; invalid clock banks follow the legacy parser."""
    output = {r["ply"]: dict(r) for r in rows}
    board = game.board()
    initial, increment = _time_control(game.headers["TimeControl"])
    names = {game.headers.get(k, "").strip().casefold() for k in ("White", "Black")}
    ratings = {c: _rating(game.headers.get("WhiteElo" if c else "BlackElo")) for c in chess.COLORS}
    all_rows = {r["ply"]: r for r in _game_rows(game, names, rows[0]["game_id"], rows[0]["date"], (initial, increment), ratings, Counter(), Counter(), Counter())}
    normal_start = board.fen() == chess.Board().fen()
    clocks = {c: initial if normal_start else None for c in chess.COLORS}
    precision = {c: None for c in chess.COLORS}
    histories = {c: deque(maxlen=8) for c in chess.COLORS}
    turns = Counter()
    prior_capture = False
    line = []
    timestamp = _timestamp(game.headers)
    site = game.headers.get("Site", "").casefold()
    platform = "lichess" if "lichess.org" in site or path.name.casefold().startswith("lichess-") else "chess.com" if "chess.com" in site or path.name.casefold().startswith("chess.com-") else "unknown"
    for ply, node in enumerate(game.mainline(), 1):
        color = board.turn
        before = clocks[color]
        post, bad = _annotation(node, "clk")
        turns[color] += 1
        if post is not None and post > initial + turns[color] * increment:
            post, bad = None, True
        if turns[color] > 1 and before is not None and post is not None:
            delta = before + increment - post
            if delta < 0 or delta > before:
                post, bad = None, True
        if bad:
            post = None
        elapsed = all_rows.get(ply, {}).get("elapsed_seconds")
        if ply in output:
            row = output[ply]
            q_before, q_after = precision[color], _precision(node) if post is not None else None
            interval = None
            if elapsed is not None and q_before is not None and q_after is not None:
                radius = (q_before + q_after) / 2
                interval = [max(0., elapsed - radius), elapsed + radius]
            row.update({
                "schema_version": SCHEMA_VERSION, "opponent_remaining_seconds": clocks[not color],
                "white_player": game.headers.get("White"), "black_player": game.headers.get("Black"),
                "own_clock_known": before is not None, "opponent_clock_known": clocks[not color] is not None,
                "own_time_history": list(histories[color]), "opponent_time_history": list(histories[not color]),
                "canonical_time_control": f"{initial:g}+{increment:g}", "platform": platform,
                "source_file": str(path), "timestamp_utc": timestamp, "timestamp_uncertain": timestamp is None,
                "timestamp_headers": {k: game.headers[k] for k in ("UTCDate", "UTCTime", "Date", "StartTime", "EndTime", "Timezone") if k in game.headers},
                "clock_precision_before_seconds": q_before, "clock_precision_after_seconds": q_after,
                "opponent_clock_precision_seconds": precision[not color],
                "observation_interval_seconds": interval, "observation_interval_assumption": INTERVAL_ASSUMPTION,
                "observation_interval_truncation_sensitivity_seconds": [max(0., elapsed - q_after), elapsed + q_before] if interval else None,
                "termination": game.headers.get("Termination"), "result": game.headers.get("Result"),
                "end_time": game.headers.get("EndTime"), "explicit_premove": None,
                "previous_move_was_capture": prior_capture,
                "line_key": sha256(" ".join(line).encode()).hexdigest(),
            })
        histories[color].append(elapsed)
        clocks[color] = post
        precision[color] = _precision(node) if post is not None else None
        prior_capture = board.is_capture(node.move)
        line.append(node.move.uci())
        board.push(node.move)
    return list(output.values())


def _earlier(left, right):
    known = all(r.get("timestamp_utc") and not r.get("timestamp_uncertain", False) for r in (left, right))
    if known:
        stamps = [datetime.fromisoformat(r["timestamp_utc"].replace("Z", "+00:00")) for r in (left, right)]
        if any(t.tzinfo is None for t in stamps):
            raise ValueError("Timezone-aware chronology required")
        return stamps[0] < stamps[1]
    return left["date"] < right["date"]


def add_familiarity(training: list[dict], validation: list[dict]):
    """Count earlier games, never the current game or an ambiguously ordered same-day game."""
    games = defaultdict(list)
    for row in training:
        games[row["game_id"]].append(row)
    indexes = [(rs[0], {" ".join(r["fen"].split()[:4]) for r in rs}, {r["line_key"] for r in rs}) for rs in games.values()]
    targets = defaultdict(list)
    for row in training + validation:
        targets[row["game_id"]].append(row)
    for rs in targets.values():
        positions, lines = Counter(), Counter()
        for earlier, pkeys, lkeys in indexes:
            if earlier["game_id"] != rs[0]["game_id"] and _earlier(earlier, rs[0]):
                positions.update(pkeys)
                lines.update(lkeys)
        deviated = False
        for row in sorted(rs, key=lambda r: r["ply"]):
            count = positions[" ".join(row["fen"].split()[:4])]
            row.update(prior_position_games=count, prior_line_games=lines[row["line_key"]],
                       first_unfamiliar_position=not deviated and count == 0, familiarity_known=True)
            deviated = deviated or count == 0


def prepare_timing_data(paths, players, reference: Path, output: Path) -> dict:
    reference, output = Path(reference), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory; existing datasets are never overwritten")
    paths = sorted(map(Path, paths))
    targets = {p.strip().casefold() for p in players if p.strip()}
    if not paths or not targets:
        raise ValueError("PGN paths and target players are required")
    audit_path = reference / "audit.json"
    audit = json.loads(audit_path.read_text())
    ids = {s: set(audit["splits"][s]["game_ids"]) for s in ("train", "val", "test")}
    if any(ids[a] & ids[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Reference split game IDs overlap")
    legacy = {s: [json.loads(line) for line in (reference / f"{s}.jsonl").read_text().splitlines() if line.strip()] for s in ("train", "val")}
    for split, rows in legacy.items():
        if {r["game_id"] for r in rows} != ids[split]:
            raise ValueError(f"Reference {split} rows do not match audit IDs")
    wanted = ids["train"] | ids["val"]
    selected = {}
    for path in paths:
        with path.open(encoding="utf-8-sig") as f:
            while True:
                game = chess.pgn.read_game(f, Visitor=_QuietGameBuilder)
                if game is None:
                    break
                if game.errors or game.headers.get("Variant", "Standard").casefold() not in {"standard", "chess", "normal"}:
                    continue
                try:
                    game_id = _identity(game)
                except ValueError:
                    continue
                if game_id not in wanted:
                    continue
                date, control = _date(game.headers), _time_control(game.headers.get("TimeControl", ""))
                ratings = {c: _rating(game.headers.get("WhiteElo" if c else "BlackElo")) for c in chess.COLORS}
                if date is None or control is None or any(v is None for v in ratings.values()):
                    continue
                rows = _game_rows(game, targets, game_id, date, control, ratings, Counter(), Counter(), Counter())
                if not rows:
                    continue
                quality = (len(rows), sum(r["elapsed_seconds"] is not None for r in rows))
                if game_id not in selected or quality > selected[game_id][0]:
                    selected[game_id] = (quality, _enrich(game, rows, path))
    missing = wanted - selected.keys()
    if missing:
        raise ValueError(f"PGNs are missing {len(missing)} reference train/validation games")
    enriched = {(r["game_id"], r["ply"]): r for _, rs in selected.values() for r in rs}
    splits = {}
    for split, rows in legacy.items():
        splits[split] = []
        for row in rows:
            new = enriched.get((row["game_id"], row["ply"]))
            if new is None or any(new[k] != value for k, value in row.items() if k != "date"):
                raise ValueError("Selected PGN annotations differ from the preserved benchmark; use its original source snapshots")
            new = dict(new, date=row["date"])
            if new["timestamp_utc"] and new["timestamp_utc"][:10] != row["date"]:
                new.update(timestamp_utc=None, timestamp_uncertain=True)
            splits[split].append(new)
    add_familiarity(splits["train"], splits["val"])
    manifest = {
        "schema_version": SCHEMA_VERSION, "parser_version": "legacy-clock-v1+causal-enrichment-v2",
        "source_sha256": {str(p): _hash(p) for p in paths},
        "reference_sha256": {name: _hash(reference / name) for name in ("audit.json", "train.jsonl", "val.jsonl")},
        "split_policy": "Preserved reference train/val membership and row order; test IDs reserved, test examples never read or exported",
        "reserved_test_game_ids": sorted(ids["test"]), "test_evaluated": False,
        "observation_interval_assumption": INTERVAL_ASSUMPTION,
        "feature_groups": GROUPS, "history_policy": "Last eight completed own/opponent turns, missing labels remain None; current duration excluded",
        "familiarity_policy": "Earlier training games only; ambiguous same-day order excluded; validation never updates the index",
        "splits": {},
    }
    for split, rows in splits.items():
        manifest["splits"][split] = {"games": len(ids[split]), "game_ids": sorted(ids[split]), "moves": len(rows),
            "timing_labels": sum(r["elapsed_seconds"] is not None for r in rows),
            "missing_labels": sum(r["elapsed_seconds"] is None for r in rows),
            "long_labels": sum(r["elapsed_seconds"] is not None and r["elapsed_seconds"] >= 99.9 for r in rows),
            "interval_labels": sum(r["observation_interval_seconds"] is not None for r in rows),
            "unknown_timestamp_games": len({r["game_id"] for r in rows if r["timestamp_uncertain"]}),
            "timing_sources": dict(Counter(r["timing_source"] for r in rows)),
            "platforms": dict(Counter(r["platform"] for r in rows))}
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    (output / "audit.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def chronological_partition(rows: list[dict], cutoff: str, *, reserved_game_ids=(), session_gap_seconds=3600):
    """Partition new, fully timestamped games; a session crossing cutoff belongs to validation."""
    boundary = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    if boundary.tzinfo is None or session_gap_seconds < 0:
        raise ValueError("Timezone-aware cutoff and nonnegative session gap required")
    groups = defaultdict(list)
    reserved = set(reserved_game_ids)
    for row in rows:
        if row["game_id"] in reserved:
            raise ValueError("Reserved holdout game cannot be reused without a separately documented retirement")
        groups[row["game_id"]].append(row)
    ordered = []
    for game_id, rs in groups.items():
        if any(r.get("timestamp_uncertain", True) or not r.get("timestamp_utc") for r in rs):
            raise ValueError("Full UTC timestamps required for prospective/session splits")
        stamps = {r["timestamp_utc"] for r in rs}
        if len(stamps) != 1:
            raise ValueError("Inconsistent game timestamps")
        stamp = datetime.fromisoformat(next(iter(stamps)).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Timezone-aware game timestamps required")
        ordered.append((stamp, game_id, rs))
    ordered.sort(key=lambda item: (item[0], item[1]))
    sessions = []
    for item in ordered:
        if not sessions or (item[0] - sessions[-1][-1][0]).total_seconds() > session_gap_seconds:
            sessions.append([])
        sessions[-1].append(item)
    result = {"train": [], "val": []}
    for session in sessions:
        split = "train" if session[-1][0] < boundary else "val"
        for _, _, rs in session:
            result[split].extend(sorted(rs, key=lambda r: r["ply"]))
    if not result["train"] or not result["val"]:
        raise ValueError("Cutoff must produce nonempty training and validation sessions")
    return result


def validate_public_rows(rows, *, target_accounts, reserved_game_ids, cutoff):
    """Fail closed before public-prior fitting; account identities and time provenance are required."""
    targets = {str(name).strip().casefold() for name in target_accounts}
    if not targets:
        raise ValueError("Target account exclusion list is required")
    reserved = set(reserved_game_ids)
    boundary = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    if boundary.tzinfo is None:
        raise ValueError("Timezone-aware public pretraining cutoff required")
    seen = set()
    result = []
    for row in rows:
        if row["game_id"] in reserved:
            raise ValueError("Reserved personal game present in public data")
        names = [row.get("white_player"), row.get("black_player")]
        if not all(isinstance(n, str) and n.strip() for n in names):
            raise ValueError("Both account identities are required in public data")
        if targets.intersection(n.strip().casefold() for n in names):
            raise ValueError("Target account present in public pretraining data")
        if not row.get("timestamp_utc") or row.get("timestamp_uncertain", True):
            raise ValueError("Public data requires full UTC timestamps")
        stamp = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp >= boundary:
            raise ValueError("Public example is not strictly before the evaluation cutoff")
        identity = (row["game_id"], row["ply"])
        if identity in seen:
            raise ValueError("Duplicate game/ply in public data")
        seen.add(identity)
        result.append(row)
    return result


def retrieval_features(training_rows, training_embeddings, query_rows, query_embeddings, k=8):
    """Nearest pooled-feature timings from strictly earlier training games only.

    Embeddings must share checkpoint and feature schema. Distances are Euclidean
    after per-vector L2 normalization. Query labels never enter retrieval.
    """
    import numpy as np

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("Positive integer neighbor count required")
    train = np.asarray(training_embeddings, dtype=np.float64)
    query = np.asarray(query_embeddings, dtype=np.float64)
    if train.ndim != 2 or query.ndim != 2 or train.shape[1] != query.shape[1] or train.shape[1] == 0:
        raise ValueError("Training/query embeddings require a common nonempty feature dimension")
    if len(train) != len(training_rows) or len(query) != len(query_rows):
        raise ValueError("Embedding rows do not match position rows")
    if not np.isfinite(train).all() or not np.isfinite(query).all():
        raise ValueError("Nonfinite retrieval embedding")
    train = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    result = []
    for row, embedding in zip(query_rows, query):
        eligible = [i for i, other in enumerate(training_rows)
                    if other["game_id"] != row["game_id"] and _earlier(other, row)
                    and _number(other.get("elapsed_seconds"))]
        if not eligible:
            result.append({"nearest_timing_logmean": 0., "neighbor_count": 0})
            continue
        distances = np.square(train[eligible] - embedding).sum(axis=1)
        nearest = np.argsort(distances, kind="stable")[:k]
        durations = [training_rows[eligible[int(i)]]["elapsed_seconds"] for i in nearest]
        result.append({"nearest_timing_logmean": float(np.log1p(durations).mean()), "neighbor_count": len(durations)})
    return result
