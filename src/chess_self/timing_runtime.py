"""Clock-aware execution and simulation; model sampling is supplied by the caller."""
from __future__ import annotations

import math
from copy import deepcopy
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable

import chess
import numpy as np


CAUSAL_FIELDS = frozenset({
    "schema_version", "game_id", "date", "timestamp", "ply", "player_color", "fen",
    "history_fens", "elo_self", "elo_oppo", "initial_seconds", "increment_seconds",
    "remaining_seconds", "opponent_remaining_seconds", "time_control", "platform",
    "own_time_history", "opponent_time_history", "own_clock_known", "opponent_clock_known",
    "prior_position_games", "prior_line_games", "first_unfamiliar_position", "familiarity_known",
    "nearest_timing_logmean", "neighbor_count", "engine_score_disagreement", "engine_move_disagreement",
    "position_seen_in_train", "previous_move_was_capture", "policy_entropy", "top_two_gap",
    "opponent_surprisal", "canonical_time_control",
})


def _nonnegative(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def execution_budget(remaining_seconds, reserve_seconds=0):
    remaining = _nonnegative(remaining_seconds, "remaining_seconds")
    reserve = _nonnegative(reserve_seconds, "reserve_seconds")
    if remaining <= reserve:
        return 0.0
    # Even with zero reserve a completed move must precede the flag boundary.
    return min(remaining - reserve, math.nextafter(remaining, 0.0))


class TurnScheduler:
    """A nonblocking state machine; timestamps and waits are measured in seconds."""

    def __init__(self, now: Callable[[], float] = time.monotonic, reserve_seconds=0):
        self.now = now
        self.reserve_seconds = _nonnegative(reserve_seconds, "reserve_seconds")
        self.generation = 0
        self.active = None
        self.events = []
        self.ended = False

    def begin(self, fen, remaining_seconds):
        if self.ended:
            raise RuntimeError("Game has ended")
        chess.Board(fen)
        self.cancel("superseded")
        self.generation += 1
        self.active = {
            "token": self.generation, "fen": fen, "turn_start": self.now(),
            "pre_clock_seconds": _nonnegative(remaining_seconds, "remaining_seconds"),
            "inference_start": None, "inference_end": None,
        }
        return self.generation

    def _turn(self, token):
        if self.active is None or self.active["token"] != token:
            raise ValueError("Stale turn token")
        return self.active

    def inference_started(self, token):
        turn = self._turn(token)
        if turn["inference_start"] is not None:
            raise ValueError("Inference already started")
        turn["inference_start"] = self.now()

    def prediction_ready(self, token, move_uci, target_seconds):
        turn = self._turn(token)
        if turn["inference_end"] is not None:
            raise ValueError("Prediction already supplied")
        if turn["inference_start"] is None:
            raise ValueError("Record inference_started first")
        move = chess.Move.from_uci(move_uci)
        if move not in chess.Board(turn["fen"]).legal_moves:
            raise ValueError("Prediction contains an illegal move")
        now = self.now()
        target = _nonnegative(target_seconds, "target_seconds")
        elapsed = now - turn["turn_start"]
        if elapsed < 0:
            raise ValueError("Clock moved backwards")
        budget = execution_budget(turn["pre_clock_seconds"], self.reserve_seconds)
        effective = min(target, budget)
        turn.update(inference_end=now, move_uci=move_uci, target_seconds=target,
                    effective_target_seconds=effective, guard_intervened=target > budget,
                    execution_budget_seconds=budget)
        status = "ready" if budget > 0 and elapsed <= budget else "budget_exhausted"
        return {"token": token, "status": status, "wait_seconds": max(effective - elapsed, 0) if status == "ready" else 0,
                "guard_intervened": turn["guard_intervened"], "elapsed_seconds": elapsed,
                "execution_budget_seconds": budget}

    def commit(self, token, current_fen, increment_seconds=0, charged_seconds=None):
        """Record a commit; an authoritative charged clock may differ from wall time."""
        turn = self._turn(token)
        if turn["inference_end"] is None:
            raise ValueError("No prediction ready")
        if current_fen != turn["fen"]:
            self.cancel("position_changed")
            raise ValueError("Position changed before commit")
        now = self.now()
        elapsed = now - turn["turn_start"]
        if elapsed < turn["effective_target_seconds"]:
            raise ValueError("Requested delay has not elapsed")
        inc = _nonnegative(increment_seconds, "increment_seconds")
        charged = elapsed if charged_seconds is None else _nonnegative(charged_seconds, "charged_seconds")
        budget = turn["execution_budget_seconds"]
        # Latency can overrun a safe sample; do not commit after its execution deadline.
        status = "committed" if budget > 0 and elapsed <= budget and charged < turn["pre_clock_seconds"] else "budget_exhausted"
        event = dict(turn, status=status, commit_time=now if status == "committed" else None,
                     observed_at=now, visible_delay_seconds=elapsed, charged_seconds=charged,
                     latency_floor_seconds=turn["inference_end"] - turn["turn_start"],
                     target_unattainable=turn["inference_end"] - turn["turn_start"] > turn["target_seconds"],
                     post_clock_seconds=max(turn["pre_clock_seconds"] - charged, 0) + (inc if status == "committed" else 0))
        self.events.append(event)
        self.active = None
        return event

    def cancel(self, reason="cancelled"):
        if self.active is not None:
            self.events.append(dict(self.active, status="cancelled", reason=reason, observed_at=self.now()))
            self.active = None

    def takeback(self):
        self.cancel("takeback")

    def end_game(self):
        self.cancel("game_ended")
        self.ended = True


def _prediction(predictor, board, row):
    prediction = dict(predictor(board.copy(stack=True), deepcopy(row)))
    if "target_seconds" not in prediction:
        raise ValueError("Predictor must return target_seconds")
    prediction["target_seconds"] = _nonnegative(prediction["target_seconds"], "target_seconds")
    prediction["inference_seconds"] = _nonnegative(prediction.get("inference_seconds", 0), "inference_seconds")
    if not prediction.get("timeout", False):
        move = chess.Move.from_uci(prediction["move_uci"])
        if move not in board.legal_moves:
            raise ValueError("Predictor returned an illegal move")
    return prediction


def _summary(events):
    completed = [e for e in events if e["status"] == "completed"]
    own = defaultdict(list)
    for event in completed:
        own[(event.get("game_id"), event["player_color"])].append(event)
    x, y = [], []
    for turns in own.values():
        for a, b in zip(turns, turns[1:]):
            if b["ply"] - a["ply"] == 2:
                x.append(math.log1p(a["charged_seconds"]))
                y.append(math.log1p(b["charged_seconds"]))
    correlation = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0 else None
    entries = {}
    for event in events:
        key = str(event.get("game_id", "virtual")) + "/" + event["player_color"]
        if event["pre_clock_seconds"] < 10:
            entries.setdefault(key, event["ply"])
    return {
        "turns": len(events), "completed_moves": len(completed),
        "flags": sum(e["status"] == "flagged" for e in events),
        "guard_interventions": sum(bool(e.get("guard_intervened")) for e in events),
        "latency_floor_misses": sum(e.get("inference_seconds", 0) > e["target_seconds"] for e in events),
        "first_under_10s_ply": entries, "log_time_lag1_correlation": correlation,
        "log_time_lag1_pairs": len(x),
    }


def recorded_replay(rows: Iterable[dict], predictor, *, reserve_seconds=0, guard=True):
    """Independent historical states; never feed a target move or elapsed label in."""
    events = []
    for source in rows:
        row = {key: value for key, value in source.items() if key in CAUSAL_FIELDS}
        board = chess.Board(row["fen"])
        if board.is_game_over():
            continue
        clock = _nonnegative(row["remaining_seconds"], "remaining_seconds")
        unavailable = clock == 0 or (guard and execution_budget(clock, reserve_seconds) == 0)
        prediction = ({"target_seconds": 0.0, "inference_seconds": 0.0} if unavailable
                      else _prediction(predictor, board, row))
        target = prediction["target_seconds"]
        budget = execution_budget(clock, reserve_seconds)
        effective = min(target, budget) if guard else target
        status = "flagged" if prediction.get("timeout") or clock == 0 or effective >= clock else "completed"
        if guard and budget == 0 and status == "completed":
            status = "no_feasible_budget"
        charged = clock if status == "flagged" else effective
        events.append({"game_id": row.get("game_id"), "ply": row.get("ply", board.ply() + 1),
                       "player_color": "white" if board.turn else "black", "move_uci": prediction.get("move_uci"),
                       "status": status, "pre_clock_seconds": clock, "target_seconds": target,
                       "effective_target_seconds": effective, "charged_seconds": charged,
                       "visible_delay_seconds": max(effective, prediction["inference_seconds"]),
                       "inference_seconds": prediction["inference_seconds"],
                       "observed_seconds": source.get("elapsed_seconds"),
                       "guard_intervened": guard and target > budget,
                       "removed_probability_mass": prediction.get("removed_probability_mass")})
    return {"mode": "recorded_state_replay", "clock_target": "recorded_clock_consumption",
            "summary": _summary(events), "events": events}


def virtual_game(predictor, initial_seconds, increment_seconds=0, *, starting_fen=chess.STARTING_FEN,
                 ratings=(1500, 1500), max_plies=400, reserve_seconds=0, guard=True,
                 first_move_rule="charge_all", clock_charge_mode="target", game_id="virtual"):
    """Generate a whole game; predictor sees generated moves, clocks and pace only.

    predictor can be one callable or a {chess.WHITE: callable, chess.BLACK: callable}
    map. target mode simulates recorded clock consumption; wall mode also charges
    declared inference latency. Neither mode sleeps or claims measured browser latency.
    """
    initial = _nonnegative(initial_seconds, "initial_seconds")
    increment = _nonnegative(increment_seconds, "increment_seconds")
    _nonnegative(reserve_seconds, "reserve_seconds")
    if first_move_rule not in {"charge_all", "free_first_turn"}:
        raise ValueError("Choose charge_all or free_first_turn explicitly")
    if clock_charge_mode not in {"target", "wall"}:
        raise ValueError("clock_charge_mode must be target or wall")
    if not isinstance(max_plies, int) or max_plies < 1:
        raise ValueError("max_plies must be a positive integer")
    board = chess.Board(starting_fen)
    if not board.is_valid():
        raise ValueError("Invalid initial board")
    clocks = {chess.WHITE: initial, chess.BLACK: initial}
    times = {chess.WHITE: deque(maxlen=8), chess.BLACK: deque(maxlen=8)}
    turns = {chess.WHITE: 0, chess.BLACK: 0}
    history = deque(maxlen=8)
    events = []
    previous_move_was_capture = False
    reason = "ply_limit"
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            reason = "board_terminal"
            break
        color = board.turn
        history.append(board.fen())
        row = {
            "game_id": game_id, "fen": board.fen(), "history_fens": list(history),
            "ply": board.ply() + 1, "player_color": "white" if color else "black",
            "initial_seconds": initial, "increment_seconds": increment,
            "remaining_seconds": clocks[color], "opponent_remaining_seconds": clocks[not color],
            "own_time_history": list(times[color]), "opponent_time_history": list(times[not color]),
            "elo_self": ratings[0 if color else 1], "elo_oppo": ratings[1 if color else 0],
            "time_control": f"{initial:g}+{increment:g}", "platform": "simulation",
            "previous_move_was_capture": previous_move_was_capture,
        }
        fn = predictor[color] if isinstance(predictor, dict) else predictor
        unavailable = clocks[color] == 0 or (guard and execution_budget(clocks[color], reserve_seconds) == 0)
        prediction = ({"target_seconds": 0.0, "inference_seconds": 0.0} if unavailable
                      else _prediction(fn, board, row))
        target, latency = prediction["target_seconds"], prediction["inference_seconds"]
        budget = execution_budget(clocks[color], reserve_seconds)
        effective = min(target, budget) if guard else target
        visible = max(effective, latency)
        charged = visible if clock_charge_mode == "wall" else effective
        if first_move_rule == "free_first_turn" and turns[color] == 0:
            charged = 0.0
        status = "completed"
        if prediction.get("timeout") or clocks[color] == 0 or charged >= clocks[color]:
            status = "flagged"
            charged = clocks[color]
        elif guard and (budget == 0 or (clock_charge_mode == "wall" and visible > budget)):
            status = "no_feasible_budget"
        event = {"game_id": game_id, "ply": row["ply"], "player_color": row["player_color"],
                 "fen": row["fen"], "move_uci": prediction.get("move_uci"), "target_seconds": target,
                 "effective_target_seconds": effective, "visible_delay_seconds": visible,
                 "inference_seconds": latency, "charged_seconds": charged,
                 "pre_clock_seconds": clocks[color], "opponent_clock_seconds": clocks[not color],
                 "guard_intervened": guard and target > budget, "status": status,
                 "removed_probability_mass": prediction.get("removed_probability_mass")}
        if status != "completed":
            event["post_clock_seconds"] = max(clocks[color] - charged, 0)
            clocks[color] = event["post_clock_seconds"]
            events.append(event)
            reason = status
            break
        clocks[color] = clocks[color] - charged + increment
        event["post_clock_seconds"] = clocks[color]
        events.append(event)
        previous_move_was_capture = board.is_capture(chess.Move.from_uci(prediction["move_uci"]))
        board.push_uci(prediction["move_uci"])
        times[color].append(charged)
        turns[color] += 1
    else:
        if board.is_game_over(claim_draw=True):
            reason = "board_terminal"
    return {"mode": "virtual_full_game", "first_move_rule": first_move_rule,
            "clock_charge_mode": clock_charge_mode, "termination": reason,
            "board_result": board.result(claim_draw=True), "final_fen": board.fen(),
            "final_clocks": {"white": clocks[chess.WHITE], "black": clocks[chess.BLACK]},
            "summary": _summary(events), "events": events}
