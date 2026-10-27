from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import chess
import torch
from maia3.dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
from maia3.model_registry import resolve_model_spec
from maia3.models import MAIA3Model
from maia3.utils import get_all_possible_moves, mirror_move

MOVES = get_all_possible_moves()
MOVE_INDEX = {move: i for i, move in enumerate(MOVES)}
SOURCE_COMMIT = "95c0985aacb62bcaeedc30c3a9bbddfb43f91dd0"
CHECKPOINT_REVISION = "34ec2be9d651102e4ad0e9eda3d838acd69f5f9a"


def config():
    return SimpleNamespace(**resolve_model_spec("maia3-23m-ponder").config)


class PersonalModel(MAIA3Model):
    """Keep upstream computation intact while exposing its timing logits."""

    def __init__(self, cfg=None):
        super().__init__(cfg or config())
        self._timing_logits = None
        self.fc_ponder.register_forward_hook(self._capture_timing)

    def _capture_timing(self, module, inputs, output):
        self._timing_logits = output

    def forward(self, tokens, self_elos, oppo_elos):
        moves, _, seconds = super().forward(tokens, self_elos, oppo_elos)
        return moves, self._timing_logits, seconds


def load_model(path: Path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model_state_dict", payload)
    model = PersonalModel()
    model.load_state_dict({k.replace("smolgen", "gab"): v for k, v in state.items()}, strict=True)
    return model.to(device).eval()


def encode(row, cfg=None):
    cfg = cfg or config()
    board = chess.Board(row["fen"])
    history = row["history_fens"]
    if not history or history[-1] != row["fen"] or len(history) > cfg.history:
        raise ValueError("History must end at the current pre-move position, with at most 8 boards")
    # Match upstream: every historical position is oriented by its own side to move.
    tokens = get_historical_tokens(
        [tokenize_board(chess.Board(fen)) for fen in history], cfg,
        base=row["initial_seconds"], inc=row["increment_seconds"],
        clk_left_before=row["remaining_seconds"], clk_ponder=0,
    )[:, :99]
    mask = get_legal_moves_mask(board, MOVE_INDEX)
    if not mask.any():
        raise ValueError("Cannot predict a move in a terminal position")
    move = row.get("move_uci")
    target = -1
    if move is not None:
        if chess.Move.from_uci(move) not in board.legal_moves:
            raise ValueError(f"Illegal target move: {move}")
        target = MOVE_INDEX[move if board.turn else mirror_move(move)]
    elapsed = row.get("elapsed_seconds")
    # The published categorical head covers [0, 99.9) seconds. Do not clamp long thinks.
    timing_valid = elapsed is not None and 0 <= elapsed < 99.9
    bucket = min(29, max(0, int((math.log10(elapsed + .1) + 1) * 10))) if timing_valid else 0
    return {
        "tokens": tokens, "self_elos": torch.tensor(float(row["elo_self"])),
        "oppo_elos": torch.tensor(float(row["elo_oppo"])), "legal": mask,
        "target": torch.tensor(target), "time_target": torch.tensor(bucket),
        "time_valid": torch.tensor(timing_valid),
        "elapsed": torch.tensor(float(elapsed) if elapsed is not None else float("nan")),
    }


class Positions(torch.utils.data.Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"Empty dataset: {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return encode(self.rows[index])


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def predict(model, row, temperature=1.0):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    batch = to_device({k: v.unsqueeze(0) for k, v in encode(row).items()}, next(model.parameters()).device)
    with torch.no_grad():
        moves, timing, seconds = model(batch["tokens"], batch["self_elos"], batch["oppo_elos"])
        probs = torch.softmax(moves.masked_fill(~batch["legal"], -torch.inf) / temperature, -1)[0]
        time_probs = timing.softmax(-1)[0]
    board = chess.Board(row["fen"])
    candidates = []
    for index in torch.argsort(probs, descending=True).tolist():
        if not batch["legal"][0, index]:
            continue
        move = MOVES[index] if board.turn else mirror_move(MOVES[index])
        candidates.append({"uci": move, "san": board.san(chess.Move.from_uci(move)), "probability": probs[index].item()})
    return {"moves": candidates, "expected_seconds": seconds.item(), "time_probabilities": time_probs.tolist(),
            "time_bin_centers_seconds": model.ponder_bucket_centers.tolist()}
