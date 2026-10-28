"""Versioned timing side heads; Maia's policy and encoder retain their contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import chess
import torch
from torch import nn
from torch.nn import functional as F

from .model import MOVE_INDEX, MOVES
from maia3.utils import mirror_move

ARCHITECTURE = "maia3-timing-side-v1"
EXTENDED_PRIOR = "normalized-original-log-probabilities-v1"
MOVE_ATTRIBUTE_NAMES = ("promotion_q", "promotion_r", "promotion_b", "promotion_n",
                        "capture", "gives_check", "castling", "policy_surprisal")


@dataclass(frozen=True)
class TimingHeadConfig:
    side_dim: int
    pooled_dim: int = 512
    hidden_dim: int = 64
    output_family: str = "absolute"
    output_bins: int = 30
    use_pooled: bool = True
    move_mode: str = "none"
    history_mode: str = "none"
    history_channels: int = 4
    history_length: int = 8

    def __post_init__(self):
        if self.side_dim < 0 or min(self.pooled_dim, self.hidden_dim, self.history_channels, self.history_length) < 1:
            raise ValueError("Invalid feature dimensions")
        if self.output_family not in {"absolute", "extended", "fraction", "beta"}:
            raise ValueError("Unknown output family")
        if self.output_family == "absolute" and self.output_bins != 30:
            raise ValueError("Original absolute bins must retain all 30 meanings")
        if self.output_family == "extended" and self.output_bins != 40:
            raise ValueError("Extended bins must retain the original 30 and add ten tail bins")
        if self.output_family == "fraction" and self.output_bins != 30:
            raise ValueError("Fraction output uses 30 bins")
        if self.output_bins < 2:
            raise ValueError("Need at least two output bins")
        if self.move_mode not in {"none", "attributes", "squares"}:
            raise ValueError("Unknown move conditioning")
        if self.history_mode not in {"none", "gru"}:
            raise ValueError("Unknown history encoder")
        if not self.use_pooled and not self.side_dim and self.move_mode == "none" and self.history_mode == "none":
            raise ValueError("A timing head needs at least one input feature")


class MaiaFeatures(nn.Module):
    """Expose original forward representations without reproducing Maia computation."""

    def __init__(self, model, final_blocks=0):
        super().__init__()
        if not 0 <= final_blocks <= min(2, len(model.transformer.layers)):
            raise ValueError("Only zero, one, or two final blocks may be adapted")
        self.model = model
        self.final_blocks = final_blocks
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if final_blocks:
            for layer in model.transformer.layers[-final_blocks:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
        self._captured = {}
        self._hooks = [model.transformer.register_forward_hook(self._squares),
                       model.last_ln.register_forward_hook(self._pooled)]
        self.model.eval()

    def _squares(self, module, inputs, output):
        self._captured["squares"] = output[:, :64]

    def _pooled(self, module, inputs, output):
        self._captured["pooled"] = output

    def train(self, mode=True):
        super().train(mode)
        # Fixed representations and teacher predictions must not acquire dropout noise.
        self.model.eval()
        if self.final_blocks:
            for layer in self.model.transformer.layers[-self.final_blocks:]:
                layer.train(mode)
        return self

    def forward(self, tokens, self_elos, oppo_elos):
        self._captured = {}
        move, timing, _ = self.model(tokens, self_elos, oppo_elos)
        return {**self._captured, "base_logits": timing, "move_logits": move}

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def parameter_groups(self, head, head_lr, backbone_lr):
        if head_lr <= 0 or backbone_lr <= 0 or (self.final_blocks and backbone_lr >= head_lr):
            raise ValueError("Adapted backbone learning rate must be positive and lower than head rate")
        groups = [{"params": list(head.parameters()), "lr": head_lr, "name": "timing_head"}]
        shared = [p for p in self.model.parameters() if p.requires_grad]
        if shared:
            groups.append({"params": shared, "lr": backbone_lr, "name": "backbone"})
        return groups


class TimingHead(nn.Module):
    def __init__(self, config: TimingHeadConfig):
        super().__init__()
        self.config = config
        dim = config.side_dim + (config.pooled_dim if config.use_pooled else 0)
        if config.move_mode != "none":
            dim += len(MOVE_ATTRIBUTE_NAMES)
        if config.move_mode == "squares":
            dim += 2 * config.pooled_dim
        self.history_encoder = None
        if config.history_mode == "gru":
            self.history_encoder = nn.GRU(config.history_channels, config.hidden_dim, batch_first=True)
            dim += config.hidden_dim
        self.hidden = nn.Linear(dim, config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, 2 if config.output_family == "beta" else config.output_bins)
        if config.output_family in {"absolute", "extended"}:
            # Preserve the original conditional bin probabilities; tail rows start with small mass.
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
            if config.output_family == "extended":
                with torch.no_grad():
                    self.output.bias[30:] = -12

    def forward(self, pooled, base_logits, side_features, squares=None,
                move_indices=None, move_attributes=None, history=None):
        cfg = self.config
        parts = [side_features]
        if cfg.use_pooled:
            parts.append(pooled)
        if cfg.move_mode != "none":
            if move_attributes is None:
                raise ValueError("Selected move attributes are required")
            parts.append(move_attributes)
        if cfg.move_mode == "squares":
            if squares is None or move_indices is None:
                raise ValueError("Selected square indices and representations are required")
            selected = torch.gather(squares, 1, move_indices.unsqueeze(-1).expand(-1, -1, cfg.pooled_dim))
            parts.append(selected.flatten(1))
        if self.history_encoder is not None:
            if history is None:
                raise ValueError("History values and missingness masks are required")
            _, state = self.history_encoder(history)
            parts.append(state[-1])
        raw = self.output(F.relu(self.hidden(torch.cat(parts, dim=-1))))
        if cfg.output_family == "absolute":
            return base_logits + raw
        if cfg.output_family == "extended":
            return torch.cat((F.log_softmax(base_logits, dim=-1) + raw[:, :30], raw[:, 30:]), dim=-1)
        return raw

    def metadata(self):
        initialization = "new output semantics; no old output rows reused"
        if self.config.output_family == "absolute":
            initialization = "zero residual on original 30 logits"
        elif self.config.output_family == "extended":
            initialization = "zero residual on normalized original 30 log probabilities; ten tail logits -12"
        return {"architecture": ARCHITECTURE, "config": asdict(self.config),
                "new_modules": [name for name, _ in self.named_parameters()],
                "extended_prior": EXTENDED_PRIOR if self.config.output_family == "extended" else None,
                "initialization": initialization}


def selected_move_inputs(rows, policy_logits, moves: Sequence[str] | None = None):
    """Human moves are for conditional scoring only; pass generated moves in playback."""
    if len(rows) != len(policy_logits) or (moves is not None and len(moves) != len(rows)):
        raise ValueError("Rows, logits and selected moves must have equal batch length")
    indices, attributes = [], []
    for i, row in enumerate(rows):
        board = chess.Board(row["fen"])
        move = chess.Move.from_uci(row["move_uci"] if moves is None else moves[i])
        if move not in board.legal_moves:
            raise ValueError(f"Selected move is illegal: {move}")
        oriented = move if board.turn else chess.Move.from_uci(mirror_move(move.uci()))
        legal = [MOVE_INDEX[m.uci() if board.turn else mirror_move(m.uci())] for m in board.legal_moves]
        logp = policy_logits[i, MOVE_INDEX[oriented.uci()]] - torch.logsumexp(policy_logits[i, legal], dim=0)
        indices.append([oriented.from_square, oriented.to_square])
        attrs = [float(move.promotion == piece) for piece in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)]
        attrs += [float(board.is_capture(move)), float(board.gives_check(move)), float(board.is_castling(move))]
        attributes.append(torch.cat((policy_logits.new_tensor(attrs), -logp.reshape(1))))
    return (torch.tensor(indices, dtype=torch.long, device=policy_logits.device), torch.stack(attributes))


def history_tensor(rows, length=8, *, device="cpu"):
    """Chronological channels: log1p own, log1p opponent, own mask, opponent mask."""
    result = torch.zeros(len(rows), length, 4, device=device)
    for i, row in enumerate(rows):
        for column, name in enumerate(("own_time_history", "opponent_time_history")):
            values = row.get(name, [])[-length:]
            for j, value in enumerate(values, length - len(values)):
                if value is not None:
                    if value < 0:
                        raise ValueError("Prior durations must be nonnegative")
                    result[i, j, column] = torch.log1p(torch.tensor(float(value)))
                    result[i, j, column + 2] = 1
    return result


def save_timing_head(path, head, *, base_checkpoint_sha256, feature_names, extra=None):
    path = Path(path)
    if len(feature_names) != head.config.side_dim or len(set(feature_names)) != len(feature_names):
        raise ValueError("Ordered feature schema does not match side input dimension")
    payload = {**head.metadata(), "state_dict": head.state_dict(),
               "base_checkpoint_sha256": base_checkpoint_sha256, "feature_names": list(feature_names),
               "extra": extra or {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_timing_head(path, *, device="cpu", base_checkpoint_sha256=None, feature_names=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("architecture") != ARCHITECTURE:
        raise ValueError("Unsupported timing architecture")
    if payload["config"]["output_family"] == "extended" and payload.get("extended_prior") != EXTENDED_PRIOR:
        raise ValueError("Legacy extended timing prior is not normalized; refit before using this implementation")
    if base_checkpoint_sha256 is not None and payload["base_checkpoint_sha256"] != base_checkpoint_sha256:
        raise ValueError("Timing head belongs to another backbone checkpoint")
    if feature_names is not None and list(feature_names) != payload["feature_names"]:
        raise ValueError("Ordered timing feature schema mismatch")
    head = TimingHead(TimingHeadConfig(**payload["config"]))
    if len(payload["feature_names"]) != head.config.side_dim:
        raise ValueError("Invalid checkpoint feature schema")
    head.load_state_dict(payload["state_dict"], strict=True)
    return head.to(device).eval(), payload
