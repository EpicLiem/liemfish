"""Native move/time joint generation from a versioned timing experiment."""
from pathlib import Path
import time
import chess
import torch

from .model import MOVES, encode, load_model
from maia3.utils import mirror_move
from .timing_cache import digest
from .timing_model import MaiaFeatures, load_timing_head
from .timing_experiments import live_features, forward_head, side_inputs
from .timing_distributions import make_distribution


class TimingPredictor:
    def __init__(self, checkpoint, head=None, *, calibration=None, event=None, mode='sample',
                 feasible=True, reserve_seconds=0., device='cpu', seed=42):
        if mode not in {'sample', 'mean', 'median'} or reserve_seconds < 0:
            raise ValueError('Invalid playback policy')
        self.extractor = MaiaFeatures(load_model(checkpoint, device)).eval()
        self.head = None; self.groups = []
        sha = digest(checkpoint)
        if head:
            self.head, payload = load_timing_head(head, device=device, base_checkpoint_sha256=sha)
            self.groups = payload['extra']['feature_groups']
            from .timing_data import feature_names
            if feature_names(self.groups) != payload['feature_names']:
                raise ValueError('Checkpoint feature ordering mismatch')
        self.calibrator = None
        if calibration:
            if head: raise ValueError('Calibrator is bound to original absolute head; refit for an adapted head')
            from .timing_calibration import Calibration
            payload = torch.load(calibration, map_location=device, weights_only=True)
            if payload['base_checkpoint_sha256'] != sha: raise ValueError('Calibration checkpoint mismatch')
            self.calibrator = Calibration(payload['mode'], payload['side_dim']).to(device)
            self.calibrator.load_state_dict(payload['state_dict'], strict=True)
            self.calibration_groups = payload['feature_groups']
        self.event = None
        if event:
            payload = torch.load(event, map_location=device, weights_only=True)
            from .timing_data import feature_names
            self.event_groups = ['clock', 'phase', 'chess', 'history_summary']
            if payload.get('architecture') != 'timing-timeout-event-v1' or payload['features'] != feature_names(self.event_groups):
                raise ValueError('Event model schema mismatch')
            self.event = torch.nn.Linear(len(payload['features']), 1).to(device)
            self.event.load_state_dict(payload['state_dict'], strict=True)
        self.device, self.mode, self.feasible, self.reserve = device, mode, feasible, reserve_seconds
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def __call__(self, board, row):
        started = time.perf_counter()
        row = dict(row)
        if board.fen() != row['fen']: raise ValueError('Board and prediction context differ')
        budget = row['remaining_seconds'] - self.reserve
        if budget <= 0: return {'timeout': True, 'target_seconds': 0.}
        # encode permits no target move; generated moves are selected below.
        row.pop('move_uci', None)
        with torch.inference_mode():
            features = live_features(self.extractor, [row], self.device)
            policy = features['move_logits'].softmax(-1)[0]
            index = int(torch.multinomial(policy, 1, generator=self.generator))
            move = MOVES[index] if board.turn else mirror_move(MOVES[index])
            if self.event is not None:
                probability = self.event(side_inputs([row], self.event_groups, self.device)).sigmoid().item()
                if torch.rand((), device=self.device, generator=self.generator).item() < probability:
                    return {'move_uci': move, 'timeout': True, 'target_seconds': row['remaining_seconds'], 'timeout_probability': probability}
            if self.head:
                raw = forward_head(self.head, features, [row], self.groups, [move])
                family = self.head.config.output_family
            else:
                raw, family = features['base_logits'], 'absolute'
            if self.calibrator:
                raw = self.calibrator(raw, side_inputs([row], self.calibration_groups, self.device))
            dist = make_distribution(raw, family, raw.new_tensor([row['remaining_seconds']]))
            removed = float(1 - dist.cdf(budget)[0])
            if self.feasible: dist = dist.truncate(budget)
            target = dist.sample(self.generator) if self.mode == 'sample' else dist.mean() if self.mode == 'mean' else dist.quantile(.5)
        return {'move_uci': move, 'target_seconds': float(target[0]),
                'removed_probability_mass': removed if self.feasible else 0.,
                'inference_seconds':time.perf_counter()-started}
