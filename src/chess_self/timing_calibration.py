"""Small calibration fits on predictions from games the predictor never fitted."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from .timing_cache import FeatureCache, assert_comparable, fresh_directory, write_json
from .timing_data import feature_names
from .timing_experiments import side_inputs, timing_objective


class Calibration(nn.Module):
    def __init__(self, mode='temperature', side_dim=0, histogram=None):
        super().__init__()
        if mode not in {'temperature', 'bias', 'regime', 'histogram'}:
            raise ValueError('Unknown calibration mode')
        self.mode = mode
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(30), requires_grad=mode == 'bias')
        self.regime = nn.Linear(side_dim, 30, bias=False) if mode == 'regime' else None
        if self.regime is not None: nn.init.zeros_(self.regime.weight)
        self.mix_logit = nn.Parameter(torch.tensor(-4.), requires_grad=mode == 'histogram')
        hist = torch.ones(30) / 30 if histogram is None else histogram
        if hist.shape != (30,) or not torch.isfinite(hist).all() or (hist <= 0).any():
            raise ValueError('Histogram must contain thirty positive probabilities')
        self.register_buffer('histogram', hist / hist.sum())

    def forward(self, logits, side=None):
        result = logits / self.log_temperature.clamp(-4, 4).exp()
        if self.mode == 'bias': result = result + self.bias
        if self.regime is not None: result = result + self.regime(side)
        if self.mode == 'histogram':
            mix = self.mix_logit.sigmoid()
            result = ((1 - mix) * result.softmax(-1) + mix * self.histogram).log()
        return result


def assert_unseen_predictor_games(cache):
    ids = cache.manifest.get('predictor_training_game_ids')
    if ids is None:
        raise ValueError('Calibration requires explicit predictor training-game provenance; use held-out or chronological OOF predictions')
    if {r['game_id'] for r in cache.rows} & set(ids):
        raise ValueError('Calibration games were fitted by the predictor; in-sample calibration is prohibited')


def fit_calibration(fit_cache, val_cache, output, *, mode='temperature', epochs=30,
                    learning_rate=.01, regularization=.01, batch_size=64, seed=42):
    from .timing_distributions import make_distribution
    from .timing_metrics import scorecard
    fit, val = FeatureCache(fit_cache), FeatureCache(val_cache)
    assert_comparable(fit, val); assert_unseen_predictor_games(fit); assert_unseen_predictor_games(val)
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or regularization < 0:
        raise ValueError('Invalid calibration settings')
    output = fresh_directory(output); torch.manual_seed(seed)
    groups = ['clock', 'phase']
    hist = torch.ones(30)
    for r in fit.rows:
        y = r.get('elapsed_seconds')
        if y is not None and 0 <= y < 99.9:
            index = min(29, int((__import__('math').log10(y + .1) + 1) * 10))
            hist[index] += 1
    model = Calibration(mode, len(feature_names(groups)), hist / hist.sum())
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    for epoch in range(epochs):
        for ids in torch.randperm(len(fit)).split(batch_size):
            ids = ids.tolist(); rows = [fit.rows[i] for i in ids]
            raw = model(fit.batch(ids)['base_logits'], side_inputs(rows, groups, 'cpu'))
            loss, used, _ = timing_objective(raw, rows, 'absolute')
            if not used: continue
            penalty = model.bias.square().mean()
            if model.regime is not None: penalty = penalty + model.regime.weight.square().mean()
            loss = loss + regularization * penalty
            opt.zero_grad(); loss.backward(); opt.step()
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite calibration objective')
            losses.append(float(loss.detach()))
    if not losses: raise ValueError('No supported calibration observations')
    outputs = []; originals = []
    with torch.no_grad():
        for start in range(0, len(val), batch_size):
            ids = list(range(start, min(start + batch_size, len(val))))
            rows = [val.rows[i] for i in ids]; raw = val.batch(ids)['base_logits']
            originals.append(raw); outputs.append(model(raw, side_inputs(rows, groups, 'cpu')))
    clocks = torch.tensor([r['remaining_seconds'] for r in val.rows])
    report = {'architecture': 'timing-calibration-v1', 'mode': mode, 'epochs': epochs, 'seed': seed,
              'regularization': regularization, 'test_evaluated': False,
              'fit_cache': fit.manifest, 'validation_cache': val.manifest,
              'baseline': scorecard(make_distribution(torch.cat(originals), 'absolute', clocks).truncate(clocks), val.rows),
              'calibrated': scorecard(make_distribution(torch.cat(outputs), 'absolute', clocks).truncate(clocks), val.rows)}
    torch.save({'architecture': report['architecture'], 'mode': mode, 'side_dim': len(feature_names(groups)),
                'feature_groups': groups, 'state_dict': model.state_dict(),
                'base_checkpoint_sha256': fit.manifest['checkpoint_sha256'],
                'fit_game_ids': sorted({r['game_id'] for r in fit.rows})}, output / 'calibration.pt')
    write_json(output / 'run.json', report)
    return report
