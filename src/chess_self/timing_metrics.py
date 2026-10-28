"""Shared seconds-scale scorecard for timing distributions and paired game uncertainty."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import torch


def canonical_control(row):
    return f"{row['initial_seconds']:g}+{row['increment_seconds']:g}"


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def paired_game_bootstrap(rows, left, right, repetitions=2000, seed=42):
    """CI for mean(left-right), resampling whole games on identical finite pairs."""
    if repetitions < 1 or len(rows) != len(left) or len(rows) != len(right):
        raise ValueError("Bootstrap requires paired rows and a positive resample count")
    left, right = np.asarray(left, float), np.asarray(right, float)
    paired = np.isfinite(left) & np.isfinite(right)
    coverage = {"input_positions": len(rows), "excluded_pairs": int((~paired).sum()),
                "left_nonfinite": int((~np.isfinite(left)).sum()),
                "right_nonfinite": int((~np.isfinite(right)).sum()),
                "population": "common_finite_pairs",
                "whole_sample_difference_defined": bool(paired.all())}
    sums, counts = defaultdict(float), defaultdict(int)
    for row, a, b in zip(rows, left, right):
        if np.isfinite(a) and np.isfinite(b):
            sums[row['game_id']] += a - b
            counts[row['game_id']] += 1
    if not counts:
        return {"difference": None, "ci95": None, "games": 0, "positions": 0, **coverage}
    keys = sorted(counts)
    totals = np.array([sums[k] for k in keys]); sizes = np.array([counts[k] for k in keys])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), (repetitions, len(keys)))
    estimates = totals[draws].sum(-1) / sizes[draws].sum(-1)
    return {"difference": float(totals.sum() / sizes.sum()), "ci95": np.quantile(estimates, [.025, .975]).tolist(),
            "games": len(keys), "positions": int(sizes.sum()), "resamples": repetitions, "seed": seed, **coverage}


def scorecard(distribution, rows, *, deployment_weights=None, event_thresholds=(2., 10., 20.),
              move_probabilities=None, move_targets=None, crps_points=256):
    """Use explicit observation intervals; missing intervals are counted, never invented.

    NLL is the probability of a recorded interval in seconds for every family.
    Any zero probability makes aggregate interval_nll null and increments its
    zero_probability count. finite_interval_nll is diagnostic, not the full score.
    """
    if len(rows) != len(distribution.reference) or not rows:
        raise ValueError("Rows must match nonempty distribution batch")
    device, dtype = distribution.reference.device, distribution.reference.dtype
    def tensor(values):
        return torch.as_tensor(values, device=device, dtype=dtype)
    y = tensor([r.get('elapsed_seconds') if r.get('elapsed_seconds') is not None else float('nan') for r in rows])
    labeled = torch.isfinite(y) & (y >= 0)
    safe_y = torch.where(labeled, y, 0)
    low, high, available = [], [], []
    for row in rows:
        interval = row.get('observation_interval_seconds')
        if interval is None:
            low.append(0.); high.append(0.); available.append(False)
        else:
            a, b = interval
            if not math.isfinite(a) or not math.isfinite(b) or not 0 <= a < b:
                raise ValueError("Observation interval must be finite, nonnegative and nonempty")
            low.append(a); high.append(b); available.append(True)
    available = torch.tensor(available, dtype=torch.bool, device=device) & labeled
    clock = tensor([r['remaining_seconds'] for r in rows])
    with torch.no_grad():
        mean, median = distribution.mean(), distribution.quantile(.5)
        nll = -distribution.log_interval_probability(tensor(low), tensor(high))
        crps = distribution.crps(safe_y, points=crps_points)
        interval_bounds = {coverage: (distribution.quantile((1-coverage)/2), distribution.quantile((1+coverage)/2)) for coverage in (.5, .8, .9)}
        event_prob = {float(t): distribution.interval_probability(torch.minimum(torch.full_like(distribution.upper, float(t)), distribution.upper), distribution.upper) for t in event_thresholds}
        impossible_mass = distribution.interval_probability(torch.minimum(clock, distribution.upper), distribution.upper)
    def array(t):
        return t.detach().cpu().numpy()
    y, labeled, available, mean, median, nll, crps, clock, impossible_mass = map(array, (y, labeled, available, mean, median, nll, crps, clock, impossible_mass))
    interval_bounds = {c: tuple(map(array, bounds)) for c, bounds in interval_bounds.items()}
    event_prob = {t: array(p) for t, p in event_prob.items()}
    removed = array(distribution.removed_mass) if hasattr(distribution, 'removed_mass') else np.zeros(len(rows))
    upper, lower = array(distribution.upper), array(distribution.lower)
    moves = None
    if move_probabilities is not None:
        probabilities = np.asarray(move_probabilities.detach().cpu() if isinstance(move_probabilities, torch.Tensor) else move_probabilities)
        targets = np.asarray(move_targets.detach().cpu() if isinstance(move_targets, torch.Tensor) else move_targets)
        if probabilities.ndim != 2 or probabilities.shape[0] != len(rows) or targets.shape != (len(rows),):
            raise ValueError("Move probabilities and target indices must match rows")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(-1), 1, atol=1e-5):
            raise ValueError("Move probabilities must be normalized")
        if ((targets < 0) | (targets >= probabilities.shape[1])).any():
            raise ValueError("Move target out of vocabulary")
        chosen = probabilities[np.arange(len(rows)), targets]
        order = np.argsort(-probabilities, axis=-1)
        with np.errstate(divide='ignore'):
            moves = (-np.log(chosen), (order[:, 0] == targets), (order[:, :3] == targets[:, None]).any(-1))
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups['all'].append(i)
        groups['control/' + canonical_control(row)].append(i)
        remaining = row['remaining_seconds']
        groups['clock/' + ('under_10s' if remaining < 10 else '10_to_30s' if remaining < 30 else '30s_plus')].append(i)
        ply = row.get('ply', 0)
        groups['phase/' + ('opening' if ply <= 20 else 'middle' if ply <= 80 else 'late')].append(i)
        groups['platform/' + (row.get('platform') or 'unknown')].append(i)
        seen = row.get('position_seen_in_train')
        if seen is not None:
            groups['position/' + ('repeated' if seen else 'unseen')].append(i)
    def average(values):
        return _number(np.mean(values)) if len(values) else None
    results = {}
    for name, indices in groups.items():
        ix = np.array(indices); timed = ix[labeled[ix]]; scored = ix[available[ix]]
        finite = scored[np.isfinite(nll[scored])]
        common = scored[(y[scored] >= 0) & (y[scored] < 99.9)]
        zero_count = int((~np.isfinite(nll[scored])).sum())
        positive_clock = timed[clock[timed] > 0]
        result = {
            'positions': len(ix), 'games': len({rows[i]['game_id'] for i in ix}),
            'timing_labels': len(timed), 'interval_labels': len(scored),
            'invalid_negative_labels': int((y[ix] < 0).sum()),
            'missing_observation_intervals': int((labeled[ix] & ~available[ix]).sum()),
            'outside_model_support': int(((y[timed] < lower[timed]) | (y[timed] >= upper[timed])).sum()),
            'mean_mae_seconds': average(np.abs(mean[timed]-y[timed])),
            'median_mae_seconds': average(np.abs(median[timed]-y[timed])),
            'median_log1p_mae': average(np.abs(np.log1p(median[timed])-np.log1p(y[timed]))),
            'mean_predicted_seconds': average(mean[timed]), 'mean_observed_seconds': average(y[timed]),
            'mean_bias_seconds': average(mean[timed]-y[timed]),
            'mean_clock_share_bias': average((mean[positive_clock]-y[positive_clock])/clock[positive_clock]),
            'crps_seconds': average(crps[timed]),
            'interval_nll': None if zero_count else average(nll[scored]),
            'zero_probability_intervals': zero_count,
            'finite_interval_nll': average(nll[finite]),
            'common_support_interval_nll': average(nll[common]),
            'common_support_labels': len(common),
            'impossible_duration_mass': average(impossible_mass[ix]),
            'removed_mass': average(removed[ix]), 'coverage': {}, 'events': {},
        }
        for c, (a,b) in interval_bounds.items():
            result['coverage'][str(c)] = {'observed': average((y[timed] >= a[timed]) & (y[timed] <= b[timed])), 'mean_width_seconds': average((b-a)[timed])}
        for t,p in event_prob.items():
            result['events'][str(t)] = {'predicted_ge': average(p[timed]), 'observed_ge': average(y[timed] >= t), 'brier': average((p[timed]-(y[timed]>=t))**2)}
        if moves is not None:
            result['move_nll'], result['move_top1'], result['move_top3'] = [average(v[ix]) for v in moves]
        results[name] = result
    controls = {k.removeprefix('control/'):v for k,v in results.items() if k.startswith('control/')}
    scalar_metrics = ('mean_mae_seconds','median_mae_seconds','crps_seconds','interval_nll','mean_bias_seconds','impossible_duration_mass')
    def weighted(weights):
        if not weights or any(k not in controls for k in weights) or any(not math.isfinite(v) or v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("Control weights must be nonnegative and reference observed controls")
        active = {k:v for k,v in weights.items() if v > 0}
        return {metric: None if any(controls[k][metric] is None for k in active) else sum(v*controls[k][metric] for k,v in active.items())/sum(active.values()) for metric in scalar_metrics}
    output = {'groups': results, 'macro_control': weighted({k:1 for k in controls}),
              'observation_semantics': 'Recorded row observation intervals; absent intervals excluded and counted.',
              'crps_method': f'Midpoint quantile quadrature, {crps_points} points; seconds',
              'per_row': [{'game_id':r['game_id'], 'ply':r.get('ply'), 'mean_seconds':_number(mean[i]), 'median_seconds':_number(median[i]), 'mean_absolute_error':_number(abs(mean[i]-y[i])) if labeled[i] else None, 'crps_seconds':_number(crps[i]) if labeled[i] else None, 'interval_nll':_number(nll[i]) if available[i] else None, 'zero_probability_interval':bool(available[i] and not np.isfinite(nll[i]))} for i,r in enumerate(rows)]}
    pairs = []
    by_game = defaultdict(list)
    for i, row in enumerate(rows):
        by_game[(row['game_id'], row.get('player_color', 'unknown'))].append(i)
    for indices in by_game.values():
        ordered = sorted(indices, key=lambda i: rows[i].get('ply', 0))
        for previous, current in zip(ordered, ordered[1:]):
            if rows[current].get('ply', 0) - rows[previous].get('ply', 0) == 2 and labeled[previous] and labeled[current]:
                pairs.append((previous, current))
    rhythm = {'successive_own_turn_pairs': len(pairs), 'observed_log1p_correlation': None, 'predicted_mean_log1p_correlation': None}
    if len(pairs) >= 2:
        first, second = np.array(pairs).T
        for name, values in [('observed', y), ('predicted_mean', mean)]:
            a, b = np.log1p(values[first]), np.log1p(values[second])
            if a.std() > 0 and b.std() > 0:
                rhythm[name + '_log1p_correlation'] = float(np.corrcoef(a, b)[0, 1])
    output['recorded_state_rhythm'] = rhythm
    if deployment_weights is not None:
        output['deployment_weighted'] = weighted(deployment_weights)
    return output


def compare_scorecards(left, right, repetitions=2000, seed=42):
    a, b = left['per_row'], right['per_row']
    if [(r['game_id'],r['ply']) for r in a] != [(r['game_id'],r['ply']) for r in b]:
        raise ValueError('Reports do not contain identical positions in identical order')
    if left.get('timing_information') != right.get('timing_information'):
        raise ValueError('Do not compare conditional-human-move scores with marginal scores')
    result = {}
    for metric in ('crps_seconds', 'mean_absolute_error', 'interval_nll'):
        values = [[r.get(metric) if r.get(metric) is not None else float('nan') for r in rows] for rows in (a,b)]
        result[metric] = paired_game_bootstrap(a, *values, repetitions=repetitions, seed=seed)
        if metric == 'interval_nll':
            result[metric].update(
                interpretation='Difference and interval apply only to common finite interval-NLL pairs; not whole-sample NLL.',
                left_zero_probability_intervals=sum(bool(r.get('zero_probability_interval')) for r in a),
                right_zero_probability_intervals=sum(bool(r.get('zero_probability_interval')) for r in b))
    return result


def annotate_training_positions(rows, training):
    """Reporting-only repetition against training boards; clocks do not define position identity."""
    if {r['game_id'] for r in rows} & {r['game_id'] for r in training}:
        raise ValueError('Evaluation and training games overlap')
    keys = {' '.join(r['fen'].split()[:4]) for r in training}
    return [dict(r, position_seen_in_train=' '.join(r['fen'].split()[:4]) in keys) for r in rows]
