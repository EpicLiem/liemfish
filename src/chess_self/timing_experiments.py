"""Controlled timing-only experiments on immutable game-level development splits."""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from datetime import date
from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn

from .model import encode, load_model, MOVE_INDEX
from maia3.utils import mirror_move
from .timing_cache import FeatureCache, assert_comparable, digest, fresh_directory, write_json
from .timing_data import feature_names, feature_vector
from .timing_model import (TimingHead, TimingHeadConfig, MaiaFeatures, history_tensor,
                           selected_move_inputs, save_timing_head, load_timing_head)


def side_inputs(rows, groups, device):
    required = {'retrieval':('nearest_timing_logmean','neighbor_count'),
                'engine':('engine_score_disagreement','engine_move_disagreement'),
                'familiarity':('prior_position_games','prior_line_games','first_unfamiliar_position','familiarity_known')}
    for group, fields in required.items():
        if group in groups and any(any(field not in row for field in fields) for row in rows):
            raise ValueError(f'Missing causal {group} inputs; supply the matching feature provider')
    return torch.tensor([feature_vector(r, groups) for r in rows], dtype=torch.float32, device=device)


def forward_head(head, features, rows, groups, moves=None):
    indices = attrs = None
    if head.config.move_mode != 'none':
        indices, attrs = selected_move_inputs(rows, features['move_logits'], moves)
    return head(features['pooled'], features['base_logits'],
                side_inputs(rows, groups, features['pooled'].device),
                features['squares'], indices, attrs,
                history_tensor(rows, head.config.history_length, device=features['pooled'].device))


def live_features(extractor, rows, device):
    items = [encode(r) for r in rows]
    batch = {k: torch.stack([item[k] for item in items]).to(device) for k in items[0]}
    features = extractor(batch['tokens'], batch['self_elos'], batch['oppo_elos'])
    features['move_logits'] = features['move_logits'].masked_fill(~batch['legal'], -torch.inf)
    probs = features['move_logits'].detach().softmax(-1)
    entropy = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
    top = probs.topk(2, -1).values
    for i, row in enumerate(rows):
        row['policy_entropy'] = float(entropy[i])
        row['top_two_gap'] = float(top[i, 0] - top[i, 1])
    from .timing_cache import add_opponent_surprisal
    add_opponent_surprisal(extractor,rows,device)
    return features


def sample_weights(rows, mode='natural', half_life_days=365, recent_days=365):
    if mode not in {'natural', 'balanced', 'recency', 'recent'}:
        raise ValueError('Unknown sampling objective')
    dates = [date.fromisoformat(r['date']) for r in rows]
    latest = max(dates)
    controls = [(r['initial_seconds'], r['increment_seconds']) for r in rows]
    counts = {c: controls.count(c) for c in set(controls)}
    if half_life_days <= 0 or recent_days <= 0:
        raise ValueError('Recency windows must be positive')
    weights = []
    for r, day, control in zip(rows, dates, controls):
        age = (latest - day).days
        weights.append(1 / counts[control] if mode == 'balanced' else
                       2 ** (-age / half_life_days) if mode == 'recency' else
                       float(age <= recent_days) if mode == 'recent' else 1.)
    return torch.tensor(weights, dtype=torch.double)


def timing_objective(parameters, rows, family, *, interval=True, feasible=True):
    from .timing_distributions import make_distribution
    device = parameters.device
    clocks = torch.tensor([r['remaining_seconds'] for r in rows], device=device)
    eligible = [i for i, r in enumerate(rows) if r.get('elapsed_seconds') is not None
                and r['elapsed_seconds'] >= 0 and r['remaining_seconds'] > 0
                and (not interval or r.get('observation_interval_seconds') is not None)]
    if not eligible:
        return parameters.sum() * 0, 0, 0
    idx = torch.tensor(eligible, device=device)
    chosen = [rows[i] for i in eligible]
    dist = make_distribution(parameters[idx], family, clocks[idx])
    if feasible:
        dist = dist.truncate(clocks[idx])
    if interval:
        low = parameters.new_tensor([r['observation_interval_seconds'][0] for r in chosen])
        high = parameters.new_tensor([r['observation_interval_seconds'][1] for r in chosen])
        log_probability = dist.log_interval_probability(low, high)
        geometric = (high > dist.lower) & (low < dist.upper)
        if not torch.isfinite(log_probability[geometric]).all():
            raise RuntimeError('Numerical interval likelihood failure inside model support')
        if not geometric.any():
            return parameters.sum() * 0, 0, len(chosen)
        return -log_probability[geometric].mean(), int(geometric.sum()), int((~geometric).sum())
    else:
        if family not in {'absolute', 'extended', 'fraction'}:
            raise ValueError('Point bucket targets only apply to categorical families')
        elapsed = parameters.new_tensor([r['elapsed_seconds'] for r in chosen])
        if family == 'fraction':
            bucket = (elapsed / clocks[idx] * parameters.shape[-1]).long()
            supported = (elapsed >= 0) & (elapsed < clocks[idx])
        else:
            bucket = ((torch.log10(elapsed + .1) + 1) * 10).long()
            supported = (elapsed >= 0) & (elapsed < 10 ** (parameters.shape[-1] / 10 - 1) - .1)
        safe = bucket.clamp(0, parameters.shape[-1] - 1)
        if family == 'fraction':
            low = safe / parameters.shape[-1] * clocks[idx]
            high = (safe + 1) / parameters.shape[-1] * clocks[idx]
        else:
            low = 10 ** (safe.float() / 10 - 1) - .1
            high = 10 ** ((safe.float() + 1) / 10 - 1) - .1
        logp = dist.log_interval_probability(low.clamp_min(0), high)
        supported = supported & dist.interval_supported(low.clamp_min(0), high)
        if not torch.isfinite(logp[supported]).all():
            raise RuntimeError('Numerical bucket likelihood failure')
        if not supported.any():
            return parameters.sum() * 0, 0, len(chosen)
        return -logp[supported].mean(), int(supported.sum()), int((~supported).sum())


def predict_cache(head, cache, groups, *, device='cpu', batch_size=32,
                  conditional=False, extractor=None, feasible=True):
    from .timing_distributions import make_distribution, mixture_distributions
    outputs, policies = [], []
    conditional_move = head is not None and head.config.move_mode != 'none'
    marginal = conditional_move and not conditional
    max_legal = max(len(list(chess.Board(r['fen']).legal_moves)) for r in cache.rows) if marginal else 1
    family = head.config.output_family if head else 'absolute'
    if head:
        head.eval()
    if extractor:
        extractor.eval()
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            ids = list(range(start, min(start + batch_size, len(cache))))
            rows = [cache.rows[i] for i in ids]
            features = live_features(extractor, rows, device) if extractor else cache.batch(ids, device)
            policies.append(features['move_logits'].cpu())
            if marginal:
                legal = [list(chess.Board(r['fen']).legal_moves) for r in rows]
                candidates = []
                for j in range(max_legal):
                    moves = [str(ms[j] if j < len(ms) else ms[0]) for ms in legal]
                    candidates.append(forward_head(head, features, rows, groups, moves).cpu())
                outputs.append(torch.stack(candidates, 1))
            else:
                outputs.append((forward_head(head, features, rows, groups) if head else features['base_logits']).cpu())
    params = torch.cat(outputs)
    logits = torch.cat(policies)
    clocks = torch.tensor([r['remaining_seconds'] for r in cache.rows])
    if (clocks <= 0).any():
        raise ValueError('Forecast evaluation requires a positive pre-move clock; audit exhausted turns separately')
    if marginal:
        weights = torch.zeros(len(cache), max_legal)
        probabilities = logits.softmax(-1)
        for i, r in enumerate(cache.rows):
            board = chess.Board(r['fen'])
            for j, move in enumerate(board.legal_moves):
                index = MOVE_INDEX[move.uci() if board.turn else mirror_move(move.uci())]
                weights[i, j] = probabilities[i, index]
        components = [make_distribution(params[:, j], family, clocks) for j in range(max_legal)]
        if feasible:
            # Normalize each conditional law first, preserving the frozen move policy.
            components = [d.truncate(clocks) for d in components]
        dist = mixture_distributions(components, weights)
    else:
        dist = make_distribution(params, family, clocks)
        if feasible:
            dist = dist.truncate(clocks)
    return dist, logits, params


def move_metrics(logits, rows):
    targets = torch.tensor([int(encode(r)['target']) for r in rows])
    logp = logits.log_softmax(-1)
    top = logits.topk(3, -1).indices
    return {'nll': float(-logp[torch.arange(len(rows)), targets].mean()),
            'top1': float((top[:, 0] == targets).float().mean()),
            'top3': float((top == targets[:, None]).any(-1).float().mean())}


def evaluate_head(head, cache, groups, *, device='cpu', batch_size=32, extractor=None,
                  feasible=True, conditional=False, crps_points=256, training_rows=None):
    from .timing_metrics import scorecard, annotate_training_positions
    dist, logits, _ = predict_cache(head, cache, groups, device=device, batch_size=batch_size,
                                   extractor=extractor, feasible=feasible, conditional=conditional)
    rows = annotate_training_positions(cache.rows, training_rows) if training_rows is not None else cache.rows
    report = scorecard(dist, rows, crps_points=crps_points)
    report['move_policy'] = move_metrics(logits, cache.rows)
    report['timing_information'] = 'human_move_conditional_diagnostic' if conditional else 'position_marginal'
    report['feasible_support'] = feasible
    report['within_bin_law'] = 'log-uniform(t+0.1) for absolute; uniform for fraction; continuous Beta'
    joint_dist = dist
    if head is not None and head.config.move_mode != 'none' and not conditional:
        joint_dist, _, _ = predict_cache(head, cache, groups, device=device, batch_size=batch_size,
                                        extractor=extractor, feasible=feasible, conditional=True)
    available = [i for i,r in enumerate(cache.rows) if r.get('elapsed_seconds') is not None and r.get('observation_interval_seconds') is not None]
    if available:
        low = torch.tensor([r.get('observation_interval_seconds', None)[0] if r.get('observation_interval_seconds') else 0. for r in cache.rows])
        high = torch.tensor([r.get('observation_interval_seconds', None)[1] if r.get('observation_interval_seconds') else 0. for r in cache.rows])
        targets = torch.tensor([int(encode(r)['target']) for r in cache.rows])
        joint = -joint_dist.log_interval_probability(low, high) - logits.log_softmax(-1)[torch.arange(len(cache)),targets]
        values = joint[available]
        failures = int((~torch.isfinite(values)).sum())
        report['joint_move_time_interval_nll'] = None if failures else float(values.mean())
        report['joint_zero_probability_intervals'] = failures
    return report


def selection_score(head, cache, groups, device, batch_size, extractor=None):
    dist, logits, _ = predict_cache(head, cache, groups, device=device,
                                   batch_size=batch_size, extractor=extractor)
    indices = [i for i, r in enumerate(cache.rows) if r.get('elapsed_seconds') is not None]
    # CRPS in seconds scores all labeled durations, including beyond finite support.
    values = torch.tensor([r.get('elapsed_seconds') or 0. for r in cache.rows])
    crps = dist.crps(values)[indices]
    scores = {'crps_seconds': float(crps.mean()), 'move_nll': move_metrics(logits, cache.rows)['nll']}
    for name, predicate in [('under10', lambda r:r['remaining_seconds'] < 10),
                            ('bullet', lambda r:r['initial_seconds'] <= 60)]:
        group = [j for j, i in enumerate(indices) if predicate(cache.rows[i])]
        scores[name] = float(crps[group].mean()) if group else None
    return scores


def train_experiment(train_cache, val_cache, output, config, *, device='cpu', epochs=8,
                     batch_size=32, learning_rate=1e-3, backbone_lr=1e-6,
                     weight_decay=.01, patience=2, seed=42, initialize=None,
                     checkpoint=None, final_blocks=0, smoke=False):
    training, validation = FeatureCache(train_cache), FeatureCache(val_cache)
    assert_comparable(training, validation)
    if config.get('public_training'):
        from .timing_data import validate_public_rows
        meta = training.manifest.get('dataset_manifest', {})
        if not meta.get('public_validation', {}).get('passed'):
            raise ValueError('E10 public pretraining requires public-check provenance')
        validate_public_rows(training.rows, target_accounts=meta['target_accounts'],
                             reserved_game_ids=meta['reserved_test_game_ids'], cutoff=meta['cutoff'])
    if epochs < 1 or batch_size < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError('Invalid optimization settings')
    if not smoke and (training.manifest['smoke_only'] or validation.manifest['smoke_only']):
        raise ValueError('Limited-game caches require --smoke')
    if not any(r.get('elapsed_seconds') is not None for r in training.rows):
        raise ValueError('No timing labels')
    if not any(r.get('elapsed_seconds') is not None for r in validation.rows):
        raise ValueError('No validation timing labels')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    groups = config.get('features', ['clock', 'phase', 'chess'])
    names = feature_names(groups)
    side_inputs(training.rows, groups, 'cpu')
    side_inputs(validation.rows, groups, 'cpu')
    if 'retrieval' in groups and not any(r.get('neighbor_count',0)>0 for r in training.rows):
        raise ValueError('Retrieval experiment has no earlier training neighbors')
    options = dict(config.get('head', {}), side_dim=len(names))
    head = TimingHead(TimingHeadConfig(**options)).to(device)
    inherited_ids = set()
    if initialize:
        head, payload = load_timing_head(initialize, device=device,
                                        base_checkpoint_sha256=training.manifest['checkpoint_sha256'], feature_names=names)
        inherited_ids = set(payload.get('extra', {}).get('trained_game_ids', []))
        if inherited_ids & {r['game_id'] for r in validation.rows}:
            raise ValueError('Initialization checkpoint has already fitted validation games')
        if asdict(head.config) != asdict(TimingHeadConfig(**options)):
            raise ValueError('Initialization architecture differs from experiment configuration')
    extractor = None
    if final_blocks:
        if checkpoint is None or digest(checkpoint) != training.manifest['checkpoint_sha256']:
            raise ValueError('Online backbone adaptation requires the exact cache checkpoint')
        extractor = MaiaFeatures(load_model(checkpoint, device), final_blocks)
        parameters = extractor.parameter_groups(head, learning_rate, backbone_lr)
    else:
        parameters = head.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    output = fresh_directory(output)
    weights = sample_weights(training.rows, config.get('sampling', 'natural'),
                             config.get('half_life_days', 365), config.get('recent_days', 365))
    reference = selection_score(None, validation, [], device, batch_size)
    baseline = selection_score(head, validation, groups, device, batch_size, extractor)
    max_regression = config.get('max_move_nll_regression', 0.0)
    guard = config.get('max_group_crps_regression', 0.0)
    manifest = {'experiment': config, 'seed': seed, 'reference': reference, 'initial': baseline,
                'selection_metric': 'marginal validation CRPS in seconds with move and group guards',
                'epochs': [], 'selected_epoch': 0, 'test_evaluated': False, 'smoke_only': smoke,
                'feature_names': names, 'architecture': head.metadata(), 'final_blocks': final_blocks,
                'training_cache': training.manifest, 'validation_cache': validation.manifest,
                'optimizer': {'lr': learning_rate, 'backbone_lr': backbone_lr, 'weight_decay': weight_decay},
                'initialization': str(initialize) if initialize else 'new timing side component',
                'accepted_candidate': False, 'continuation': 'weights only; optimizer is newly initialized'}
    def save(epoch):
        sha = training.manifest['checkpoint_sha256']
        if extractor:
            torch.save({'model_state_dict': extractor.model.state_dict()}, output / ('initial-backbone.pt' if epoch == 0 else 'backbone.pt'))
            sha = digest(output / ('initial-backbone.pt' if epoch == 0 else 'backbone.pt'))
        save_timing_head(output / ('initial.pt' if epoch == 0 else 'best.pt'), head, base_checkpoint_sha256=sha, feature_names=names,
                         extra={'epoch': epoch, 'feature_groups': groups, 'smoke_only': smoke,
                                'trained_game_ids': sorted(inherited_ids | {r['game_id'] for r in training.rows})})
    save(0)
    best = min(baseline['crps_seconds'], reference['crps_seconds']); stale = 0
    for epoch in range(1, epochs + 1):
        head.train()
        if extractor: extractor.train()
        order = torch.multinomial(weights, len(training), replacement=True).tolist() if config.get('sampling', 'natural') != 'natural' else torch.randperm(len(training)).tolist()
        losses = []; used = excluded = 0
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]; rows = [training.rows[i] for i in ids]
            features = live_features(extractor, rows, device) if extractor else training.batch(ids, device)
            optimizer.zero_grad(set_to_none=True)
            raw = forward_head(head, features, rows, groups)
            loss, count, unsupported = timing_objective(raw, rows, head.config.output_family,
                                                      interval=config.get('interval_labels', True))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite timing loss')
            if count:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']], 1., error_if_nonfinite=True)
                optimizer.step(); losses.append(float(loss.detach()))
            used += count; excluded += unsupported
        if not used: raise ValueError('No supported timing observations in training objective')
        metrics = selection_score(head, validation, groups, device, batch_size, extractor)
        accepted = metrics['move_nll'] <= reference['move_nll'] + max_regression + 1e-6
        for key in ('under10', 'bullet'):
            if reference[key] is not None:
                accepted &= metrics[key] <= reference[key] + guard
        improved = accepted and metrics['crps_seconds'] < best
        if improved:
            best = metrics['crps_seconds']; stale = 0; manifest['selected_epoch'] = epoch; manifest['accepted_candidate'] = True; save(epoch)
        else: stale += 1
        manifest['epochs'].append({'epoch': epoch, 'loss': sum(losses) / len(losses),
                                   'supported_training_observations': used, 'unsupported_training_observations': excluded,
                                   'validation': metrics, 'passes_guards': bool(accepted), 'selected': bool(improved)})
        write_json(output / 'run.json', manifest)
        print(f"epoch {epoch}: marginal CRPS {metrics['crps_seconds']:.4f}; guards={accepted}", flush=True)
        if stale >= patience: break
    if extractor:
        torch.save({'model_state_dict': extractor.model.state_dict()}, output / 'last-backbone.pt')
        last_sha = digest(output / 'last-backbone.pt')
    else:
        last_sha = training.manifest['checkpoint_sha256']
    save_timing_head(output / 'last.pt', head, base_checkpoint_sha256=last_sha, feature_names=names,
                     extra={'epoch':epoch,'feature_groups':groups,'smoke_only':smoke,'accepted':False,
                            'trained_game_ids':sorted(inherited_ids | {r['game_id'] for r in training.rows})})
    return manifest
