"""Compare move policies on an explicitly selected development set."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chess_self.model import Positions, load_model, predict, to_device


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def summarize(records):
    if not records:
        raise ValueError('No positions to evaluate')
    return {'positions': len(records), **{
        key: float(np.mean([r[key] for r in records])) for key in ('move_nll', 'top1', 'top3')
    }}


def evaluate_policy(model, dataset, device='cpu', batch_size=32):
    """Keep each target's score so comparisons use identical games and positions."""
    model.eval()
    records = []
    with torch.inference_mode():
        for raw in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            batch = to_device(raw, device)
            logits, _, _ = model(batch['tokens'], batch['self_elos'], batch['oppo_elos'])
            logits = logits.masked_fill(~batch['legal'], -torch.inf)
            nll = F.cross_entropy(logits, batch['target'], reduction='none')
            if not torch.isfinite(nll).all():
                raise ValueError('Nonfinite policy score')
            top = logits.topk(min(3, logits.shape[-1]), dim=-1).indices
            offset = len(records)
            for i in range(len(nll)):
                row = dataset.rows[offset + i]
                records.append({'game_id': row['game_id'], 'ply': row['ply'],
                                'move_nll': nll[i].item(),
                                'top1': int(top[i, 0] == batch['target'][i]),
                                'top3': int((top[i] == batch['target'][i]).any())})
    grouped = defaultdict(list)
    for record in records:
        grouped[record['game_id']].append(record)
    return {'all': summarize(records),
            'per_game': {key: summarize(rows) for key, rows in sorted(grouped.items())},
            'positions': records}


def paired_comparison(reference, candidate, *, seed=42, resamples=4000):
    """Paired whole-game bootstrap; each sampled game retains all its moves."""
    left, right = reference['positions'], candidate['positions']
    identities = lambda rows: [(r['game_id'], r['ply']) for r in rows]
    if identities(left) != identities(right):
        raise ValueError('Candidate positions differ from reference')
    groups = defaultdict(list)
    for a, b in zip(left, right):
        groups[a['game_id']].append([b[k] - a[k] for k in ('move_nll', 'top1', 'top3')])
    values = [np.asarray(rows) for rows in groups.values()]
    sums = np.asarray([rows.sum(axis=0) for rows in values])
    counts = np.asarray([len(rows) for rows in values])
    rng = np.random.default_rng(seed)
    deltas = np.empty((resamples, 3))
    for index in range(resamples):
        selected = rng.integers(0, len(values), size=len(values))
        deltas[index] = sums[selected].sum(axis=0) / counts[selected].sum()
    point = sums.sum(axis=0) / counts.sum()
    return {'direction': 'candidate minus reference', 'games': len(values),
            'resamples': resamples, 'seed': seed,
            'note': 'Whole-game resampling does not account for correlated sessions.',
            'metrics': {key: {'difference': float(point[i]),
                              'bootstrap_95_interval': np.quantile(deltas[:, i], [.025, .975]).tolist()}
                        for i, key in enumerate(('move_nll', 'top1', 'top3'))}}


def probe_policy(model, probes, elo_self=None, elo_oppo=None):
    result = []
    for probe in probes:
        row = dict(probe['row'])
        if elo_self is not None:
            row['elo_self'] = elo_self
        if elo_oppo is not None:
            row['elo_oppo'] = elo_oppo
        moves = predict(model, row)['moves']
        selected = next(((i, move) for i, move in enumerate(moves, 1)
                         if move['uci'] == probe['uci']), None)
        if selected is None:
            raise ValueError(f"Illegal probe move: {probe['uci']}")
        rank, move = selected
        result.append({'game': probe.get('game'), 'ply': probe.get('ply'), 'fen': row['fen'],
                       'elo_self': row['elo_self'], 'elo_oppo': row['elo_oppo'],
                       'uci': move['uci'], 'probability': move['probability'], 'rank': rank,
                       'top3': moves[:3]})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Explicit development JSONL; never original test data')
    parser.add_argument('--checkpoint', action='append', required=True, metavar='NAME=PATH', help='First is paired reference')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--probes', type=Path, help='JSON array of {row,uci} diagnostic positions; not a selection set')
    parser.add_argument('--elo-self', type=float)
    parser.add_argument('--elo-oppo', type=float)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists; choose a fresh report path')
    if 'test' in args.data.stem.lower():
        parser.error('Use development data, not a test-named file')
    if args.batch_size < 1 or args.threads < 1:
        parser.error('Batch size and threads must be positive')
    checkpoints = {}
    for value in args.checkpoint:
        name, separator, path = value.partition('=')
        if not name or not separator or not path or name in checkpoints:
            parser.error('Checkpoints require unique NAME=PATH arguments')
        checkpoints[name] = Path(path)
    torch.set_num_threads(args.threads)
    dataset = Positions(args.data)
    keys = [(r['game_id'], r['ply']) for r in dataset.rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate game/ply positions in development set')
    for row in dataset.rows:
        if args.elo_self is not None:
            row['elo_self'] = args.elo_self
        if args.elo_oppo is not None:
            row['elo_oppo'] = args.elo_oppo
    probes = json.loads(args.probes.read_text()) if args.probes else []
    report = {'data': str(args.data), 'data_sha256': sha256(args.data),
              'device': args.device, 'batch_size': args.batch_size,
              'rating_overrides': {'self': args.elo_self, 'opponent': args.elo_oppo},
              'probes_sha256': sha256(args.probes) if args.probes else None,
              'probe_role': 'Diagnostic only; previously observed positions',
              'candidates': {}, 'paired_comparisons': {}}
    reference = next(iter(checkpoints))
    for name, checkpoint in checkpoints.items():
        model = load_model(checkpoint, args.device)
        metrics = evaluate_policy(model, dataset, args.device, args.batch_size)
        metrics.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                       probes=probe_policy(model, probes, args.elo_self, args.elo_oppo))
        report['candidates'][name] = metrics
        if name != reference:
            report['paired_comparisons'][name] = paired_comparison(
                report['candidates'][reference], metrics, seed=args.seed)
        print(json.dumps({'candidate': name, **metrics['all']}), flush=True)
        del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    main()
