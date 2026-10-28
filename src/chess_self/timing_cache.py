"""Versioned frozen features; a cache is bound to exact weights and source rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .model import encode, load_model

CACHE_VERSION = 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output is not empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def build_cache(positions, checkpoint, output, *, device='cpu', batch_size=32,
                shard_size=256, limit_games=None, predictor_training_ids=None):
    from .timing_model import MaiaFeatures
    if batch_size < 1 or shard_size < 1:
        raise ValueError('Batch and shard sizes must be positive')
    positions, checkpoint = Path(positions), Path(checkpoint)
    if positions.name == 'test.jsonl':
        raise ValueError('Experiment caches do not accept the reserved test split')
    rows = read_rows(positions)
    if not rows or any(r.get('schema_version') != 'timing-data-v2' for r in rows):
        raise ValueError('Prepare timing schema v2 rows first')
    if limit_games is not None:
        if limit_games < 1:
            raise ValueError('limit_games must be positive')
        ids = list(dict.fromkeys(r['game_id'] for r in rows))[:limit_games]
        rows = [r for r in rows if r['game_id'] in ids]
    output = fresh_directory(output)
    model = MaiaFeatures(load_model(checkpoint, device)).eval()
    manifest = {'cache_version': CACHE_VERSION, 'checkpoint_sha256': digest(checkpoint),
                'positions_sha256': digest(positions), 'source_positions': str(positions.resolve()),
                'smoke_only': limit_games is not None, 'count': len(rows), 'shards': [],
                'predictor_training_game_ids': predictor_training_ids,
                'squares_dtype': 'float16', 'test_evaluated': False}
    dataset_manifest = positions.parent / 'manifest.json'
    if dataset_manifest.exists():
        manifest['dataset_manifest'] = json.loads(dataset_manifest.read_text())
        reserved = set(manifest['dataset_manifest'].get('reserved_test_game_ids', []))
        if reserved & {r['game_id'] for r in rows}:
            raise ValueError('Reserved test game in development cache')
    for start in range(0, len(rows), shard_size):
        chunk = rows[start:start + shard_size]
        parts = {}
        with torch.inference_mode():
            for offset in range(0, len(chunk), batch_size):
                batch_rows = chunk[offset:offset + batch_size]
                encoded = [encode(r) for r in batch_rows]
                batch = {k: torch.stack([v[k] for v in encoded]).to(device) for k in encoded[0]}
                features = model(batch['tokens'], batch['self_elos'], batch['oppo_elos'])
                features['move_logits'] = features['move_logits'].masked_fill(~batch['legal'], -torch.inf)
                probs = features['move_logits'].softmax(-1)
                entropy = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
                top = probs.topk(2, -1).values
                for i, row in enumerate(batch_rows):
                    row['policy_entropy'] = float(entropy[i])
                    row['top_two_gap'] = float(top[i, 0] - top[i, 1])
                add_opponent_surprisal(model,batch_rows,device)
                for key, value in features.items():
                    value = value.detach().cpu()
                    if key == 'squares':
                        value = value.half()
                    parts.setdefault(key, []).append(value)
        filename = f'features-{start:07d}.pt'
        torch.save({k: torch.cat(v) for k, v in parts.items()}, output / filename)
        manifest['shards'].append({'file': filename, 'start': start, 'count': len(chunk),
                                   'sha256': digest(output / filename)})
        print(f'Cached {start + len(chunk)}/{len(rows)} positions', flush=True)
    (output / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    manifest['rows_sha256'] = digest(output / 'rows.jsonl')
    write_json(output / 'manifest.json', manifest)
    return manifest


class FeatureCache(Dataset):
    def __init__(self, path, verify=True):
        self.path = Path(path)
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        if self.manifest.get('cache_version') != CACHE_VERSION:
            raise ValueError('Unsupported feature cache')
        if verify and digest(self.path / 'rows.jsonl') != self.manifest['rows_sha256']:
            raise ValueError('Cached row hash mismatch')
        self.rows = read_rows(self.path / 'rows.jsonl')
        if len(self.rows) != self.manifest['count']:
            raise ValueError('Cache row count mismatch')
        self.index = []
        for si, shard in enumerate(self.manifest['shards']):
            if shard['start'] != len(self.index):
                raise ValueError('Noncontiguous cache shards')
            if verify and digest(self.path / shard['file']) != shard['sha256']:
                raise ValueError('Cached feature hash mismatch')
            self.index.extend((si, j) for j in range(shard['count']))
        if len(self.index) != len(self.rows):
            raise ValueError('Cache shard count mismatch')
        self._mapped = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        shard, offset = self.index[index]
        if shard not in self._mapped:
            self._mapped[shard] = torch.load(self.path / self.manifest['shards'][shard]['file'],
                                            map_location='cpu', weights_only=True, mmap=True)
        return {k: v[offset].float() for k, v in self._mapped[shard].items()}

    def batch(self, indices, device='cpu'):
        items = [self[i] for i in indices]
        return {k: torch.stack([item[k] for item in items]).to(device) for k in items[0]}


def assert_comparable(training, validation):
    if training.manifest['checkpoint_sha256'] != validation.manifest['checkpoint_sha256']:
        raise ValueError('Feature caches use different backbone weights')
    if {r['game_id'] for r in training.rows} & {r['game_id'] for r in validation.rows}:
        raise ValueError('Training and validation games overlap')


def add_opponent_surprisal(extractor, rows, device='cpu'):
    """Score the already completed opponent move using its reconstructable prior clock."""
    import chess
    candidates, locations = [], []
    for i, row in enumerate(rows):
        row['opponent_surprisal'] = None
        history = row.get('history_fens', [])
        times = row.get('opponent_time_history', [])
        post = row.get('opponent_remaining_seconds')
        if len(history) < 2 or not times or times[-1] is None or post is None:
            continue
        before = post - row['increment_seconds'] + times[-1]
        if before <= 0: continue
        board = chess.Board(history[-2]); actual = None
        for move in board.legal_moves:
            copy = board.copy(); copy.push(move)
            if copy.fen() == row['fen']:
                actual = move.uci(); break
        if actual is None: continue
        candidate = dict(row, fen=history[-2], history_fens=history[:-1], move_uci=actual,
                         elo_self=row['elo_oppo'], elo_oppo=row['elo_self'],
                         remaining_seconds=before, elapsed_seconds=None)
        candidates.append(candidate); locations.append(i)
    if not candidates: return
    encoded = [encode(r) for r in candidates]
    batch = {k: torch.stack([e[k] for e in encoded]).to(device) for k in encoded[0]}
    with torch.no_grad():
        logits = extractor(batch['tokens'],batch['self_elos'],batch['oppo_elos'])['move_logits']
        logp = logits.masked_fill(~batch['legal'], -torch.inf).log_softmax(-1)
        values = -logp[torch.arange(len(candidates),device=device),batch['target']]
    for i,value in zip(locations,values):rows[i]['opponent_surprisal'] = float(value)
