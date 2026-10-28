import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('policy_candidates', Path(__file__).parents[1] / 'scripts/evaluate_policy_candidates.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Dataset:
    rows = [{'game_id': 'one', 'ply': 1}, {'game_id': 'two', 'ply': 3}]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {'tokens': torch.tensor([index]), 'self_elos': torch.tensor(600.),
                'oppo_elos': torch.tensor(600.), 'legal': torch.tensor([True, True, True, False]),
                'target': torch.tensor(index)}


class Model(torch.nn.Module):
    def forward(self, tokens, *args):
        logits = torch.tensor([[3., 2., 1., 100.], [1., 4., 2., 100.]])
        return logits[tokens[:, 0]], None, None


def test_legal_mask_and_position_order():
    result = module.evaluate_policy(Model(), Dataset(), batch_size=1)
    assert result['all']['top1'] == 1
    assert result['all']['top3'] == 1
    assert result['all']['move_nll'] < 1
    assert list(result['per_game']) == ['one', 'two']
    assert [(r['game_id'], r['ply']) for r in result['positions']] == [('one', 1), ('two', 3)]


def test_bootstrap_is_paired_and_preserves_game_clusters():
    rows = [{'game_id': 'long', 'ply': i, 'move_nll': 2., 'top1': 0., 'top3': 1.} for i in range(10)]
    rows.append({'game_id': 'short', 'ply': 1, 'move_nll': 3., 'top1': 0., 'top3': 1.})
    changed = [dict(r, move_nll=r['move_nll'] - 1) for r in rows]
    result = module.paired_comparison({'positions': rows}, {'positions': changed}, resamples=100)
    assert result['games'] == 2
    assert result['metrics']['move_nll']['difference'] == -1
    assert result['metrics']['move_nll']['bootstrap_95_interval'] == [-1, -1]
    assert result == module.paired_comparison({'positions': rows}, {'positions': changed}, resamples=100)
    with pytest.raises(ValueError, match='positions differ'):
        module.paired_comparison({'positions': rows}, {'positions': changed[::-1]})


def test_test_file_and_existing_output_are_rejected(tmp_path):
    with pytest.raises(SystemExit):
        module.main(['--data', 'data/processed/test.jsonl', '--checkpoint', 'x=x.pt', '--output', str(tmp_path / 'out.json')])
    path = tmp_path / 'out.json'
    path.write_text('preserve')
    with pytest.raises(SystemExit):
        module.main(['--data', 'val.jsonl', '--checkpoint', 'x=x.pt', '--output', str(path)])
    assert path.read_text() == 'preserve'


def test_probe_rating_overrides_do_not_mutate_input(monkeypatch):
    seen = []
    def predict(model, row):
        seen.append(row)
        return {'moves': [{'uci': 'e2e4', 'probability': .7}, {'uci': 'd2d4', 'probability': .3}]}
    monkeypatch.setattr(module, 'predict', predict)
    row = {'fen': 'fen', 'elo_self': 400, 'elo_oppo': 500}
    result = module.probe_policy(None, [{'row': row, 'uci': 'd2d4'}], 680, 700)
    assert row['elo_self'] == 400
    assert seen[0]['elo_self'] == 680
    assert result[0]['rank'] == 2
    assert result[0]['probability'] == .3
