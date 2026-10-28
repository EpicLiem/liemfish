import importlib.util
from pathlib import Path

import torch
import pytest

spec = importlib.util.spec_from_file_location('recent_job', Path(__file__).parents[1] / 'scripts/recent_policy_job.py')
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


def test_bounded_candidates_preserve_timing():
    commands = dict(job.commands())
    assert list(commands) == ['setup', 'cuda', 'initialize', 'warm', 'reset', 'compare']
    for name in ('warm', 'reset'):
        args = commands[name]
        assert '--policy-only' in args
        assert args[args.index('--timing-weight') + 1] == '0'
        assert args[args.index('--epochs') + 1] == '12'
        assert args[args.index('--data') + 1] == 'data/processed/recent-policy-v1'


def test_reset_changes_only_policy_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(job, 'ROOT', tmp_path)
    (tmp_path / 'models/personalized').mkdir(parents=True)
    old = {'proj_sq_from.weight': torch.tensor([1.]), 'last_ln.weight': torch.tensor([2.])}
    base = {'proj_sq_from.weight': torch.tensor([3.]), 'last_ln.weight': torch.tensor([4.])}
    torch.save({'model_state_dict': old}, tmp_path / 'models/personalized/heads-01.pt')
    torch.save({'model_state_dict': base}, tmp_path / 'models/maia3-23m-ponder.pt')
    job.initialize()
    result = torch.load(tmp_path / f'models/runs/{job.JOB}/reset-initial.pt', weights_only=True)['model_state_dict']
    assert result['proj_sq_from.weight'].item() == 3
    assert result['last_ln.weight'].item() == 2
    with pytest.raises(ValueError, match='already exists'):
        job.initialize()
