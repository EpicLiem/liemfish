import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location('evaluate_selected_timing', ROOT/'scripts/evaluate_selected_timing.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_plan_uses_selected_heads_and_identical_validation_inputs():
    plan = module().commands(Path('reports/fresh'))
    commands = [command for _, command in plan]
    assert len(commands) == 33
    assert sum('cache' in c for c in commands) == 1
    evaluations = [c for c in commands if 'evaluate' in c]
    assert len(evaluations) == 16
    assert sum('--unconstrained' in c for c in evaluations) == 8
    assert {c[c.index('--crps-points')+1] for c in evaluations} == {'256', '1024'}
    assert {c[c.index('--cache')+1] for c in evaluations} == {'reports/fresh/cache'}
    heads = [c[c.index('--head')+1] for c in evaluations if '--head' in c]
    assert len(set(heads)) == 3
    assert all(p.endswith('/best.pt') for p in heads)
    assert not any('last.pt' in p or 'test.jsonl' in p for c in commands for p in c)
    assert len([c for c in commands if 'compare' in c]) == 16


def test_default_is_dry_run_without_output_writes(tmp_path, capsys):
    m = module()
    output = tmp_path/'new'
    m.main(['--output', str(output)])
    plan = json.loads(capsys.readouterr().out)
    assert not output.exists()
    assert plan['test_evaluated'] is False


def test_existing_output_fails_before_provenance_or_execution(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, 'provenance', lambda *_: pytest.fail('Read inputs before overwrite guard'))
    with pytest.raises(ValueError, match='fresh output'):
        m.run_plan(tmp_path, [])


def test_failure_records_progress_without_starting_later_steps(tmp_path, monkeypatch):
    import subprocess
    m = module()
    monkeypatch.setattr(m, 'provenance', lambda *_: {'test_evaluated': False})
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command == ['fail']:
            raise subprocess.CalledProcessError(2, command)
    monkeypatch.setattr(m.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        m.run_plan(tmp_path/'fresh', [('first',['ok']), ('second',['fail']), ('third',['never'])])
    report=json.loads((tmp_path/'fresh/provenance.json').read_text())
    assert report['status']=='failed'
    assert report['completed_steps']==['first']
    assert calls==[['ok'],['fail']]


def test_provenance_rejects_smoke_selection(tmp_path):
    m=module()
    path=tmp_path/m.RUNS/m.ARMS[0]/'run.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'accepted_candidate':True,'smoke_only':True,'selected_epoch':1}))
    with pytest.raises(ValueError,match='full-data'):
        m.provenance(tmp_path)
