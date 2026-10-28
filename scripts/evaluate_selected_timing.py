"""Plan or run matched validation scorecards for the three selected timing heads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path('reports/collected/chess-self-timing-retry/models-runs/timing-01')
ARMS = ('e03-position', 'e04-window', 'e05-squares')
CHECKPOINT = Path('models/personalized/heads-01.pt')
TRAINING = Path('data/timing-v2/train.jsonl')
VALIDATION = Path('data/timing-v2/val.jsonl')


def commands(output, *, device='cpu', batch_size=32, root=ROOT):
    output = Path(output)
    base = [str(root / '.venv/bin/python'), '-m', 'chess_self.experiments']
    cache = output / 'cache'
    steps = [('cache-validation', base + ['cache', '--positions', str(VALIDATION),
        '--checkpoint', str(CHECKPOINT), '--output', str(cache), '--device', device,
        '--batch-size', str(batch_size)])]
    for points in (256, 1024):
        for support in ('feasible', 'raw'):
            prefix = output / f'{support}-{points}'
            reports = {}
            for arm in ('reference', *ARMS):
                reports[arm] = prefix / f'{arm}.json'
                command = base + ['evaluate', '--cache', str(cache), '--output', str(reports[arm]),
                    '--batch-size', str(batch_size), '--crps-points', str(points),
                    '--training-positions', str(TRAINING)]
                if arm != 'reference':
                    command += ['--head', str(RUNS / arm / 'best.pt')]
                if support == 'raw':
                    command += ['--unconstrained']
                steps.append((f'{support}-{points}-{arm}', command))
            for left, right in [(arm, 'reference') for arm in ARMS] + [('e05-squares', 'e04-window')]:
                steps.append((f'{support}-{points}-{left}-vs-{right}', base + ['compare',
                    '--left', str(reports[left]), '--right', str(reports[right]),
                    '--output', str(prefix / f'{left}-vs-{right}.json')]))
    return steps


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def provenance(root=ROOT):
    paths = [CHECKPOINT, TRAINING, VALIDATION]
    selected = {}
    for arm in ARMS:
        run_path = RUNS / arm / 'run.json'
        run = json.loads((root / run_path).read_text())
        if not run.get('accepted_candidate') or run.get('smoke_only') is not False or not run.get('selected_epoch'):
            raise ValueError(f'{arm} must have an accepted, full-data selected checkpoint')
        selected[arm] = {'epoch': run['selected_epoch'], 'checkpoint': str(RUNS / arm / 'best.pt'),
                         'config': str(Path('configs/timing') / f'{arm}.json')}
        paths.extend([run_path, RUNS / arm / 'best.pt', Path(selected[arm]['config'])])
    return {'schema_version': 1, 'test_evaluated': False,
            'selection': 'Saved best.pt from full-data development selection; no new selection or fitting.',
            'selected': selected,
            'inputs': {str(path): {'sha256': sha256(root / path), 'bytes': (root / path).stat().st_size}
                       for path in paths}}


def run_plan(output, steps, *, root=ROOT):
    output = Path(output)
    if not output.is_absolute():
        output = root / output
    if output.exists():
        raise ValueError('Choose a fresh output directory; evaluation never overwrites or resumes')
    evidence = provenance(root)
    evidence['commands'] = steps
    evidence['started_utc'] = datetime.now(timezone.utc).isoformat()
    evidence['status'] = 'running'
    evidence['completed_steps'] = []
    output.mkdir(parents=True, exist_ok=False)
    report = output / 'provenance.json'
    def save():
        report.write_text(json.dumps(evidence, indent=2) + '\n')
    save()
    try:
        for name, command in steps:
            print(f'START {name}', flush=True)
            subprocess.run(command, cwd=root, check=True)
            evidence['completed_steps'].append(name)
            save()
        evidence['status'] = 'completed'
    except BaseException as error:
        evidence['status'] = 'failed'
        evidence['error'] = str(error)
        raise
    finally:
        evidence['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--run', action='store_true', help='Execute locally; default only prints the plan')
    action.add_argument('--dry-run', action='store_true', help='Print commands without writing files or loading models')
    parser.add_argument('--output', type=Path, default=Path('reports/timing-full-evaluation-v1'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device for feature caching')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    steps = commands(args.output, device=args.device, batch_size=args.batch_size)
    if args.run:
        run_plan(args.output, steps)
    else:
        print(json.dumps({'output': str(args.output), 'test_evaluated': False, 'commands': steps}, indent=2))


if __name__ == '__main__':
    main()
