"""Two bounded policy-only recent-game candidates; retains the timing backbone."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
JOB = 'recent-policy-01'
POLICY = ('proj_sq_from.', 'proj_sq_to.', 'promo_bias_proj.')


def initialize():
    import torch
    old = torch.load(ROOT / 'models/personalized/heads-01.pt', map_location='cpu', weights_only=True)
    base = torch.load(ROOT / 'models/maia3-23m-ponder.pt', map_location='cpu', weights_only=True)
    state = old.get('model_state_dict', old)
    pretrained = base.get('model_state_dict', base)
    for key in state:
        if key.startswith(POLICY):
            state[key] = pretrained[key].clone()
    output = ROOT / f'models/runs/{JOB}/reset-initial.pt'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('Initial checkpoint already exists')
    torch.save({'model_state_dict': state, 'policy_reset': True}, output)


def commands():
    py = str(ROOT / '.venv/bin/python')
    result = [('setup', ['bash', 'scripts/setup.sh']),
              ('cuda', [py, '-c', 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())']),
              ('initialize', [py, 'scripts/recent_policy_job.py', '--initialize'])]
    for name, checkpoint in [('warm', 'models/personalized/heads-01.pt'),
                             ('reset', f'models/runs/{JOB}/reset-initial.pt')]:
        result.append((name, [py, '-m', 'chess_self.cli', 'train',
            '--data', 'data/processed/recent-policy-v1', '--checkpoint', checkpoint,
            '--output', f'models/runs/{JOB}/{name}', '--device', 'cuda',
            '--policy-only', '--timing-weight', '0', '--learning-rate', '0.0001',
            '--epochs', '12', '--patience', '3', '--batch-size', '32', '--seed', '42']))
    result.append(('compare', [py, 'scripts/evaluate_policy_candidates.py',
        '--data', 'data/processed/recent-policy-v1/val.jsonl',
        '--checkpoint', 'old=models/personalized/heads-01.pt',
        '--checkpoint', f'warm=models/runs/{JOB}/warm/best.pt',
        '--checkpoint', f'reset=models/runs/{JOB}/reset/best.pt',
        '--output', f'reports/{JOB}/comparison.json', '--device', 'cuda', '--batch-size', '32']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--initialize', action='store_true')
    args = parser.parse_args()
    if args.initialize:
        initialize()
        return
    if args.dry_run:
        print(json.dumps(commands(), indent=2))
        return
    if not args.deadline:
        parser.error('--deadline is required')
    deadline = datetime.fromisoformat(args.deadline.replace('Z', '+00:00'))
    if deadline.tzinfo is None or deadline <= datetime.now(timezone.utc):
        raise ValueError('Future timezone-aware deadline required')
    os.chdir(ROOT)
    job = ROOT / 'jobs' / JOB
    job.mkdir(parents=True, exist_ok=True)
    # Exclusive claim prevents accidentally launching concurrent experiments.
    with (job / 'worker.lock').open('x') as lock:
        lock.write(str(os.getpid()))
    state = {'status': 'running', 'pid': os.getpid(), 'started': datetime.now(timezone.utc).isoformat(),
             'deadline': args.deadline, 'steps': []}
    def save():
        tmp = job / 'status.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(job / 'status.json')
    child = None
    def interrupted(signum, frame):
        raise RuntimeError(f'Worker received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    save()
    try:
        for name, command in commands():
            remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                raise TimeoutError('Experiment deadline reached')
            state['step'] = name
            save()
            print('START', name, flush=True)
            child = subprocess.Popen(command, start_new_session=True,
                                     env={**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4'})
            state['child_pid'] = child.pid
            save()
            code = child.wait(timeout=remaining)
            state['steps'].append({'name': name, 'exit_code': code})
            save()
            if code:
                raise RuntimeError(f'{name} exited {code}')
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state['finished'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
