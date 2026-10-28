"""Bounded detached setup/cache/three-arm timing run; never starts another rental."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def commands():
    py = str(ROOT / '.venv/bin/python')
    base = [py, '-m', 'chess_self.experiments']
    checkpoint = 'models/personalized/heads-01.pt'
    result = [('setup', ['bash', 'scripts/setup.sh']),
              ('cuda', [py, '-c', "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name()); assert torch.cuda.get_device_properties(0).total_memory > 20*1024**3"])]
    for split in ('train', 'val'):
        result.append(('cache-' + split, base + ['cache', '--positions', f'data/timing-v2/{split}.jsonl',
            '--checkpoint', checkpoint, '--device', 'cuda', '--output', f'data/timing-cache/{split}', '--batch-size', '32']))
    for name in ('e03-position', 'e04-window', 'e05-squares'):
        result.append((name, base + ['train', '--train-cache', 'data/timing-cache/train',
            '--val-cache', 'data/timing-cache/val', '--config', f'configs/timing/{name}.json',
            '--output', f'models/runs/timing-01/{name}', '--device', 'cuda', '--epochs', '8', '--batch-size', '32']))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline',required=True,help='UTC ISO timestamp for worker stop, before rental termination')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    deadline=datetime.fromisoformat(args.deadline.replace('Z','+00:00'))
    if deadline.tzinfo is None:raise ValueError('Timezone-aware deadline required')
    if args.dry_run:
        print(json.dumps(commands(),indent=2));return
    os.chdir(ROOT)
    job=ROOT/'jobs/timing-01'
    job.mkdir(parents=True,exist_ok=True)
    status_file=job/'status.json'
    if status_file.exists():raise ValueError('Job already exists; refusing duplicate start')
    state={'status':'running','pid':os.getpid(),'started':datetime.now(timezone.utc).isoformat(),
           'deadline':args.deadline,'steps':[]}
    def save():
        tmp=status_file.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(status_file)
    save()
    try:
        for name,command in commands():
            remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
            if remaining<=0:raise TimeoutError('Experiment deadline reached')
            state['step']=name;save();print('START',name,flush=True)
            child=subprocess.Popen(command,start_new_session=True)
            state['child_pid']=child.pid;save()
            try:code=child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise TimeoutError('Experiment deadline reached')
            state['steps'].append({'name':name,'exit_code':code});save()
            if code:raise RuntimeError(f'{name} exited {code}')
        state['status']='completed'
    except BaseException as error:
        state['status']='failed';state['error']=str(error)
        raise
    finally:
        state['finished']=datetime.now(timezone.utc).isoformat();save()

if __name__=='__main__':main()
