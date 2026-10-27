"""Detached, logged jobs. Uses only the standard library so setup can run first."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def job_dir(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("Job name must contain only letters, digits, underscores and hyphens")
    return ROOT / "jobs" / name


def commands(mode, name):
    python = str(ROOT / ".venv/bin/python")
    cli = str(ROOT / ".venv/bin/chess-self")
    result = []
    if mode in ("setup", "full"):
        result.append(["bash", "scripts/setup.sh"])
    if mode in ("train", "full"):
        result += [[python, "scripts/preflight.py", "--output", f"reports/gpu-preflight-{name}"],
                   ["bash", "scripts/train-gpu.sh", f"models/runs/{name}"]]
    if mode == "smoke":
        result.append([cli, "train", "--device", "cpu", "--smoke", "--output", f"reports/job-smoke-{name}"])
    return result


def record(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def run(name, mode):
    directory = job_dir(name)
    state = {"name": name, "mode": mode, "pid": os.getpid(), "status": "running",
             "started_at": datetime.now(timezone.utc).isoformat(), "steps": []}
    record(directory / "status.json", state)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        for command in commands(mode, name):
            state["current_command"] = command
            record(directory / "status.json", state)
            print("Running:", command, flush=True)
            result = subprocess.run(command, cwd=ROOT, env=env)
            state["steps"].append({"command": command, "exit_code": result.returncode})
            if result.returncode:
                raise RuntimeError(f"Step failed with exit code {result.returncode}")
        state["status"] = "completed"
    except Exception as error:
        state.update(status="failed", error=str(error))
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    record(directory / "status.json", state)
    return 0 if state["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "status", "_run"])
    parser.add_argument("name")
    parser.add_argument("--mode", choices=["setup", "train", "full", "smoke"], default="full")
    args = parser.parse_args()
    directory = job_dir(args.name)
    if args.action == "start":
        directory.mkdir(parents=True, exist_ok=False)
        record(directory / "status.json", {"status": "starting", "name": args.name, "mode": args.mode})
        with (directory / "output.log").open("wb") as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_run", args.name,
                                     "--mode", args.mode], cwd=ROOT, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": proc.pid, "job": str(directory), "log": str(directory / "output.log")}))
    elif args.action == "status":
        state = json.loads((directory / "status.json").read_text())
        if state["status"] == "running":
            try:
                os.kill(state["pid"], 0)
                state["process_exists"] = True
            except ProcessLookupError:
                state.update(status="interrupted", process_exists=False)
        print(json.dumps(state, indent=2))
    else:
        raise SystemExit(run(args.name, args.mode))


if __name__ == "__main__":
    main()
