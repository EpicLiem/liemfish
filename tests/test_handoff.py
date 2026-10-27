import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_pipeline_is_ordered_and_does_not_evaluate_test_data():
    commands = script("jobs").commands("full", "test-run")
    assert commands[0] == ["bash", "scripts/setup.sh"]
    assert commands[1][1] == "scripts/preflight.py"
    assert commands[2] == ["bash", "scripts/train-gpu.sh", "models/runs/test-run"]
    assert not any("test.jsonl" in arg for command in commands for arg in command)


@pytest.mark.parametrize("name", ["../oops", "-bad", "a;b", "x y", "a/b"])
def test_job_names_cannot_escape_directory(name):
    with pytest.raises(ValueError):
        script("jobs").job_dir(name)


@pytest.mark.parametrize("host", ["-oProxyCommand=oops", "ubuntu@x;rm", "ubuntu@x/abc", "user name@host"])
def test_remote_host_injection_rejected(host):
    with pytest.raises(ValueError):
        script("remote").validate(host, "chess-self", "heads-01")


def test_failed_step_stops_pipeline_and_records_failure(tmp_path, monkeypatch):
    jobs = script("jobs")
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    directory = jobs.job_dir("failure")
    directory.mkdir(parents=True)
    monkeypatch.setattr(jobs, "commands", lambda *_: [[sys.executable, "-c", "raise SystemExit(7)"],
                                                   [sys.executable, "-c", "raise SystemExit(99)"]])
    assert jobs.run("failure", "full") == 1
    status = json.loads((directory / "status.json").read_text())
    assert status["status"] == "failed"
    assert len(status["steps"]) == 1
    assert status["steps"][0]["exit_code"] == 7


def test_collection_handles_missing_weights_and_preserves_logs(tmp_path, monkeypatch):
    remote = script("remote")
    monkeypatch.setattr(remote, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["remote.py", "collect", "--host", "ubuntu@example.com"])
    calls = []

    def fake(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1 if command[0] == "ssh" and "models/runs" in command[-1] else 0)

    monkeypatch.setattr(remote.subprocess, "run", fake)
    remote.main()
    copied = [command[-2] for command in calls if command[0] == "scp"]
    assert any("/jobs/" in path for path in copied)
    assert any("/reports/" in path for path in copied)
    assert not any("models/runs" in path for path in copied)


def test_collection_attempts_other_outputs_after_transfer_failure(tmp_path, monkeypatch):
    remote = script("remote")
    monkeypatch.setattr(remote, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["remote.py", "collect", "--host", "ubuntu@example.com"])
    copied = []

    def fake(command, **kwargs):
        if command[0] == "scp":
            copied.append(command)
            if len(copied) == 1:
                raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(remote.subprocess, "run", fake)
    with pytest.raises(SystemExit, match="Some transfers failed"):
        remote.main()
    assert len(copied) == 3
