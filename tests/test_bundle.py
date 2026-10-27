import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bundle.py"
spec = importlib.util.spec_from_file_location("bundle", SCRIPT)
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    for directory in bundle.DIRECTORIES:
        (root / directory).mkdir(parents=True)
    for name in bundle.FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + "\n")
    for name in (
        "src/chess_self/__init__.py", "tests/test_example.py", "docs/gpu.md",
        "sources/maia3/maia3/models.py", "sources/maia3/pyproject.toml",
        "data/raw/a.pgn", "data/raw/a.manifest.json", "data/processed/train.jsonl",
        "data/processed/val.jsonl", "data/processed/test.jsonl", "data/processed/audit.json",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + "\n")
    shutil.copyfile(SCRIPT, root / "scripts/bundle.py")
    (root / "scripts/setup.sh").write_text("#!/bin/sh\necho setup\n")
    (root / "scripts/setup.sh").chmod(0o755)
    return root


def extract(archive, destination):
    with tarfile.open(archive) as stream:
        stream.extractall(destination, filter="data")
    return destination / "chess-self"


def test_bundle_round_trip_manifest_and_portability(project, tmp_path):
    output = tmp_path / "portable.tar.gz"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(project), "--output", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    root = extract(output, tmp_path / "another-machine")
    manifest = bundle.verify(root)
    expected = {p.relative_to(project).as_posix() for p in project.rglob("*") if p.is_file()}
    assert set(manifest["files"]) == expected
    for name in expected:
        assert manifest["files"][name] == {
            "sha256": hashlib.sha256((project / name).read_bytes()).hexdigest(),
            "size": (project / name).stat().st_size,
        }
    assert "sources/maia3/maia3/models.py" in manifest["files"]
    assert {name for name in expected if name.startswith("data/")} <= manifest["files"].keys()
    assert (root / "scripts/setup.sh").stat().st_mode & 0o111
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(root / "scripts/bundle.py"), "--verify", str(root)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Verified" in result.stdout


def test_exclusions_and_deterministic_archive(project, tmp_path):
    omitted = (
        ".venv/bin/python", ".git/config", "reports/result.json", "models/export/model.onnx",
        "models/other.pt", "bundles/old.tar.gz", "sources/maia3/.git/config",
        "sources/maia3/maia3.egg-info/PKG-INFO", "src/__pycache__/x.pyc",
        "tests/.pytest_cache/README.md", "scripts/.env", "scripts/.env.production",
        "docs/credentials.json", "sources/maia3/private.pem", "data/raw/download.log",
        "data/processed/reports/metrics.json", "src/outputs/train.json", "src/model.onnx",
    )
    for name in omitted:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("must not package")
    first, second = tmp_path / "a.tar.gz", tmp_path / "b.tar.gz"
    manifest = bundle.build(project, first)
    for path in project.rglob("*"):
        if path.is_file():
            os.utime(path, (12345, 12345))
    bundle.build(project, second)
    assert first.read_bytes() == second.read_bytes()
    assert not set(omitted) & manifest["files"].keys()
    with tarfile.open(first) as stream:
        for member in stream.getmembers():
            assert member.name == "chess-self" or member.name.startswith("chess-self/")
            assert member.isfile() or member.isdir()
            assert member.uid == member.gid == member.mtime == 0
    bundle.verify(extract(first, tmp_path / "extracted"))


def test_runtime_verification_keeps_source_and_data_protected(project, tmp_path):
    archive = tmp_path / "runtime.tar.gz"
    bundle.build(project, archive)
    root = extract(archive, tmp_path / "runtime")
    for name in ("jobs/setup/status.json", "reports/readiness.json", "src/chess_self.egg-info/PKG-INFO",
                 "src/chess_self/__pycache__/x.pyc", ".bootstrap/config", ".venv/config", "models/runs/one/best.pt"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated")
    with pytest.raises(ValueError):
        bundle.verify(root)
    bundle.verify(root, allow_runtime=True)
    (root / "data/processed/train.jsonl").write_text("changed data")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        bundle.verify(root, allow_runtime=True)


@pytest.mark.parametrize("name,target", [
    ("src/link", "outside"), ("sources/maia3/link", "src"),
    ("data/raw/link", "outside"), ("scripts/.env", "outside"),
])
def test_symlinks_rejected(project, tmp_path, name, target):
    outside = tmp_path / "outside"
    outside.write_text("secret")
    (project / name).symlink_to(outside if target == "outside" else project / target)
    with pytest.raises(ValueError, match="Symlink"):
        bundle.build(project, tmp_path / "bad.tar.gz")
    assert not (tmp_path / "bad.tar.gz").exists()


def test_symlink_allowlist_ancestor_rejected(project, tmp_path):
    data = project / "data"
    data.rename(tmp_path / "data")
    data.symlink_to(tmp_path / "data", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        bundle.build(project, tmp_path / "bad.tar.gz")


@pytest.mark.parametrize("change", ["changed", "missing", "extra", "symlink"])
def test_verify_detects_tampering(project, tmp_path, change):
    output = tmp_path / "bundle.tar.gz"
    bundle.build(project, output)
    root = extract(output, tmp_path / "extract")
    target = root / "data/processed/train.jsonl"
    if change == "changed":
        target.write_bytes(b"X" * target.stat().st_size)
    elif change == "missing":
        target.unlink()
    elif change == "extra":
        (root / "unexpected-secret").write_text("untracked")
    else:
        target.unlink()
        target.symlink_to(project / "data/processed/train.jsonl")
    with pytest.raises(ValueError):
        bundle.verify(root)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "src/../../escape", "src/./file", "src\\escape", bundle.MANIFEST])
def test_verify_rejects_manifest_paths(project, tmp_path, name):
    output = tmp_path / "bundle.tar.gz"
    bundle.build(project, output)
    root = extract(output, tmp_path / "extract")
    manifest_path = root / bundle.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name] = {"size": 0, "sha256": "0" * 64}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe manifest path"):
        bundle.verify(root)


def test_missing_required_file_fails_without_replacing_output(project, tmp_path):
    (project / "models/maia3-23m-ponder.pt").unlink()
    output = tmp_path / "bundle.tar.gz"
    output.write_bytes(b"previous")
    with pytest.raises(FileNotFoundError):
        bundle.build(project, output)
    assert output.read_bytes() == b"previous"


def test_output_inside_allowlist_rejected(project):
    with pytest.raises(ValueError, match="outside the bundled allowlist"):
        bundle.build(project, project / "data/raw/bundle.tar.gz")


def test_default_output(project):
    result = subprocess.run([sys.executable, str(SCRIPT), "--root", str(project)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (project / "bundles/chess-self-gpu.tar.gz").is_file()


def test_output_with_parent_components_cannot_overwrite_source(project):
    (project / "scratch").mkdir()
    target = project / "README.md"
    original = target.read_bytes()
    with pytest.raises(ValueError, match="outside the bundled allowlist"):
        bundle.build(project, project / "scratch/../README.md")
    assert target.read_bytes() == original


def test_output_symlink_rejected(project, tmp_path):
    target = tmp_path / "existing"
    target.write_bytes(b"preserve")
    output = tmp_path / "bundle.tar.gz"
    output.symlink_to(target)
    with pytest.raises(ValueError, match="Symlink output"):
        bundle.build(project, output)
    assert target.read_bytes() == b"preserve"


def test_fifo_rejected(project, tmp_path):
    os.mkfifo(project / "data/raw/pipe")
    with pytest.raises(ValueError, match="Not a regular"):
        bundle.build(project, tmp_path / "bundle.tar.gz")
