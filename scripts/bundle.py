"""Build a portable source/data/checkpoint bundle, or verify its extracted contents."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "BUNDLE-MANIFEST.json"
DIRECTORIES = (
    "src", "tests", "scripts", "docs", "sources/maia3", "data/raw", "data/processed",
)
FILES = (
    "README.md", "AGENTS.md", "LICENSE", "pyproject.toml", "uv.lock", ".gitignore",
    "sources/provenance.json", "sources/checkpoint-api.json", "models/maia3-23m-ponder.pt",
)
EXCLUDED = {
    "__pycache__", "node_modules", "venv", "env", "cache", "caches",
    "reports", "runs", "outputs", "output", "logs", "bundles", "wandb",
    "build", "dist", "htmlcov", "credentials", "credentials.json", "secrets",
    "secrets.json", "secrets.toml", "id_rsa", "id_ed25519",
}
EXCLUDED_SUFFIXES = (
    ".pyc", ".pyo", ".onnx", ".onnx.data", ".log", ".egg-info", ".dist-info",
    ".pem", ".key", ".p12", ".pfx", ".part", ".tmp", ".tar.gz", ".tgz",
    ".pt", ".pth", ".ckpt", ".safetensors",
)


def excluded(name):
    lower = name.lower()
    return (
        (name.startswith(".") and name not in {".gitignore", ".gitattributes"})
        or lower in EXCLUDED or lower.endswith(EXCLUDED_SUFFIXES)
    )


def checked(path, root):
    """Reject links at every component, including an allowlist entry's ancestors."""
    path.relative_to(root)
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink is not permitted: {current}")
    info = path.lstat()
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise ValueError(f"Not a regular file or directory: {path}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes root: {path}")
    return info


def walk(path, root, prune):
    info = checked(path, root)
    if prune and excluded(path.name):
        return
    if stat.S_ISDIR(info.st_mode):
        for child in sorted(path.iterdir()):
            yield from walk(child, root, prune)
    else:
        yield path


def selected_files(root):
    result = []
    for relative in DIRECTORIES:
        path = root / relative
        if not stat.S_ISDIR(checked(path, root).st_mode):
            raise ValueError(f"Required directory missing: {relative}")
        result.extend(walk(path, root, True))
    for relative in FILES:
        path = root / relative
        if not stat.S_ISREG(checked(path, root).st_mode):
            raise ValueError(f"Required file missing: {relative}")
        result.append(path)
    return sorted(result, key=lambda path: path.relative_to(root).as_posix())


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def archive_info(name, size=0, mode=0o644, directory=False):
    info = tarfile.TarInfo("chess-self/" + name if name else "chess-self")
    info.size = size
    info.mode = mode
    if directory:
        info.type = tarfile.DIRTYPE
    return info


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.checksum = hashlib.sha256()

    def read(self, size):
        block = self.stream.read(size)
        self.checksum.update(block)
        return block


def build(root, output):
    root = Path(root).absolute()
    checked(root, root)
    root = root.resolve()
    paths = selected_files(root)
    output = Path(output).absolute()
    for path in (output, *output.parents):
        if path.is_symlink():
            raise ValueError(f"Symlink output path is not permitted: {path}")
    output = output.resolve()
    if output.is_relative_to(root):
        relative = output.relative_to(root).as_posix()
        if relative == MANIFEST or relative in FILES or any(
            relative == directory or relative.startswith(directory + "/")
            for directory in DIRECTORIES
        ):
            raise ValueError("Output must be outside the bundled allowlist")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "algorithm": "sha256", "files": {}}
    directories = {"", *DIRECTORIES}
    for path in paths:
        directories.update(parent.as_posix() for parent in path.relative_to(root).parents if parent != Path("."))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".bundle-", delete=False) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                    for directory in sorted(directories):
                        archive.addfile(archive_info(directory, mode=0o755, directory=True))
                    for path in paths:
                        before = checked(path, root)
                        name = path.relative_to(root).as_posix()
                        with path.open("rb") as stream:
                            opened = os.fstat(stream.fileno())
                            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                                raise ValueError(f"File changed while opening: {name}")
                            reader = HashingReader(stream)
                            mode = 0o755 if before.st_mode & 0o111 else 0o644
                            archive.addfile(archive_info(name, before.st_size, mode), reader)
                            after = os.fstat(stream.fileno())
                        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                            after.st_size, after.st_mtime_ns, after.st_ctime_ns
                        ):
                            raise ValueError(f"File changed during bundling: {name}")
                        manifest["files"][name] = {"sha256": reader.checksum.hexdigest(), "size": before.st_size}
                    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
                    archive.addfile(archive_info(MANIFEST, len(payload)), io.BytesIO(payload))
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def runtime_files(path, root):
    parts = path.relative_to(root).parts
    if parts and (parts[0] in {".venv", ".bootstrap", ".pytest_cache", "jobs", "reports", "bundles"}
                  or parts[:2] in {("models", "runs"), ("models", "export")}
                  or "__pycache__" in parts or any(p.endswith(".egg-info") for p in parts)):
        return
    info = checked(path, root)
    if stat.S_ISDIR(info.st_mode):
        for child in sorted(path.iterdir()):
            yield from runtime_files(child, root)
    else:
        yield path


def verify(root, allow_runtime=False):
    root = Path(root).absolute()
    checked(root, root)
    root = root.resolve()
    checked(root / MANIFEST, root)
    manifest = json.loads((root / MANIFEST).read_text())
    if not isinstance(manifest, dict) or manifest.get("version") != 1 or manifest.get("algorithm") != "sha256" or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported bundle manifest")
    records = manifest["files"]
    for name, record in records.items():
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or path.as_posix() != name or "\\" in name or name == MANIFEST:
            raise ValueError(f"Unsafe manifest path: {name}")
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64 or type(record.get("size")) is not int or record["size"] < 0:
            raise ValueError(f"Invalid manifest record: {name}")
    files = runtime_files(root, root) if allow_runtime else walk(root, root, False)
    actual = {path.relative_to(root).as_posix() for path in files} - {MANIFEST}
    if actual != set(records):
        raise ValueError(f"Bundle file set mismatch; missing={sorted(set(records) - actual)}, unexpected={sorted(actual - set(records))}")
    allowed = {path.relative_to(root).as_posix() for path in selected_files(root)}
    if allowed != actual:
        raise ValueError("Bundle contains files outside the allowlist")
    for name, record in records.items():
        path = root / name
        if path.stat().st_size != record["size"] or digest(path) != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root to bundle")
    parser.add_argument("--output", type=Path, help="Archive path (default: ROOT/bundles/chess-self-gpu.tar.gz)")
    parser.add_argument("--verify", type=Path, metavar="ROOT", help="Verify an extracted chess-self directory offline")
    parser.add_argument("--allow-runtime", action="store_true", help="Ignore generated environments, logs, caches and jobs during verification")
    args = parser.parse_args()
    try:
        if args.verify is not None:
            manifest = verify(args.verify, allow_runtime=args.allow_runtime)
            print(f"Verified {len(manifest['files'])} files in {args.verify}")
        else:
            output = args.output or args.root / "bundles/chess-self-gpu.tar.gz"
            manifest = build(args.root, output)
            print(f"Bundled {len(manifest['files'])} files into {output}")
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"Bundle error: {error}\n")


if __name__ == "__main__":
    main()
