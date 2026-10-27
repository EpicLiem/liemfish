#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
commit=95c0985aacb62bcaeedc30c3a9bbddfb43f91dd0
if [ ! -d sources/maia3 ]; then
  git clone https://github.com/daniel-monroe/maia3.git sources/maia3
  git -C sources/maia3 checkout "$commit"
fi
test "$(git -C sources/maia3 rev-parse HEAD)" = "$commit" || { echo "Unexpected Maia3 revision"; exit 1; }
uv sync --frozen --extra dev --python 3.12
.venv/bin/python scripts/checkpoint.py
.venv/bin/python -m pytest -q
.venv/bin/chess-self doctor
