#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f BUNDLE-MANIFEST.json ]; then
  python3 scripts/bundle.py --verify . --allow-runtime
fi
if ! command -v uv >/dev/null || ! uv --version | grep -q '^uv 0.13.6'; then
  python3 -m venv .bootstrap
  .bootstrap/bin/python -m pip install 'uv==0.13.6'
  export PATH="$PWD/.bootstrap/bin:$PATH"
fi
commit=95c0985aacb62bcaeedc30c3a9bbddfb43f91dd0
if [ ! -d sources/maia3 ]; then
  git clone https://github.com/daniel-monroe/maia3.git sources/maia3
  git -C sources/maia3 checkout "$commit"
fi
if [ -d sources/maia3/.git ]; then
  test "$(git -C sources/maia3 rev-parse HEAD)" = "$commit" || { echo "Unexpected Maia3 revision"; exit 1; }
  git -C sources/maia3 diff --quiet HEAD || { echo "Modified upstream source"; exit 1; }
elif [ ! -f BUNDLE-MANIFEST.json ]; then
  echo "Source snapshot requires a verified bundle manifest"
  exit 1
fi
uv sync --frozen --extra dev --python 3.12
.venv/bin/python scripts/checkpoint.py
.venv/bin/python -m pytest -q
.venv/bin/chess-self doctor
