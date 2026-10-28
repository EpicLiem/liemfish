# Chess Self

- Python package in `src/chess_self`; run commands from this directory using `.venv/bin/python` or `.venv/bin/chess-self`.
- `bash scripts/setup.sh` restores pinned Maia3 source, dependencies and checkpoint. `uv sync --frozen --extra dev` uses `uv.lock`.
- Run `.venv/bin/python -m pytest -q`. Use a fresh output directory for every training or smoke run.
- `sources/maia3` is an unmodified upstream checkout pinned to 95c0985aacb62bcaeedc30c3a9bbddfb43f91dd0. Preserve its history orientation and move vocabulary.
- The user authorized publication of game datasets and trained weights. Data and selected weights use Git LFS; rental metadata, credentials, caches and unselected reports remain ignored. Preserve the published artifact allowlist.
- Keep game-level holdouts intact; never use test data for training decisions. Elapsed and post-move clock values are labels, never model features.
- This project's current milestone is GPU training readiness. CPU smoke artifacts are not personalized models. Report browser parity separately from native ONNX parity.
- `python3 scripts/bundle.py` builds the allowlisted transfer archive; verify it after extraction. Rebuild after source/data edits. Linux torch uses the explicit CUDA 12.6 index in the lock.
- `scripts/remote.py` is opt-in SSH deployment/collection; never invoke it without an authorized endpoint. `scripts/jobs.py start NAME --mode full` runs detached setup, GPU preflight and training. Inspect `jobs/NAME/status.json` and `output.log`.
