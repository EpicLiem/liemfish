# Chess Self

- Python package in `src/chess_self`; run commands from this directory using `.venv/bin/python` or `.venv/bin/chess-self`.
- `bash scripts/setup.sh` restores pinned Maia3 source, dependencies and checkpoint. `uv sync --frozen --extra dev` uses `uv.lock`.
- Run `.venv/bin/python -m pytest -q`. Use a fresh output directory for every training or smoke run.
- `sources/maia3` is an unmodified upstream checkout pinned to 95c0985aacb62bcaeedc30c3a9bbddfb43f91dd0. Preserve its history orientation and move vocabulary.
- Personal archives, processed examples, trained weights and reports stay local and are ignored by Git. Do not upload or publish them without permission.
- Keep game-level holdouts intact; never use test data for training decisions. Elapsed and post-move clock values are labels, never model features.
- This project's current milestone is GPU training readiness. CPU smoke artifacts are not personalized models. Report browser parity separately from native ONNX parity.
