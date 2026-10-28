# Chess Self

A personal move-and-clock imitation project based on Maia3-23M-ponder. It learns the moves you actually played, including preferences and mistakes, rather than trying to maximize engine strength.

**Current milestone: completed personalization and three full-data timing experiments.** The selected E05 square-conditioned timing head is the leading development candidate (epoch 5). The original test set remains untouched. See [the next evaluation plan](docs/timing-review-followup.md) for scorecards, uncertainty, matched controls and runtime acceptance.

**Lambda transfer package:** source, pinned weights and prepared splits can be bundled with checksums. Both timing rentals have been terminated. Remote scripts act only when explicitly invoked with an authorized endpoint. Existing transfer archives are snapshots; rebuild after changes. See [the Lambda quick start](docs/lambda-quickstart.md).

## Your data

Signed-in Chrome profiles confirmed **MonisVerse** (Chess.com) and **MonisChess** (Lichess). Public exports were saved locally on October 27, 2026. No browser cookies, credentials, or private messages were copied.

| Stage | Games | Your moves | Timing labels |
|---|---:|---:|---:|
| Downloaded | 875 | — | — |
| Training | 598 | 15,185 | 14,598 |
| Validation | 74 | 1,844 | 1,770 |
| Untouched test | 76 | 1,708 | 1,632 |

Preparation excludes 86 variant games, 10 unsupported time controls and 31 games with missing ratings. Fifteen moves lack usable pre-move clocks. The audit records two inconsistent clock deltas. Missing timing labels are masked; long thinks outside the published head's `[0, 99.9)` second range are not clamped into training labels.

Whole games are sorted by UTC date (game hash breaks same-day ties), then split 80/10/10. Training currently ends November 3, 2024; validation extends to October 15, 2026. This measures adaptation across time, and your changing style can make the task harder. Evaluation reports results after move 10 and on positions unseen in training, as well as control and clock groups. Repeated openings are counted in `data/processed/audit.json`.

## Clone with weights and data

This repository hosts the game exports, prepared splits and selected checkpoints using Git LFS. Install Git LFS before cloning:

```sh
git lfs install
git clone https://github.com/EpicLiem/liemfish.git
cd liemfish
git lfs pull
bash scripts/setup.sh
```

See [the published artifact manifest](artifacts-manifest.json) for file sizes and SHA-256 hashes and [the portfolio integration handoff](docs/portfolio-integration.md) for the selected browser models and license notes. The source is AGPL-3.0; no separate upstream weight license was supplied in the model card. Game data retains its source provenance; the source-code license does not establish ownership of third-party data. Caches, environments, smoke weights and rental/account records are excluded.

## Use locally

The prepared environment is `.venv`. Run from this directory:

```sh
.venv/bin/python -m pytest -q
.venv/bin/chess-self doctor
.venv/bin/chess-self --help
```

To rebuild a fresh environment, run `bash scripts/setup.sh` with Python 3 and venv support available. It bootstraps uv 0.13.6 if needed, verifies the bundle or upstream checkout and pretrained checksum, restores dependencies from `uv.lock`, runs tests, and checks the existing dataset. Linux explicitly uses PyTorch 2.13.0 + CUDA 12.6; macOS uses PyTorch 2.13.0 from PyPI. A code-only checkout needs its games fetched/prepared before the final doctor step can pass.

To acquire a new snapshot, use a **new directory**; the downloader intentionally preserves old snapshots:

```sh
.venv/bin/chess-self fetch chess.com MonisVerse --output data/raw-new
.venv/bin/chess-self fetch lichess MonisChess --output data/raw-new
.venv/bin/chess-self prepare --pgn data/raw-new/*.pgn --players MonisVerse MonisChess --output data/processed-new
```

The fetch manifests record URLs, download dates, game counts and SHA-256 hashes. Preparation labels only these accounts' turns. It keeps all preceding moves as context, both actual ratings and real clock settings. `%clk` values are post-move; elapsed time uses the same player's previous clock plus increment minus the current reading. First-move timing is excluded because platform conventions differ. Missing clocks are never forward-filled.

## GPU handoff

See [docs/gpu-handoff.md](docs/gpu-handoff.md). Start with an NVIDIA CUDA machine; the supplied brief's 16 GB card is a reasonable initial target, but memory use and throughput must be measured on it. No machine has been provisioned and no paid resources have been started.

```sh
bash scripts/train-gpu.sh models/runs/heads-01
```

The initial run freezes the backbone and trains the move and timing heads at `1e-5`, batch 16, up to eight epochs, with early stopping after two validation checks without improvement. Timing cross-entropy has weight 0.2. These are experimental defaults, not validated hyperparameters. Model selection uses validation move log loss; the unchanged checkpoint remains eligible. Inspect timing metrics too before accepting a model. `--final-blocks 1` enables a separate experiment on the last block, with a fresh output directory.

`best.pt` is a complete checkpoint and `run.json` records hashes, settings, baseline, every epoch, selected epoch, and successful/skipped optimizer updates. An epoch-zero selection means the unchanged model won. CPU `--smoke` runs are explicitly marked smoke-only and must not be used as personal models. Test data is not read for training or selection (only its file hash is recorded).

The log prints progress every 50 batches and an estimate to the next validation pass. Epoch durations and throughput are recorded in `run.json`; checkpoints and reports use atomic file replacement. The detached launcher survives an SSH disconnect, but cannot survive termination of the cloud instance. Completed epochs retain the best checkpoint; restarting optimizer state is not implemented.

After choosing an experiment using validation, compare baseline and personal model on the same untouched test set once:

```sh
.venv/bin/chess-self evaluate --checkpoint models/maia3-23m-ponder.pt --positions data/processed/test.jsonl --output reports/test-baseline.json --device cuda
.venv/bin/chess-self evaluate --checkpoint models/runs/heads-01/best.pt --positions data/processed/test.jsonl --output reports/test-personal.json --device cuda
.venv/bin/chess-self export --checkpoint models/runs/heads-01/best.pt --output models/export/personal.onnx
```

Top-1/top-3 agreement and move log loss measure imitation. Timing reports include mean, median and 90th-percentile absolute error, and comparison against medians calculated only from training data. A good aggregate result does not establish performance in sparse controls. This initial version does not rebalance controls or change the timing head's range.

## Prediction and browser export

`chess-self predict --position path/to/position.json --checkpoint path/to/best.pt` returns legal move probabilities and a position-level timing distribution. A position JSON uses the schema in `docs/browser-contract.md`; prepared JSONL rows can be extracted as examples. Predictions reflect the selected checkpoint, not automatically a trained version of you.

`docs/example-position.json` is a generic starting-position example with illustrative ratings and clocks. Try `.venv/bin/chess-self predict --position docs/example-position.json`.

The pretrained export is `models/export/maia3-23m-ponder.onnx`, with its ordered move vocabulary and parity report alongside it. [The browser contract](docs/browser-contract.md) describes history order, orientation, clocks, ratings, move indices and timing outputs. **ONNX Runtime CPU parity is verified; WebAssembly/WebGPU execution and browser latency are not.** A browser worker, model cache and chess UI remain the next product milestone after a useful model is selected.

## Project contents and provenance

- `src/chess_self/`: fetching, data preparation, model adapter, training/evaluation, export and CLI.
- `tests/`: clock accounting, leakage prevention, move encoding, parameter updates, fetch behavior and export checks.
- `docs/`: original background document, browser contract and GPU runbook.
- `data/`, `models/`, `reports/`: local artifacts, excluded from Git.
- `sources/maia3/`: unmodified author implementation at `95c0985aacb62bcaeedc30c3a9bbddfb43f91dd0`.

The checkpoint is pinned to Hugging Face revision `34ec2be9d651102e4ad0e9eda3d838acd69f5f9a`, SHA-256 `0ac9fb74256f9be7d74808404364b90130c4d193f999c45b01e9b5ef64281bd1`. It is loaded strictly with PyTorch `weights_only=True`. The original implementation's AGPL-3.0 license is preserved in `LICENSE`; no model-card license was supplied for the timing weights.

Sources: [author timing implementation](https://github.com/daniel-monroe/maia3/tree/95c0985aacb62bcaeedc30c3a9bbddfb43f91dd0), [ponder checkpoint](https://huggingface.co/UofTCSSLab/Maia3-23M-ponder), [python-chess clocks](https://python-chess.readthedocs.io/en/latest/pgn.html), [Chess.com export API](https://www.chess.com/news/view/published-data-api), [Lichess export API](https://lichess.org/api#tag/Games/operation/apiGamesUser).

## Timing experiment suite

The E01–E15 runner, configurations, input requirements, acceptance checks and GPU commands are documented in [docs/timing-experiments.md](docs/timing-experiments.md). Start with `python -m chess_self.experiments --help`. The completed personalized reference is `models/personalized/heads-01.pt`; new timing experiments preserve the original benchmark and reserved test games.
