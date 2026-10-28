# Liemfish

**A chess model trained to play like Liem, including the time spent thinking.**

Liemfish learns from my Chess.com and Lichess games: the moves I choose, the positions I hesitate over, and how I use the clock. It personalizes Maia3's move policy, then adds a timing model that conditions on the board, recent thinking times, and the move being played.

The repository includes the training pipeline, game datasets, trained checkpoints, validation reports, and browser-ready ONNX exports. You can inspect the experiments or run the selected model locally.

## Predicting a player

A familiar opening can take a fraction of a second. A difficult capture can consume a large part of the remaining clock. Liemfish models a distribution of thinking times for each position and selected move, so repeated predictions can produce different durations.

The selected timing model reads representations of the move's source and destination squares alongside clock, position, and recent timing features. It adjusts the personalized model's timing distribution while keeping its move policy frozen. At inference, the sampler restricts durations to the available clock budget.

The timing experiments compare duration predictions under the same move policy. The [model contract](docs/timing-model-contract.md) describes the features and distributions; the [experiment guide](docs/timing-experiments.md) covers the alternatives.

## Measured results

The recent-game move policy was adapted on **99 games from 2024–2025** and selected on **39 development games from 2026**. It reduced move prediction loss by **3.24%** relative to the previous personalized policy. First-choice agreement reached **55.19%**, and top-three agreement reached **82.67%**, across 935 positions.

In 36 sampled games per model against Stockfish, screened direct material-loss errors fell from **5.98% to 3.73% of AI moves** at the same rating inputs. The model still blunders, and this finite-search diagnostic is not measured Elo or proof of perfect player imitation. All recorded training moves were retained, including mistakes. [Methods, results and limitations](reports/recent-policy-01/RESULTS.md)

The E05 timing weights are unchanged. On 896 recent timing labels, mean prediction error remained about **5.32 seconds**, with negligible change in the distribution score. These are development results; the original 76-game test set remains untouched.

## Browser inference, verified

The backbone and timing exports pass native-to-ONNX numerical parity checks. The browser package runs both models in a Web Worker using single-threaded WebAssembly.

The two models total **91.76 MB** before transfer compression:

- [Personalized backbone and move policy](models/export/recent-policy-features.onnx): 91.34 MB.
- [Selected E05 timing head](models/export/e05-recent-policy.onnx): 0.42 MB.

Use [@epicliem/liemfish-browser](packages/browser/README.md) to load the models in a Web Worker and predict a legal move with a sampled thinking time. The package provides TypeScript types, load progress, cancellation, and verified asset preparation. Its default runtime is WebAssembly; the separate parity harness also supports WebGPU.

```sh
npm install --save-exact @epicliem/liemfish-browser@0.2.0
```

See the [inference guide](docs/inference.md) for the selected artifacts and native Python API.

## Run it locally

Install Git LFS, then clone the source, weights, and datasets:

```sh
git lfs install
git clone https://github.com/EpicLiem/liemfish.git
cd liemfish
git lfs pull
bash scripts/setup.sh
```

Setup restores the pinned Maia3 implementation and Python dependencies, verifies the base checkpoint, runs tests, and checks the dataset. Python 3 with venv support is required to bootstrap setup; the project environment uses Python 3.12.

Run the personalized move predictor on the included example position:

```sh
.venv/bin/chess-self predict \
  --position docs/example-position.json \
  --checkpoint models/personalized/recent-policy-epoch6.pt
```

This CLI returns legal move probabilities and the personalized reference's timing distribution. To sample a move and duration with the selected E05 timing head, use [`TimingPredictor`](src/chess_self/timing_predict.py) with the [native inference example](docs/inference.md#native-python-api). The example ratings and clocks are illustrative inputs, not an estimate of Liem's playing strength.

Run the checks:

```sh
.venv/bin/python -m pytest -q
node --test browser/*.test.mjs
```

The tests cover clock reconstruction, game-level separation, move encoding, training updates, ONNX parity, and stale-turn cancellation.

## The games behind the model

The dataset starts with 875 public game exports from **MonisVerse** on Chess.com and **MonisChess** on Lichess. Filtering leaves 748 usable games. The original split contains 598 training, 74 validation, and 76 reserved test games. The recent policy uses a new chronological development split of the original non-test games; its 99 training games contain 2,680 player moves. The original training split contains 15,185 player moves and 14,598 usable timing labels. [Preparation audit](data/processed/audit.json)

Whole games stay in one split. Clock reconstruction accounts for increments and missing observations; missing times remain missing. Post-move clocks and elapsed time are labels, never input features for the current prediction. The recent policy fits games through November 2025; 2026 development games select the checkpoint and supply historical rating context.

[Raw exports](data/raw), [prepared splits](data/processed), and [timing features](data/timing-v2) are included through Git LFS where appropriate. The [artifact manifest](artifacts-manifest.json) records sizes and SHA-256 hashes.

## Reproduce or extend the experiments

The [timing experiment guide](docs/timing-experiments.md) covers feature ablations, distribution alternatives, calibration, and evaluation. E03 position, E04 recent timing history, and E05 selected-square experiments have completed full-data runs. The guide lists the available configurations and their input requirements.

Use fresh output directories and preserve the reserved test set until final evaluation.

For a new training run, the [training guide](docs/training.md) and [Lambda runbook](docs/lambda-quickstart.md) document setup, checkpoint collection, and detached jobs.

## Credits and license

Liemfish builds on [Maia3's timing implementation](https://github.com/daniel-monroe/maia3/tree/95c0985aacb62bcaeedc30c3a9bbddfb43f91dd0) and the [Maia3-23M-ponder checkpoint](https://huggingface.co/UofTCSSLab/Maia3-23M-ponder). Setup pins the implementation revision and verifies the checkpoint hash. Chess rules and PGN parsing use [python-chess](https://python-chess.readthedocs.io/).

The source is licensed under [AGPL-3.0](LICENSE), reflecting its use of Maia3 code. The upstream timing weights did not supply a separate model-card license; the code license alone does not establish weight or third-party game-data redistribution terms. Source provenance is retained with the published artifacts.
