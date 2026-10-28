# Liemfish

**A chess model trained to play like Liem, including the time spent thinking.**

Liemfish learns from my Chess.com and Lichess games: the moves I choose, the positions I hesitate over, and how I use the clock. It personalizes Maia3's move policy, then adds a timing model that conditions on the board, recent thinking times, and the move being played.

The repository includes the training pipeline, game datasets, trained checkpoints, validation reports, and browser-ready ONNX exports. You can inspect the experiments or run the selected model locally.

## Predicting a player

A familiar opening can take a fraction of a second. A difficult capture can consume a large part of the remaining clock. Liemfish models a distribution of thinking times for each position and selected move, so repeated predictions can produce different durations.

The selected timing model reads representations of the move's source and destination squares alongside clock, position, and recent timing features. It adjusts the personalized model's timing distribution while keeping its move policy frozen. At inference, the sampler restricts durations to the available clock budget.

That separation makes the experiment inspectable: improvements in timing do not come from quietly changing which moves the model chooses. The [model contract](docs/timing-model-contract.md) describes the features and distributions; the [experiment guide](docs/timing-experiments.md) covers the alternatives.

## Measured results

Personalization raised first-choice agreement with Liem's recorded moves from **50.76% to 53.47%** on 1,844 validation positions. Top-three agreement reached 81.51%. These measure imitation of the recorded player, not chess strength. [Move results](reports/training-summary.json)

The selected square-conditioned timing head, E05 at epoch 5, reduced mean-time absolute error by **4.85%**, from 5.108 to 4.860 seconds. Its full-distribution score, CRPS, improved by 2.54%; lower CRPS means the predicted timing distribution fits the observed durations better. Evaluation covered 1,770 timed moves across 74 games, with comparisons resampled by whole game. [Timing results and uncertainty](reports/timing-full-evaluation-v1/RESULTS.md)

These are development results on validation games used for model selection. The 76-game test set remains untouched. Clock-pressure behavior still needs work: the below-ten-second subgroup is small, and its improvement interval crosses zero.

## Browser inference, verified

The selected ONNX pair has run in Chrome through both WebGPU and WebAssembly. Both passed numerical parity checks for white-to-move, black-to-move, low-clock, and batch-two fixtures. WebGPU single-position warm medians were **23.0 to 23.6 ms** on the development Mac; WASM medians were 57.8 to 59.9 ms. These are fixture measurements, not a cross-device benchmark. [Browser reports](reports/timing-browser-selected/summary.json)

The two models total **91.76 MB** before transfer compression:

- [Personalized backbone and move policy](models/export/personalized-timing-features.onnx): 91.34 MB.
- [Selected E05 timing head](models/export/e05-selected-epoch5.onnx): 0.42 MB.

The browser folder contains a parity harness and a turn scheduler that accounts for inference time, cancels stale moves, and checks clock expiry before committing. A complete playable browser game is still an integration task. The native Python predictor already accepts a position and returns a legal sampled move with a target turn duration. See the [integration handoff](docs/portfolio-integration.md) for the exact inputs, outputs, and remaining browser adapter work.

## Run it locally

Install Git LFS, then clone the source, weights, and datasets:

```sh
git lfs install
git clone https://github.com/EpicLiem/liemfish.git
cd liemfish
git lfs pull
bash scripts/setup.sh
```

Setup restores the pinned Maia3 implementation and Python dependencies, verifies the base checkpoint, runs tests, and checks the dataset. It does not start training or rent a GPU. Python 3 with venv support is required to bootstrap setup; the project environment uses Python 3.12.

Run the personalized move predictor on the included example position:

```sh
.venv/bin/chess-self predict \
  --position docs/example-position.json \
  --checkpoint models/personalized/heads-01.pt
```

This CLI returns legal move probabilities and the personalized reference's timing distribution. To sample a move and duration with the selected E05 timing head, use [`TimingPredictor`](src/chess_self/timing_predict.py) with the [native inference example](docs/portfolio-integration.md#existing-native-requestresponse-contract). The example ratings and clocks are illustrative inputs, not an estimate of Liem's playing strength.

Run the checks:

```sh
.venv/bin/python -m pytest -q
node --test browser/*.test.mjs
```

The current suite passes 190 Python tests and 18 JavaScript tests. It covers clock reconstruction, game-level separation, move encoding, training updates, ONNX parity, and stale-turn cancellation.

## The games behind the model

The dataset starts with 875 public game exports from **MonisVerse** on Chess.com and **MonisChess** on Lichess. Filtering leaves 748 usable games, split chronologically into 598 training, 74 validation, and 76 reserved test games. Training contains 15,185 player moves and 14,598 usable timing labels. [Preparation audit](data/processed/audit.json)

Whole games stay in one split. Clock reconstruction accounts for increments and missing observations; missing times remain missing. Post-move clocks and elapsed time are labels, never input features for the current prediction. Training ends in November 2024, so this checkpoint represents historical play.

[Raw exports](data/raw), [prepared splits](data/processed), and [timing features](data/timing-v2) are included through Git LFS where appropriate. The [artifact manifest](artifacts-manifest.json) records sizes and SHA-256 hashes. It was verified against a fresh GitHub clone after publication.

## Reproduce or extend the experiments

The [timing experiment guide](docs/timing-experiments.md) covers feature ablations, distribution alternatives, calibration, and evaluation. E03 position, E04 recent timing history, and E05 selected-square experiments have completed full-data runs. Additional configurations are experimental options, not claims of completed training.

Start with the [review follow-up](docs/timing-review-followup.md) for the prepared matched control and outstanding acceptance checks. Use fresh output directories and preserve the reserved test set until final evaluation.

For a new training run, the [GPU handoff](docs/gpu-handoff.md) and [Lambda runbook](docs/lambda-quickstart.md) document setup, checkpoint collection, and detached jobs. Remote scripts act only when explicitly invoked.

## Credits and license

Liemfish builds on [Maia3's timing implementation](https://github.com/daniel-monroe/maia3/tree/95c0985aacb62bcaeedc30c3a9bbddfb43f91dd0) and the [Maia3-23M-ponder checkpoint](https://huggingface.co/UofTCSSLab/Maia3-23M-ponder). Setup pins the implementation revision and verifies the checkpoint hash. Chess rules and PGN parsing use [python-chess](https://python-chess.readthedocs.io/).

The source is licensed under [AGPL-3.0](LICENSE), reflecting its use of Maia3 code. The upstream timing weights did not supply a separate model-card license; the code license alone does not establish weight or third-party game-data redistribution terms. Source provenance is retained with the published artifacts.
