# Selected timing models: next evaluation

The completed development runs selected E03 epoch 4, E04 epoch 4, and E05 epoch 5. Use their `best.pt` files in `reports/collected/chess-self-timing-retry/models-runs/timing-01/`. `last.pt` is not the selected model. No original test examples are needed for this work.

## Full scorecards and uncertainty

```bash
python3 scripts/evaluate_selected_timing.py --dry-run
python3 scripts/evaluate_selected_timing.py --run
```

The default is a read-only plan. Explicit `--run` creates a fresh local output directory, caches all 1,844 validation positions and runs 33 stages: reference and three selected heads, raw and feasible timing, 256- and 1,024-point CRPS, and paired game bootstrap comparisons. It saves input hashes, selected epochs, commands and progress. It neither trains nor rents hardware. CPU is the default; `--device cuda` affects caching on an already available machine.

Scorecards retain per-position scores, control/clock groups, MAE, interval and joint NLL, calibration, support failures and training-only repeated/unseen groups. Position identity uses the first four FEN fields. The index is reporting-only and never includes validation boards as training examples. Raw and feasible reports distinguish learned tails from runtime truncation. Timing measurement intervals still assume nearest rounding inferred from printed clock digits; that observation model remains unverified.

Comparison intervals use whole-game bootstrap resampling. Every metric states its common-finite population and excluded-pair count. NLL also reports each side's zero-probability count: a finite-subset interval does not make whole-sample NLL finite. Inspect the 256/1,024 score difference before interpreting small gains. These are development comparisons after checkpoint selection, not independent test evidence.

## Matched refit control

`configs/timing/e03-refit-control.json` uses pooled backbone features and no new side inputs, a 64-unit hidden layer, natural sampling and interval likelihood. Match the completed optimization budget explicitly:

```bash
.venv/bin/python -m chess_self.experiments train --train-cache data/timing-cache/train --val-cache data/timing-cache/val --config configs/timing/e03-refit-control.json --device cuda --epochs 8 --batch-size 32 --learning-rate 0.001 --weight-decay 0.01 --patience 2 --seed 42 --output models/runs/e03-refit-control-seed42
```

Prepare the caches with the existing guide first. This control matches the objective, hidden width, selection guards and optimization budget, but has 1,152 fewer input weights than E03. It does not isolate all capacity effects. It has not been fitted. Additional seeds should use fresh directories and the same predeclared settings.

## Selected E05 runtime artifacts

The selected full-data head has been exported to `models/export/e05-selected-epoch5.onnx`, with its native CPU ONNX parity report alongside it. Its checkpoint is hash-bound to `models/personalized/heads-01.pt`. Use the existing `models/export/personalized-timing-features.onnx` backbone and `reports/e05-selected-browser-fixtures.json`.

Fixtures explicitly select distinct white and black positions, a minimum-clock case and a two-position batch. Each case records game, ply, FEN, color and move. The harness's scheduler points to the explicit white case. The full head replaces the old smoke-only fixture head. Native ONNX parity is separate from browser WASM/WebGPU execution; actual browser acceptance remains to be measured using `browser/index.html` and the local-file controls described in `timing-runtime.md`.

## Generated sequences

Prepare matched reference and selected-E05 replay/virtual reports using the same validation input, initial controls, Elo context, seeds and game budget. Start with the commands in `timing-experiments.md`, use fresh outputs, and retain both mean/median/sample playback modes. Recorded replay is a one-step check; virtual games feed generated clocks/history back into prediction. Browser achieved delays are a third, distinct measurement. No saved smoke rollout establishes full-game realism.

## Extended support

E06 now normalizes original logits before appending low-mass tail logits. A common shift of the original logits cannot change the tail prior. Absolute 30-bin heads and the three completed runs are unchanged. Extended checkpoints carry explicit initialization semantics; old unnormalized E06 smoke checkpoints must be regenerated rather than silently reinterpreted.
