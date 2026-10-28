# Full validation and selected E05 browser checks

Validation evaluation dated October 27, 2026. Reserved test examples were not evaluated.

## Validation

All 33 planned stages completed: 1,844 positions, 1,770 timed moves across 74 games; four models, raw/feasible support, 256/1,024-point CRPS, and 2,000-resample paired whole-game bootstrap comparisons. Checkpoint hashes, commands and completion are in provenance.json. Per-row and control/clock/repeated-position scorecards are in the corresponding condition directories.

| Model | Selected epoch | Feasible CRPS, 1,024 points | Mean-time MAE, seconds |
| --- | ---: | ---: | ---: |
| Reference | original | 3.587096 | 5.107767 |
| E03 position | 4 | 3.554842 | 4.987367 |
| E04 timing history | 4 | 3.530224 | 4.927145 |
| E05 selected squares | 5 | 3.496083 | 4.859893 |

Lower scores are better. E05 improves CRPS by 0.091013 seconds (2.54%); the game-bootstrap 95% interval for candidate minus reference is [-0.122205, -0.062476]. Against E04 the difference is -0.034141, interval [-0.057644, -0.013386]. Mean-time MAE improves by 0.247874 seconds (4.85%). Move policy remains unchanged.

Increasing CRPS resolution from 256 to 1,024 points changes absolute scores by about 0.00008 seconds and the E05-reference gain by about 0.0000034 seconds. This numerical approximation does not explain the measured gain.

The below-ten-second subgroup improves by 0.010843 CRPS seconds, but its interval [-0.077883, 0.059925] crosses zero (49 moves, 12 games). Bullet improves by 0.048423, interval [-0.080912, -0.005967], with only six games and 130 timed moves. These are development intervals after checkpoint selection on the same validation data; they do not eliminate selection bias or establish future-game gains.

One long label remains outside all models' support. Whole-sample interval and joint NLL are undefined. The finite interval-NLL comparison covers 1,769 observations, excluding 74 missing timing labels and one support failure; it must not be presented as whole-sample NLL.

Raw impossible-duration mass below ten seconds decreases from 29.24% to 24.37% under this continuous evaluator. Truncation removes it by construction. Training has not removed the need for clock support constraints. The training data remains historical; these evaluations add no recent games.

## Actual browser execution

Headless Google Chrome 156.0.8078.48 ran the existing browser harness with the selected full E05 export, matching feature backbone, and explicit white/black/low-clock/batch-two fixtures. Both WASM and WebGPU passed all four cases and all five output tensors, with three warm repetitions per case. Browser-loaded model/fixture hashes matched the local files.

- WASM single-position warm medians: 57.8–59.9 ms; batch two: 109.5 ms.
- WebGPU warm medians: 23.0–23.6 ms; batch two: 23.0 ms.
- Local scheduler: one dummy commit at 1.0013 seconds (WASM), 1.0009 seconds (WebGPU), against a one-second target.

Reports: ../timing-browser-selected/wasm.json and webgpu.json. Measurements are from this Mac and fixture workload; per-operator GPU placement and full live gameplay are not certified. The scheduler fixture uses a supplied legal move and target, not an end-to-end model-generated game.

## Reproduction



```bash
python3 scripts/evaluate_selected_timing.py --run --output reports/NEW_FRESH_OUTPUT
node scripts/check_selected_browser.mjs
```

Browser runner uses installed Chrome and Playwright; environment overrides CHROME_EXECUTABLE and PLAYWRIGHT_MODULE select alternatives. Existing measured browser report paths are overwritten by a rerun; copy them first to preserve this measurement.
