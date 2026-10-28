# Recent-game policy

The selected checkpoint continues the existing policy on 99 games from October 2024 through November 2025. Epoch 6 gave the lowest move loss on 39 development games from February through October 2026. It predicts those recorded moves better and makes fewer screened direct material-loss mistakes in the sampled-game comparison. It still blunders; this is not a claim of perfect player imitation or measured Elo.

## Recorded-move agreement

All models were evaluated on the same 935 positions with their recorded ratings and clocks. Lower negative log likelihood (NLL) is better.

| Model | Selected epoch | Move NLL | Top-one agreement | Top-three agreement |
| --- | ---: | ---: | ---: | ---: |
| Previous personalized policy | original | 1.410485 | 54.55% | 80.96% |
| Recent continuation, selected | 6 | 1.364824 | 55.19% | 82.67% |
| Reset move projections, recent training | 10 | 1.373374 | 53.69% | 82.57% |

The selected NLL is 3.24% lower. A paired whole-game bootstrap gives a difference interval of -0.0683 to -0.0237 NLL. Because these games selected the epoch and candidate, this interval does not correct selection optimism and is not independent test performance.

Training used legal-move cross entropy, all recorded player moves including mistakes, AdamW at 0.0001, batch 32, seed 42, at most 12 epochs and patience 3. Only the three move projection weights changed. The warm run stopped after epoch 9 and retained epoch 6. The reset candidate retained the same existing backbone but initialized its move projections from pretrained Maia3. Neither candidate fitted engine moves or used an engine safety filter.

The chronological split is recorded in [audit.json](../../data/processed/recent-policy-v1/audit.json). The former validation role is explicitly retired: some former validation games now train the policy, and the 2026 development games were already used for historical analysis and earlier checkpoint selection. Original reserved test games remain excluded and unscored.

## Sampled gameplay

Stockfish 19 played against each policy at 10,000 nodes per turn. Both policies used rating inputs 680/680 and identical seed lists: 100–111 for the first comparison, then 200–223 for confirmation, alternating colors. Each game was bounded to 100 plies. Clocks were synthetic two-second turns on 10+0 to isolate move policy; timing playback was not exercised.

A screened direct material-loss error requires both an engine loss of at least 200 centipawns and the engine continuation's immediate reply capturing at least three net material points after optimal legal same-square recaptures. Analysis uses 30,000 nodes for best and selected moves with a cleared hash. This is a finite-search diagnostic, not a human-validated definition of an obvious blunder.

| Condition | Games | AI moves | Screened material losses | Rate |
| --- | ---: | ---: | ---: | ---: |
| Previous policy, shipped 491/529 ratings | 12 | 239 | 19 | 7.95% |
| Previous policy, 680/680 | 36 | 819 | 49 | 5.98% |
| Recent policy, 680/680 | 36 | 939 | 35 | 3.73% |

At matched ratings, the observed reduction is 37.7%. The first 12-game comparison was 15/261 versus 10/319; the separate 24-game confirmation was 34/558 versus 25/620. The previous policy lost all 36 games; the recent policy lost 35 and reached the 100-ply cap in one unfinished game. Paired seed-block bootstrap difference interval: -4.29 to -0.29 percentage points. Policies produce different trajectories, so these are not identical-position comparisons; the synthetic clocks, opponent and small seed population limit generalization. The low-rating row has only the first twelve seeds and should not be compared as an equally sized experiment.

The new model is not guaranteed to retain every real human mistake. On a small diagnostic slice of 14 actual 2026 material-loss moves, its NLL worsened from 2.647 to 2.979, and top-one agreement went from 3/14 to 2/14. Training retained those kinds of moves without filtering, but fewer generated hangs comes with some loss of agreement on this particular mistake slice. The sample is too small to calibrate a human blunder rate.

## Timing and runtime

Every nonpolicy backbone tensor was verified exactly equal before rebinding the unchanged E05 timing head. The head uses policy surprisal, so changing move probabilities can still change duration predictions. On the same 896 available timing labels, conditioned on the recorded human moves:

| Timing measure | Previous policy | Recent policy |
| --- | ---: | ---: |
| CRPS, seconds | 3.969419 | 3.969525 |
| Mean prediction MAE, seconds | 5.325905 | 5.323062 |

The CRPS difference is +0.000106 seconds, with whole-game interval -0.00468 to +0.00442; MAE difference is -0.00284 seconds. These are effectively unchanged conditional timing diagnostics, not a guarantee about the timing of newly generated trajectories.

The backbone export passed native/ONNX CPU parity. The timing ONNX bytes are identical to the previous E05 export; its metadata now binds the new policy checkpoint. The packed browser package passed an actual Chrome/WASM worker load and legal move/duration smoke test. [Browser smoke](browser-smoke.json).

## Artifacts

- Native policy: `models/personalized/recent-policy-epoch6.pt`
- Native timing head: `models/personalized/e05-recent-policy.pt`
- Browser policy: `models/export/recent-policy-features.onnx`
- Browser timing: `models/export/e05-recent-policy.onnx`
- Model comparison: [comparison.json](comparison.json)
- Timing comparison: [timing-warm.json](timing-warm.json)
- Gameplay summary: [acceptance-summary.json](acceptance-summary.json)

Training and evaluation ran locally on CPU. Lambda launch was canceled before confirmation; no instance was rented and rental spend was $0.

## Reproduction

Use `scripts/prepare_recent_policy.py` to create a fresh development split, then `chess-self train --policy-only --timing-weight 0` with the settings above. `scripts/recent_policy_job.py --dry-run` prints the bounded two-candidate GPU equivalent; no GPU is required for these small projection-only runs. `scripts/evaluate_policy_candidates.py`, `scripts/play_policy_candidates.py`, `scripts/rebind_policy_timing.py` and `scripts/compare_policy_timing.py` expose explicit checkpoint and output arguments and preserve existing output paths.
