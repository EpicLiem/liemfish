# Personal imitation experiments: October 2026

The public-data pilot and three-seed repetitions did not establish a useful improvement in personal move prediction. Liemfish therefore retains the released reference policy. The experiment plan required useful personal behavior gains before escalating to larger adversarial training runs. Those criteria were not met.

This report covers the experiments completed on 28 October 2026. The accompanying [results JSON](imitation-v1-results.json) contains the numerical comparisons, reference-checkpoint hashes, and hashes of the underlying reports. Full checkpoints and per-position records remain in the retained experiment archive.

Move negative log-likelihood (NLL) measures how much probability the model assigns to the recorded human move. Continuous ranked probability score (CRPS) measures the quality of the full timing distribution. Lower values are better for both; a difference below zero favors the candidate. The discriminator uses balanced binary cross-entropy (BCE) to distinguish human moves from reference-policy samples in the same positions.

## Move prediction

The screen compared matched supervised and balanced-BCE heads on 200,000 public training contexts, then fitted small personal residuals on 2,680 moves from 99 games. Three public and three personal learning rates were screened with seed 42. The selected rates were frozen before independent seeds 17, 29, and 43. Population-only, scalar-bias, and personal-only controls are retained in the run artifacts.

| Recipe | Screen correction strength | Replication strengths: 17 / 29 / 43 |
|---|---:|---|
| Supervised correction | 0 | 0 / 0 / 0 |
| Human-versus-policy discriminator | 0 | 0 / 0.25 / 0 |

Zero strength reproduces the reference exactly. The one nonzero replicated correction reduced late-development move NLL from **1.628378 to 1.628029**, a **0.0214%** point improvement. Its paired 95% interval for candidate-minus-reference NLL was **-0.002643 to +0.002557 nat/move**. This fails the proposed 1% improvement target and does not establish improvement. Calibration selected the strength on these same games, so the interval also does not correct selection optimism.

On 56 stable large mistakes in the calibration split, that candidate's NLL difference was -0.004855, with interval [-0.010616, +0.001409]. It satisfied the proposed 0.05-nat noninferiority margin on this slice, but this does not turn the negligible overall gain into a useful result. Its marginal timing CRPS changed from 3.504858 to 3.504872 seconds, effectively unchanged and slightly worse by point estimate.

The separate policy-projection control selected epoch 6. Its calibration NLL difference was +0.002436 [-0.004839, +0.011996], also not an improvement. Its frozen nonpolicy tensors were verified unchanged and final forecasts were evaluated with native FP32 encoder features.

Several nonzero corrections modestly improved point likelihood on actual mistakes while worsening overall move prediction. These results do not justify a blanket filter against hanging pieces, or show that removing real personal blunders improves imitation.

## Timing

The selected E05 reference, which conditions timing on the chosen move’s source and destination squares, had late-development marginal CRPS **3.504852 seconds** using the common final integration. Lower is better. Full per-arm results include early development, low-clock subsets, support failures, and paired uncertainty.

| Candidate | Calibration CRPS | Difference from selected E05 |
|---|---:|---:|
| Original Maia timing, retained by many refit arms | 3.607146 | +0.102293 |
| E04 recurrent history | 3.613059 | +0.108207 |
| E07 Beta time fraction | 3.930429 | +0.425577 |
| E09 all older training games | 3.563673 | +0.058821 |
| E10 public timing plus personal residual | 3.549756 | +0.044904 |
| E06 fixed 1% tail initialization | 3.501996 | -0.002857 |
| Training-fitted constant tail prior | 3.504513 | -0.000339 |

All 21 timing variants covering feature sets, history, duration distributions, and training-data choices completed. E10 additionally trained a public head, a personal residual, and a personal-only comparator. Public data yielded 192,103 valid training timing observations and 19,230 development observations. Missing/invalid timing and nonpositive clocks were excluded explicitly; long observations outside the duration support were retained for evaluation. E10 final comparisons were rescored at the same 1,024-point integration setting as the other arms.

No fitted timing head beat the selected E05 reference on calibration. A refit arm retaining epoch 0 retains the original Maia timing logits, not the already-trained E05 model. E06 is different: it starts from frozen selected E05 plus a fixed tail. Its selected epoch 0 is not a learned gain.

The 1% tail's calibration CRPS interval was [-0.007993, +0.000775], inconclusive. There was only one training observation above 99.9 seconds among 2,581 eligible labels. Fitting one constant tail gate on training intervals gave mass **0.000829998**. Its CRPS difference was -0.000339 [-0.000767, -0.000041], approximately **0.0097%** of baseline CRPS. This is a small support correction, not meaningful evidence of improved personal rhythm. No browser replacement was made.

The original 30-bin timing law cannot score the 180.3-second early-development observation with positive interval probability. Extending support fixes that failure; finite-only likelihood must not be presented as the full original likelihood. Calibration has no observation above 99.9 seconds. E02 selected temperature 1, leaving the existing timing calibration unchanged.

## Data, human mistakes, and runtime

- Personal splits: 99 training games / 2,680 moves; 24 early-development games / 606 moves; 15 calibration games / 329 moves. The two development groups contain 12 and 4 conservatively grouped sessions. They were already used in earlier model work and remain reused development.
- Public pilot: 200,000 training and 20,000 development contexts, balanced at 40% blitz, 40% rapid, 20% bullet, with held-out player partitions. The separate 20,000-context test and original 76 personal test games were not scored.
- Public source data consists of bounded chronological prefixes from October to December 2025 archives. Prefix hashes are verified; this is not a uniform sample of complete months, and full-archive hashes were not claimed verified.
- The pinned Stockfish audit covered all 935 development moves in 39 games, with 379 higher-budget rechecks. It found 119 stable large-loss/mate-deterioration moves and 30 stable direct-material-screen moves; 37 threshold-unstable cases were excluded from the stable slices. The historical 14-move slice remains separate. Engine loss and mechanical capture exposure do not establish human noticeability.
- E01 replay completed 48 paired virtual games each with sampled, mean, and median controlled-player timing. Opponents, starts, colors, and seeds were fixed. Different generated clocks can change later trajectories. Zero simulated flags are structural under feasible completed-move sampling, not evidence of a realistic timeout rate.
- The current 0.2.0 browser engine was tested on 1,000 unique positions after 50 warmups, plus six edge fixtures, using Chrome/WASM on the M3 Max. All outputs were legal and clock-feasible. Median inference was 61.5 ms and p95 was 66 ms; eight requested durations were shorter than inference. This measures the existing engine, not a newly exported discriminator.

## Paired games for the nonzero candidate

The BCE seed-29 correction at strength 0.25 completed another 48 CPU-only games after GPU training, paired against the reference by start, opponent, color, and RNG stream. Independent replay verified all 3,401 candidate moves and 3,148 reference moves were legal, contexts matched the evolving boards, and charged durations stayed within available clocks.

The rate at which the moved piece was immediately capturable was 21.76% for the candidate versus 23.86% for the reference. The paired difference was **-2.10 percentage points**, with 95% interval **[-4.78, +0.58]**: inconclusive. This is mechanical exposure, not engine-verified net material loss or human noticeability, and no excess gap relative to actual human play was established. Mean controlled clock consumption per game was 278.51 versus 273.17 seconds; the paired interval for the difference was [-35.88, +48.50] seconds.

Native CPU panel wall time was 85.52 versus 40.33 seconds. These batched totals are not browser latency measurements. The candidate remains experimental and does not change the stopping decision. The results JSON includes the paired-game comparison.

## Stopping decisions

The plan explicitly calls for simple-recipe repetitions and a stop if the detector adds no useful personal behavior improvement over the supervised control. That stopping condition applies here.

No two-million-context scale-up, alternating policy updates, distillation, rollout GAIL, five-seed finalist expansion, or release replacement was performed. A05 successor/encoder/auxiliary experiments require a specific representational failure or validated tactical labels; those were not established by this pilot. The supplementary all-legal tactical comparator, familiarity retrieval variant, and engine-difficulty predictor were not executed in this stage. Interpretive tactical labels were not fabricated from engine scores.

A09 joint move-and-time discrimination remains blocked on a validated observation model and adequate support. E13 backbone adaptation did not receive a useful frozen-head result to adapt. E14 timeout modeling was deferred because these records do not reliably identify unfinished-turn causes. Neither locked test was opened for scoring.

## Verification and compute

The archived experiment snapshot passed 373 tests with `.venv/bin/python -m pytest -q --disable-warnings`; 25 export-related warnings remained. The source bundle passed extraction verification for all 183 files, and all 20 recorded experiment-source hashes matched between the GPU machine and local checkout. These checks are separate from the native/ONNX parity, real CUDA training, and browser measurements reported above.

The run used one A10 24GB at $1.29/hour because A100 capacity was unavailable. The conservative submission-to-confirmed-termination estimate was **$1.71**, within the $20 budget. This is an elapsed-time estimate, not an invoice. The provider confirmed no running instances after termination.

Before termination, all 18,246 collected files, including 198 `.pt` artifacts, passed remote/local SHA-256 verification. The archive retains checkpoints, optimizer state, training records, reports, and logs. Reproducible feature tensors were omitted; their manifests, source data, and reference checkpoints were retained.
