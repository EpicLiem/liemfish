# Timing audit: heads-01

Analyzed the saved epoch-8 checkpoint locally on all 1,844 validation positions from 74 games. 1,770 positions have timing labels; one exceeds the head's 99.9-second training range. Test data was not evaluated. No training or remote instance was started. Training-label statistics alone supply the simple baselines.

## Conclusion

The timing head has not collapsed to a constant or an unconditional time distribution. It uses board/history information and beats training-only pace baselines. However, using its expected time as an actual move delay compresses the observed variation, and its behavior under severe clock pressure is weak. The public inference function exposes a distribution and an expected time, but does not implement a clock-safe stochastic playing policy.

## Measurements

- Constant training mean: 6.65s validation MAE; constant median: 6.52s.
- Training median by time control and clock band: 6.26s.
- Canonicalized control + clock band + game phase median, falling back to control median for fewer than 30 training labels: 6.17s.
- Pretrained expected time: 5.47s; personalized expected time: 5.14s.
- Personalized distribution median: 4.93s, without retraining.
- Personalized expected-time correlation with actual time: 0.558. Predictions range from 1.69s to 26.03s.
- Randomly replacing board/history with another validation board/history of the same time control, preserving recipient clock and ratings, worsens MAE to 7.25s. Mean prediction changes by 4.31s. This demonstrates dependence on board/history, not specifically tactical understanding: swapping boards also changes game phase and can create unusual board/clock combinations.
- Timing bucket negative log likelihood: personalized 2.650, training control/clock histogram 2.903, scrambled boards 3.137 (lower is better). Histograms use one pseudocount per bin; only in-range labels enter this metric.
- Whole-game bootstrap, 2,000 resamples: personalized mean MAE minus control/clock median MAE is -1.118s, 95% interval [-1.397,-0.835]. Bucket NLL advantage is -0.252, interval [-0.292,-0.210]. This is descriptive uncertainty on the validation games, not independent test evidence; these games were also used for epoch selection.

## Where it fails

- Moves actually under 2s: actual average 1.17s, predicted expected time 4.65s (370 labels).
- Moves actually at least 20s: actual average 31.20s, predicted expected time 14.03s (202 labels). Personalization worsens MAE in this subgroup from 16.28s to 17.16s.
- Under 10s on clock: expected-time MAE 2.00s; distribution-median MAE 1.24s; training control/clock median baseline 0.82s (49 labels).
- One-minute games: expected-time MAE 1.82s; distribution-median MAE 1.41s; simple baseline 1.44s (130 labels across six games).
- Ten-minute games: expected-time MAE 6.10s versus baseline 7.36s. These supply 1,051 of 1,770 labels, so the pooled 5.14s is dominated by slower games.
- Expected time exceeds remaining clock in 18 of 1,844 positions. Current prediction code does not constrain support by remaining clock.

Grouping by the realized outcome will itself show regression toward the mean even for a well-calibrated probabilistic model. These fast/slow averages are evidence that a deterministic mean is unsuitable for imitation, not proof that the distribution has forgotten fast moves or long thinks. Indeed, the personalized distribution assigns approximately 10.8% aggregate probability to bucket centers >=20s, versus 11.4% actual >=20s moves. This threshold comparison is approximate because bins straddle thresholds; it does not establish conditional calibration.

## Design findings

The model sees up to eight boards, time control, increment, remaining own clock and both ratings. It learns a 30-bin log-spaced time distribution with cross entropy, not a regression to average elapsed seconds. Published inference computes an expectation across those bins.

This run froze the transformer and trained output heads plus final normalization. It has no explicit recent move-time sequence or opponent clock input. Move and timing heads share state features, but timing is not explicitly conditioned on the sampled move. These are limitations to investigate, not demonstrated causes of every timing error.

Checkpoint selection uses validation move NLL alone. Timing needs its own acceptance criteria. In this run epoch 7 had timing MAE 5.141s versus epoch 8's 5.142s, so selection did not cause a material timing regression here.

## Concrete next changes

1. Preserve the time distribution for playing and sample delays from it, with an explicit safe clock budget and a policy for out-of-support/near-zero times. Keep expectation and median as diagnostics, not automatic playing delays. Sampling alone can increase per-move MAE; judge imitation using conditional distribution scores, calibration, fast-move rates and long-think rates.
2. Add timing-specific acceptance gates, especially bullet and under-10-second groups, alongside move accuracy. Include distribution NLL and a continuous distribution score, train-only baselines, and per-game uncertainty. Do not silently trade timing for move accuracy.
3. Before another GPU rental, investigate training-only features for recent own pace and opponent clock, and build a timing-specific calibration/head candidate. Compare on validation with chronology and label isolation preserved. Partial backbone unfreezing is an experiment to justify, not an automatic fix.
4. Use a fresh future-game holdout for final acceptance once the design is fixed. The untouched test set remains untouched during this analysis.

## Reproduction

Run from the project root:

```
.venv/bin/python reports/analyze_timing.py
.venv/bin/python reports/summarize_timing.py
.venv/bin/python reports/timing-extra.py
```

All completed locally. The final command asserts probability normalization, 1,844 prediction rows, and reproduction of the saved GPU timing MAE within 1e-4 seconds. CPU result: 5.142102665s; saved GPU result: 5.142102663s. No model or application source was changed. Raw probabilities, metrics, analysis scripts and this report are saved locally under reports/.
