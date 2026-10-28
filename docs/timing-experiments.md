# Timing experiment runner

For the completed E03/E04/E05 runs and prepared next checks, see [the review follow-up](timing-review-followup.md).

The reference is `models/personalized/heads-01.pt` (SHA-256 `8758ea33f911233b28c2c134192a3f127b5c7669e22c08178417e41393c40427`). It remains unchanged. New experiments write fresh output directories. The original benchmark and its reserved test games remain intact.

Run from the project root using `.venv/bin/python -m chess_self.experiments`. `chess-self timing ...` is also available. No command rents a machine or launches a remote job.

## Prepare once

The enriched dataset is already available locally at `data/timing-v2`. To reproduce into a new location:

```bash
.venv/bin/python -m chess_self.experiments prepare --pgn data/raw/*.pgn --players MonisVerse MonisChess --reference data/processed --output data/timing-v2-new
.venv/bin/python -m chess_self.experiments audit --data data/timing-v2 --output reports/timing-label-audit.json
```

The dataset adds opponent clock, prior own/opponent durations and missingness, original precision, full timestamps, source, ending metadata, and causal familiarity. It preserves legacy train/validation identities, order and labels exactly; test IDs are read from audit metadata, not test examples. Rounding intervals assume independent nearest rounding of the two clock readings. This is a declared sensitivity assumption, not a verified platform observation likelihood. Zero clock deltas are not labeled premoves.

## Cache and run on a GPU machine

Frozen head experiments share the same representation caches. Replace `cuda` with `cpu` for local work. Cache manifests bind the row contents and shard hashes to the exact backbone checkpoint. Squares are stored in float16, other tensors in float32; train and validation use the same cache convention. Memory-mapped shards avoid repeated whole-cache loads.

```bash
.venv/bin/python -m chess_self.experiments cache --positions data/timing-v2/train.jsonl --checkpoint models/personalized/heads-01.pt --device cuda --output data/timing-cache/train
.venv/bin/python -m chess_self.experiments cache --positions data/timing-v2/val.jsonl --checkpoint models/personalized/heads-01.pt --device cuda --output data/timing-cache/val
.venv/bin/python -m chess_self.experiments train --train-cache data/timing-cache/train --val-cache data/timing-cache/val --config configs/timing/e03-position.json --device cuda --output models/timing/e03-position
```

`--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`, `--patience` and `--seed` are explicit optimization controls. Match these when comparing arms. Frozen timing training leaves move probabilities unchanged. Use `--limit-games 2` for cache smoke checks, then `train --smoke --epochs 1`; limited caches cannot silently become full experiments.

Each run saves `initial.pt`, `last.pt`, `run.json`, and **only if accepted** `best.pt`. Acceptance requires marginal validation CRPS improvement over both the initial candidate and original reference, with no configured move-NLL or bullet/under-ten-second CRPS regression. Missing validation subgroups are reported as absent, not passed evidence. Set tolerances in the copied configuration before running; defaults are zero. `last.pt` is an experimental artifact, not an accepted model. Smoke fits do not establish quality.

`--initialize prior/best.pt` reuses a compatible timing head, with a newly initialized optimizer. This is weight continuation, not exact optimizer resumption. Feature order, architecture and backbone hash must match. Prior head training games cannot overlap new validation games.

## Experiment coverage

| ID | Runnable arms / commands | Required inputs or interpretation |
| --- | --- | --- |
| E01 | `playback --mode mean`, `median`, `sample`; with/without `--unconstrained` and measured `--reserve-seconds`; `evaluate` raw vs feasible | Same weights. Absolute bins are log-uniform in `t+0.1`, fraction bins uniform in fraction. Partial bins are integrated before normalization. Upcoming increment is never available early. |
| E02 | `calibrate --mode temperature`, `bias`, `regime`, `histogram` | Separate held-out calibration and validation caches with predictor training-game provenance; no in-sample fitting. Histogram is a smoothed global calibration-slice arm. |
| E03 | `e03-clock`, `e03-clock-phase`, `e03-position`, `e03-tabular` configs | Same frozen head baseline. Tabular arm omits pooled features from the residual, retaining base timing logits as prior. |
| E04 | `e04-summary`, `e04-window`, `e04-gru` | All see at most the same eight previous durations per player, with missingness. No current duration or full-game mean is an input. |
| E05 | `e05-attributes`, `e05-squares` | Train on human moves; select by exact legal-policy marginal timing. `evaluate --conditional` is explicitly labeled a human-move diagnostic. Playback conditions on the generated move. |
| E06 | `e06-extended` | Forty absolute bins, retaining the original first thirty boundaries, extending to 999.9 seconds. New tail logits start with low mass. All-label and common-support results remain separate. |
| E07 | `e07-fraction`, `e07-beta` | New output layers; old absolute logits are not relabeled. Both score the same observation intervals in seconds. Beta uses positive parameters derived from raw outputs. |
| E08 | `audit`, `e08-point-buckets`, `e08-interval` | Matched categorical vs precision-interval training. Unsupported labels are counted, not clipped into a final bucket. |
| E09 | `e09-natural`, `e09-recent`, `e09-recency`, `e09-balanced`; `partition` for a new chronological cutoff | Recency on old training data does not add 2025/2026 games. New prospective comparisons need genuinely new input games and a declared cutoff. |
| E10 | `public-check`, `e10-public-prior`, then `e10-personal --initialize ...` | Requires supplied public rows, account exclusions, reserved IDs and pre-evaluation cutoff. Public-prior fitting is blocked without checked provenance. |
| E11 | `e11-counts`, `augment --training-cache ...`, `e11-retrieval` | Exact earlier-game counts or nearest normalized pooled embeddings and prior timing summaries. Validation uses training index only. |
| E12 | `e12-policy`, `augment --engine /path/to/stockfish`, `e12-engine` | Entropy/gap from frozen policy; opponent surprisal is masked if unavailable. Engine features are separate offline inputs, not fabricated scores. |
| E13 | `e13-shared`, `train --final-blocks 1` (or 2) `--checkpoint ... --backbone-lr 1e-6` | Recomputes features online, updates selected final block(s), monitors move NLL, saves matching backbone alongside the timing head. |
| E14 | `events`, then `playback --event ...` | Separate binary timeout-event model. Requires verified completed/timeout turn records, including pre-turn state. Ambiguous endings are excluded; timeout records must not have invented exact duration labels. |
| E15 | `playback` recorded replay / virtual games; `export-backbone`, `export`, `browser-fixtures`; browser harness | Generated moves, clocks and histories feed back into native virtual games. Actual browser parity/latency is a separate measured run. |

The brief's conditional later extensions (fast/Beta mixtures, session latent state, prototypes, per-control models, LoRA and engine distillation) are not silently enabled. The listed arms implement the stated first comparisons; their results determine whether those extensions are warranted.

## Evaluate and compare

```bash
.venv/bin/python -m chess_self.experiments evaluate --cache data/timing-cache/val --output reports/timing-reference.json
.venv/bin/python -m chess_self.experiments evaluate --cache data/timing-cache/val --head models/timing/e03-position/best.pt --output reports/timing-e03.json
.venv/bin/python -m chess_self.experiments compare --left reports/timing-e03.json --right reports/timing-reference.json --output reports/timing-e03-comparison.json
```

The common scorecard reports interval NLL (not incompatible density/bin NLLs), CRPS in seconds, predictive median/mean diagnostics, bias, event calibration, 50/80/90% interval coverage/width, support failures, control/clock/phase/platform groups, and macro-control summaries. Python `scorecard(..., deployment_weights=...)` accepts a predeclared deployment mixture. CRPS uses 256-point quantile quadrature; increase quadrature when tiny differences matter. Stable log likelihood retains extremely small valid probabilities. Zero-support intervals yield a null aggregate NLL and explicit failure count; the finite-only diagnostic is not the overall score.

Comparisons require identical game/ply keys and matching information settings. Bootstrap resamples whole games. Repeated development-set selection does not create independent test evidence. The original test set remains reserved for an explicitly authorized final acceptance run.

For E13 evaluation add `--backbone models/timing/e13/backbone.pt` with the matching side `best.pt`. For exploratory rejected heads pair `last.pt` with `last-backbone.pt`; do not treat these as accepted.

## Calibration provenance

`cache --predictor-training-ids path/to/ids.json` records a JSON list of every game fitted by that predictor. Both calibration and validation caches must be disjoint from these IDs and from one another. Never pass an empty list for an actually fitted checkpoint. The existing heads-01 checkpoint fitted every old training game, so a slice of those same games is **not** a valid calibration set for it. Supply future held-out games or generate out-of-fold predictions from predictors trained strictly before each calibration fold. The runner deliberately refuses unknown provenance.

```bash
.venv/bin/python -m chess_self.experiments calibrate --fit-cache data/calibration-cache --val-cache data/calibration-validation-cache --mode temperature --output models/timing/calibration
```

Calibration currently targets the original absolute timing logits. A calibrator is hash-bound to its backbone and cannot be attached to a changed side head. Changed output families need a newly matched calibrator, not reuse of the original artifact.

## New data, retrieval and public pretraining

`partition --rows NEW.jsonl --cutoff 2026-10-01T00:00:00Z --reserved-audit data/processed/audit.json --output NEW_DATA` partitions complete timestamped sessions, assigning a crossing session wholly to validation. Reserved test IDs cause an error. A previous holdout must be formally retired outside this tool before being included in a new training dataset.

`public-check --rows PUBLIC.jsonl --target-accounts MonisVerse MonisChess --cutoff CUTOFF --reserved-audit data/processed/audit.json --output CHECKED_PUBLIC` validates both player identities, timestamps, duplicates and reserved games. Split public development games chronologically before fitting and keep the checked metadata with the training cache. E10 validates that metadata again. No public corpus is downloaded automatically.

For E11, augment the training cache against itself (strictly earlier **other** games) and validation against the training cache:

```bash
.venv/bin/python -m chess_self.experiments augment --cache data/timing-cache/train --training-cache data/timing-cache/train --output data/timing-cache/train-retrieval
.venv/bin/python -m chess_self.experiments augment --cache data/timing-cache/val --training-cache data/timing-cache/train --output data/timing-cache/val-retrieval
```

Offline engine augmentation requires a supplied engine executable. It records shallow/deeper scores from the same pre-move board with depth and time budgets. An engine-feature head needs those same features at inference; do not deploy with missing engine features as if the comparison were equivalent. Distillation is a later experiment.

## Playback and runtime

```bash
.venv/bin/python -m chess_self.experiments playback --checkpoint models/personalized/heads-01.pt --positions data/timing-v2/val.jsonl --mode sample --output reports/timing-replay.json
.venv/bin/python -m chess_self.experiments playback --checkpoint models/personalized/heads-01.pt --head models/timing/e05-squares/best.pt --base 60 --increment 0 --games 20 --max-plies 300 --output reports/timing-virtual.json
```

Replay strips human current moves/durations before prediction. Virtual games use model-generated history and have an explicit first-move charging convention. The default virtual clock charges target consumption; visible delay includes measured native inference latency. This differs from an actual browser's charged clock. Reports distinguish guards, requested/achieved delay, flags and trajectories. The browser scheduler handles turn tokens, cancellations, game termination and stale results without making live account moves. See `timing-runtime.md`.

## Export and browser checks

```bash
.venv/bin/python -m chess_self.experiments export-backbone --checkpoint models/personalized/heads-01.pt --output models/export/personalized-timing-features.onnx
.venv/bin/python -m chess_self.experiments export --head models/timing/e05-squares/best.pt --output models/export/personalized-timing-side.onnx
.venv/bin/python -m chess_self.experiments browser-fixtures --checkpoint models/personalized/heads-01.pt --head models/timing/e05-squares/best.pt --positions data/timing-v2/val.jsonl --output reports/timing-browser-fixtures.json
```

The feature and side exports both verify CPU ONNX Runtime parity. The browser harness loads those local files and fixtures and measures WASM/WebGPU separately. It does not upload personal data. Native ONNX success is not browser success. See `timing-model-contract.md` and `timing-runtime.md`.

## Transfer and verification

```bash
python3 scripts/bundle_timing.py --output bundles/chess-self-timing-gpu.tar.gz
# After extracting on the rented machine, before setup:
python3 scripts/bundle_timing.py --verify /path/to/chess-self
bash scripts/setup.sh
.venv/bin/python -m pytest -q
node --test browser/timing-runtime.test.mjs
```

The timing bundle adds configurations, browser tools, schema-v2 data and the personalized reference to the existing portable source bundle. Cached features are regenerated on the destination. Existing all-inclusive archives are older snapshots and are not silently replaced.

## Implementation verification

Development smoke artifacts live in `reports/timing-smoke/`. These use two training games and two validation games to exercise paths, not to estimate improvement. The frozen-feature arms and one-final-block adaptation were run on CPU. Side-head and backbone exports have native CPU ONNX parity reports; browser fixtures were checked against both exported sessions. Actual browser backend checks are performed by the local harness, not inferred from native results.

To inspect a schema-dependent head in replay, pass rows containing the matching familiarity/retrieval/engine features. Prediction rejects missing required feature groups. Native virtual games support the basic clock/history/move heads directly; richer heads require a causal feature provider instead of substituting zeros. Policy entropy, gap and reconstructable previous-opponent surprisal are computed from the model; unknown prior clocks produce an explicit missing-surprisal mask.

For a new split that would reuse old validation games, `partition` additionally requires `--retire-validation` and records the retired IDs. Original reserved test IDs remain blocked.
