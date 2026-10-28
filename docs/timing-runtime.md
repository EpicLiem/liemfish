# Timing playback and execution (E01 / E15)

`src/chess_self/timing_runtime.py` and `browser/timing-runtime.mjs` supply clock bookkeeping and execution instrumentation. They do not provide a browser chess UI or claim WASM/WebGPU model parity. Node tests exercise the browser-compatible state machine with a fake monotonic clock; actual browser inference latency remains a separate deployment measurement.

## Prediction adapter

Both Python experiment functions accept `predictor(board, row)`. The board is a copy. The row contains pre-move information; recorded replay removes current human moves, timing labels, and ending metadata before invoking the predictor.

Return:

```python
{
    "move_uci": "e2e4",           # selected legal model move, not the human target
    "target_seconds": 1.2,        # declared mean, median or sampled duration
    "inference_seconds": 0.03,    # optional simulated execution floor; default zero
    "removed_probability_mass": 0.02,  # optional sampler diagnostic
    "timeout": False,            # optional distinct unfinished-turn event from E14
}
```

The experiment adapter owns move selection, random seeds, output-family semantics, and probability truncation. For move-conditioned timing, select the move first and ask for that move's conditional time distribution. Apply interval-aware truncation in the distribution module before sampling. The runtime's final scalar deadline guard is a last-resort execution safeguard, **not** a substitute for handling partly feasible probability bins. Its intervention count must be reported. Setting `guard=False` deliberately retains infeasible samples for the original-sampling control arm.

A completed move has a duration strictly below the current clock. The upcoming increment is never part of its budget. Reserve seconds represent an explicit execution allowance; there is no percentage-of-clock strategic cap. A clock already exhausted or a reserve consuming the entire budget terminates before model inference. A distinct predicted timeout event is honored without moving or applying increment.

## Recorded-state replay

```python
from chess_self.timing_runtime import recorded_replay
report = recorded_replay(validation_rows, predictor, reserve_seconds=0, guard=True)
```

Every row keeps the recorded human board and pre-move clock. The rows are independent conditional predictions, not a generated game. Results include original observed duration only in output diagnostics, never as predictor input. `charged_seconds` is simulated recorded-clock consumption. `visible_delay_seconds` is the larger of sampled/selected duration and supplied inference latency. Replay does not sleep, send moves, or claim observed platform accounting.

## Virtual full games

```python
from chess_self.timing_runtime import virtual_game
report = virtual_game(
    predictor, initial_seconds=60, increment_seconds=0,
    ratings=(1500, 1500), max_plies=400,
    first_move_rule="charge_all", clock_charge_mode="target",
)
```

A predictor can also be a `{chess.WHITE: white_predictor, chess.BLACK: black_predictor}` mapping. Every next turn receives the generated board, up to eight generated board states, both generated clocks, and the last eight realized durations for each player. History is chronological, oldest first. Terminal boards, flags, exhausted execution budget, and the move limit stop the game. Move legality is checked before a move is applied.

Two explicit clock modes keep the modeled target distinct from display latency:

- `target`: subtract the target recorded-clock consumption; visible delay still includes the supplied inference floor.
- `wall`: subtract the larger of target duration and supplied inference latency; this estimates an execution-limited local clock.

Neither mode measures actual inference automatically. Supply measured latency separately to study runtime constraints. `charge_all` charges every turn, including each player's first. The alternate `free_first_turn` charges zero on each player's first completed turn, then charges normally. Increment is added after every completed move in both modes. Choose a platform-appropriate convention explicitly; the runtime does not infer one from the account name.

Reports contain every clock transition, requested and effective duration, modeled visible delay, inference floor, selected move, guard intervention, and distinct timeout outcome. Summary fields include first entry below ten seconds, flags, latency-floor misses, and lag-one correlation of log-duration on consecutive same-player turns. `board_result` describes the board alone: for a simulated flag it can remain `*`; use `termination` and the flagged event for the clock outcome. A move-limit stop is not a completed chess game. Repeated seeds and control-matched games are required for behavioral conclusions.

## Nonblocking scheduler

`TurnScheduler` records `turn_start`, `inference_start`, `inference_end`, and `commit_time`. Call `begin(fen, remaining_seconds)`, `inference_started(token)`, then `prediction_ready(token, move_uci, target_seconds)`. The plan returns:

```
wait_seconds = max(effective_target_seconds - (now - turn_start), 0)
```

Thus model computation is counted toward the delay. The host waits and calls `commit(token, current_fen, increment_seconds=...)`; early commits, changed positions, and stale tokens are rejected. `takeback()`, `cancel(reason)`, and `end_game()` invalidate in-flight predictions. A safe sample can still miss its deadline because inference or a timer ran late; that case never earns increment or logs a completed move.

The native scheduler accepts an optional authoritative `charged_seconds` at commit, because platform lag accounting may differ from wall time. Its default is elapsed wall time. Clock outputs are estimates unless backed by platform receipts. The Python state machine records execution; it does not itself operate a UI.

## Browser adapter

Import `TurnScheduler` and `executeTurn` from `browser/timing-runtime.mjs`. Use seconds throughout. Supply host callbacks:

- `infer()` resolves `{move_uci, target_seconds}`.
- `wait(seconds)` awaits a timer. A takeback or ended game can cancel it proactively; token checks also prevent a move when it returns.
- `currentPosition()` returns a stable position key matching `begin`.
- `validateMove(uci)` checks legality against the live board.
- `commitMove(uci)` synchronously and atomically applies the local move; the callback must not return a promise. Network submission/acknowledgment requires separate host instrumentation.

`executeTurn` validates the token after inference and after waiting, rechecks early timers, validates position and legality immediately before the commit, and records a failed host commit as `commit_failed`. The host owns the actual board and may persist `scheduler.events` as JSON. An async network send cannot be made atomic with a token check by this local module; do not report its estimated commit time as a server receipt.

## Checks

From the repository root:

```bash
.venv/bin/python -m pytest -q tests/test_timing_runtime.py
node --test browser/timing-runtime.test.mjs
```

The checks use fake clocks; no real games, actual waiting, network requests, or holdout evaluation are required. They cover stale inference, takebacks, terminal cancellation, changed positions, illegal moves, early and late timers, zero budgets, increment accounting, generated-history feedback, and label isolation. ONNX parity and real-browser latency are not covered by these tests.

## E15 browser ONNX parity and latency harness

`browser/index.html` runs the exported feature backbone followed by the timing side model. Choose both local `.onnx` files and the matching native fixture JSON. No file is uploaded: the browser reads it with `File.arrayBuffer`, and all inference runs locally. ONNX Runtime Web **1.31.1** and its WASM files are installed locally with a lockfile; no CDN is used. The page's content-security policy restricts connections to its own origin.

Install and serve the harness yourself when ready for a real browser measurement:

```bash
cd browser
npm ci --ignore-scripts
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/`. Select **WebAssembly (CPU)** or **WebGPU** explicitly, load the matching exports, then run the checks and save the JSON report. Stop the local server after use. Localhost supplies a secure context for WebGPU; the chosen browser/device must also support it. The implementation deliberately uses one WASM thread, so cross-origin isolation is not required. Reload the page between provider comparisons to reset runtime state.

Each fixture case reports batch size, all-output parity, maximum absolute/relative error, nonfinite values, first-call latency, and repeated warm-call latency. Timing covers the feature model, side model, and output readback; array comparison is excluded. The first call per case is labeled `cold` in the JSON, but only the first case is session-cold: later cases share loaded sessions and compiled kernels. Session load time is recorded separately. Model/fixture SHA-256 hashes make the report attributable to exact inputs. An explicit provider request does not certify placement of every operator, and the report says so.

The accepted fixture format is:

```json
{
  "schema_version": 1,
  "tolerances": { "atol": 0.0001, "rtol": 0.001 },
  "config": { "architecture_id": "native-export-identifier" },
  "cases": [{
    "name": "opening-batch1",
    "inputs": {
      "tokens": { "data": [], "shape": [1, 64, 99] },
      "self_elos": { "data": [], "shape": [1] },
      "oppo_elos": { "data": [], "shape": [1] }
    },
    "side_inputs": {
      "side_features": { "data": [], "shape": [1, 1] },
      "move_indices": { "data": [], "shape": [1, 2], "dtype": "int64" },
      "move_attributes": { "data": [], "shape": [1, 1] },
      "history": { "data": [], "shape": [1, 1] }
    },
    "expected": {
      "move_logits": { "data": [], "shape": [1, 4352] },
      "base_logits": { "data": [], "shape": [1, 30] },
      "pooled": { "data": [], "shape": [1, 512] },
      "squares": { "data": [], "shape": [1, 64, 512] },
      "timing_raw": { "data": [], "shape": [1, 30] }
    }
  }]
}
```

This illustrates field names, not an executable fixture: the native exporter fills every flat `data` array and the architecture's actual dimensions. Batch dimensions are positive; trailing dimensions may be zero for disabled feature groups. Arrays must contain exactly the declared number of finite values. `move_indices` uses safe integer JSON numbers, converted to `BigInt64Array`. Other tensors default to float32. All fields must share the case's batch size. The harness feeds only names retained in each ONNX session's input interface, so optimized-away side inputs cause no error. The expected timing output is raw model parameters/logits, not samples or the scheduler's elapsed duration.

The optional top-level `scheduler` object enables a measured local scheduler check:

```json
{
  "case_name": "opening-batch1",
  "position_key": "native position FEN or another stable key",
  "move_uci": "e2e4",
  "legal_moves": ["e2e4", "d2d4"],
  "remaining_seconds": 60,
  "target_seconds": 0.2,
  "reserve_seconds": 0
}
```

The exporter must supply the true complete legal move list and selected legal move. The harness times an actual two-model inference, then counts that elapsed time toward the target delay and records a dummy commit. It does not decode a model move or send one to a chess site. Targets are limited to 0–10 seconds for this diagnostic so a bad fixture cannot hold the test indefinitely. Scheduler inference timing includes parity comparison overhead and is labeled accordingly. The low-level scheduler tests independently exercise takebacks, stale predictions, and ended games; the optional measured check exercises the local timer/inference path. A report can pass tensor parity while its scheduler reports a missed budget: inspect both.

Additional local verification commands:

```bash
cd browser
npm test
node --check parity-harness.mjs
```

These tests validate the fixture contract, tensor comparison, filtered ONNX inputs, pipeline wiring, resource disposal, latency summaries, and scheduler behavior. They do not establish browser model compatibility. No server or browser execution is started by installation or these tests.

Implementation references: [ONNX Runtime Web execution](https://onnxruntime.ai/docs/tutorials/web/), [runtime flags and WASM asset paths](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html), and [WebGPU execution provider](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html).
