# Liemfish portfolio integration handoff

Project root: `/Users/liem/random/chess-self`. Git origin is `https://github.com/EpicLiem/liemfish.git`; remote configuration does not mean these artifacts are published. Portfolio files were not modified.

## Selected artifacts

All paths below are relative to the project root.

| Artifact | Bytes | Role |
| --- | ---: | --- |
| models/export/personalized-timing-features.onnx | 91,339,625 | Personalized heads-01 move policy and backbone representations |
| models/export/e05-selected-epoch5.onnx | 418,125 | Selected full-data E05 epoch-5 timing correction |
| models/personalized/heads-01.pt | 108,643,175 | Native personalized backbone checkpoint |
| reports/collected/chess-self-timing-retry/models-runs/timing-01/e05-squares/best.pt | 464,797 | Native selected timing head |

Browser models total 91,757,750 bytes (91.76 MB), before transfer compression. Side head alone cannot choose moves. Do not use `maia3-23m-ponder.onnx`, `maia3-23m-timing-features.onnx`, `timing-smoke-side.onnx` or last.pt as the selected personalized pair.

Adjacent export JSON files contain hashes, exact inputs and feature order. Backbone ONNX SHA256: `8109a8d96e19e2d36de56018b8429b41a99c7c1adae6b8a4a2bdee1454cd6e7a`. Side ONNX SHA256: `85b5dc3ab33dfde62ca511ea0c5a4f9e8ca1ece0846fa2bc2a1aaadca56a1921`. Underlying backbone checkpoint SHA256: `8758ea33f911233b28c2c134192a3f127b5c7669e22c08178417e41393c40427`.

## Runtime status and license

ONNX Runtime Web 1.31.1 is installed/locked in browser/. Both models passed actual headless Chrome 156 WASM and WebGPU parity: four white/black/low-clock/batch-two cases, all five output tensors, three warm repetitions. Single-position warm medians: WASM 57.8–59.9 ms, WebGPU 23.0–23.6 ms. Reports: reports/timing-browser-selected/{wasm,webgpu,summary}.json. Local dummy scheduler timing passed. This is not a finished browser chess engine: the JS arbitrary-position encoder, legal policy sampling, side-feature construction and continuous timing decoder still need integration. browser/parity-core.mjs consumes already encoded fixtures; it is not a FEN API. No HTTP server/API exists. The working end-to-end native API is TimingPredictor below.

Repository and vendored Maia implementation contain AGPL-3.0 license text (LICENSE and sources/maia3/LICENSE). The base timing weights have no supplied model-card license; the upstream page still says “No model card”: https://huggingface.co/UofTCSSLab/Maia3-23M-ponder (checked October 27, 2026). Do not infer a separate permissive weight-redistribution license from the code license. Public asset publication needs that distinction resolved; this handoff does not publish anything.

## Existing native request/response contract

`TimingPredictor(checkpoint, head, mode='sample', feasible=True, reserve_seconds=0., device='cpu', seed=42)` loads once. Call `predictor(chess.Board(row['fen']), row)` sequentially: extractor hooks and RNG are stateful; use a queue or one instance per game, not concurrent forward calls.

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "history_fens": ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"],
  "elo_self": 491,
  "elo_oppo": 529,
  "initial_seconds": 600,
  "increment_seconds": 0,
  "remaining_seconds": 600,
  "opponent_remaining_seconds": 600,
  "own_time_history": [],
  "opponent_time_history": [],
  "previous_move_was_capture": false
}
```

FEN is complete standard-chess state before the AI move. history_fens contains the last 1–8 actual positions after successive plies, oldest to current, ending exactly at fen; do not include future positions or manually pad the Python input. Ratings are raw, side-to-move-relative (self=AI when AI moves). Clocks are seconds before the move, excluding the upcoming increment. Own/opponent histories each contain up to eight completed durations in chronological order, relative to the current player; null means unknown, and must not be replaced with a guessed duration. The immediately preceding move's capture flag comes from the board before that move. No current elapsed_seconds or target human move is needed. No game_id, timestamps, platform or outcome is required for this E05 head. Opponent clock may be null if truly unknown; base/increment/current clock and ratings must be finite, nonnegative values. Check terminal positions in the host before inference; the encoder rejects terminal boards.

Normal return:
```json
{"move_uci":"g1f3","target_seconds":2.4,"removed_probability_mass":0.01,"inference_seconds":0.053}
```

This is illustrative, not a fixed response. RNG samples a legally masked move and timing conditioned on that selected move. `target_seconds` is total target turn duration, not additional sleep. Normal responses omit timeout. Exhausted budget returns `{"timeout":true,"target_seconds":0.0}` without a move. No learned timeout-event model is attached. Exceptions are local Python errors; there is no existing HTTP status/error schema. Add requestId/positionKey and error envelopes in the host adapter, and discard stale replies after restart/takeback/position changes.

## Browser tensor contract and exact sources to port

Preferred deployment shape: a client-only Web Worker loads the pair lazily on Play; keep sessions resident and use WebGPU with explicit WASM fallback. Never import the runtime into Next server components. Preserve artifact hashes and measure fallback on the target browser. No service is required once the adapter is implemented.

1. Backbone inputs: float32 tokens [1,64,99], self_elos [1], oppo_elos [1]. Raw ratings are clamped/interpolated inside the graph. Encoder: src/chess_self/model.py:encode and sources/maia3/maia3/dataset.py:tokenize_board/get_historical_tokens. Repeat earliest board on the left to eight internally; orient EACH historical board by its own side to move (black: flip ranks and swap colors). Square index a1=0. Twelve planes per board: P,N,B,R,Q,K,p,n,b,r,q,k after orientation. Last three channels repeat base/100, increment/100, own pre-clock/100 across all squares. Castling/EP stay in host FEN for legality, not token planes.
2. Outputs: move_logits [1,4352], base_logits [1,30], pooled [1,512], squares [1,64,512]. Mask illegal moves to -Infinity and softmax at temperature 1. Sample, do not force argmax for imitation. Vocabulary: ordinary index=64*from+to; promotions=4096+32*from_file+4*to_file+piece(q,r,b,n) in oriented rank7→8. Mirror ranks back for black; preserve promotion suffix. Use docs/browser-contract.md for exact indexing, but its old single-export timing outputs do not describe this two-model pair.
3. E05 inputs: pooled/base_logits/squares from backbone; side_features float32 [1,50], move_indices int64 [1,2], move_attributes float32 [1,8]. This exported head prunes the unused history tensor. Feature order is exact in models/export/e05-selected-epoch5.json. Implement src/chess_self/timing_data.py:feature_vector with groups clock,phase,chess,history_window. Its first18 values are clocks/phase/chess; last32 interleave log1p(duration),known for each of eight own then eight opponent turns, padded left with null/mask0. Do not supply log-transformed ratings or replace missing history with observed zeros.
4. src/chess_self/timing_model.py:selected_move_inputs defines current-player-oriented source/destination and eight attributes: queen/rook/bishop/knight promotion flags, capture, gives-check, castling, negative log probability under the normalized LEGAL policy. Attributes describe the sampled move, not a supplied human target.
5. Output timing_raw [1,30] is the final absolute-time logits, already including the backbone residual. Do not add base_logits again. Bin edges e_i=10^(-1+i/10)-0.1, i=0..30; distribution is uniform in log(t+0.1) within each bin. Truncate each bin at budget=min(clock-reserve, nextDown(clock)), retaining fractional mass log((upper+.1)/(lower+.1))/log((e_i+1+.1)/(e_i+.1)), then renormalize. Sample a bin and interpolate uniformly in log-space within its retained interval; never sample bucket centers for faithful timing. Reject zero budget instead of renormalizing empty support. Reference implementation src/chess_self/timing_distributions.py.
6. browser/timing-runtime.mjs TurnScheduler/executeTurn subtracts all elapsed inference from target waiting, rechecks position/turn token/legality, commits once, then adds increment. Advance both board history and the completed player's timing history using actual observed durations. No universal arbitrary delay cap: optional accelerated playback must be labeled as such. Millisecond browser timers require conversion from seconds. The parity harness uses a fixture move and target for its dummy commit; it does not implement steps1–5 from arbitrary FEN.

## Commands and verified checks

```bash
cd /Users/liem/random/chess-self
# Restore dependencies only if needed (does not train/rent):
uv sync --frozen --extra dev --python 3.12
.venv/bin/python -m pytest -q
node --test browser/*.test.mjs
node scripts/check_selected_browser.mjs
# Interactive parity harness, local files selected manually:
cd browser
npm ci --ignore-scripts
python3 -m http.server 8765 --bind 127.0.0.1
```

Interactive page http://127.0.0.1:8765 uses the selected pair above and reports/e05-selected-browser-fixtures.json. It is only a parity page. The automated script defaults to installed Chrome and a local Playwright path; CHROME_EXECUTABLE/PLAYWRIGHT_MODULE override them. Stop the server after use.

Native adapter startup:
```python
from pathlib import Path
import chess, torch
from chess_self.timing_predict import TimingPredictor
torch.set_num_threads(1)
predictor = TimingPredictor(
    Path('models/personalized/heads-01.pt'),
    Path('reports/collected/chess-self-timing-retry/models-runs/timing-01/e05-squares/best.pt'),
    device='cpu', mode='sample', feasible=True, seed=42)
# row is the JSON contract above:
result = predictor(chess.Board(row['fen']), row)
```

Prior full verification: 190 Python tests and 18 JS unit tests passed; both real browser providers passed all four cases. Current handoff smoke called the selected native predictor on the starting position and returned legal g1f3 in ~52 ms. A 1200/1200 test context was used only for that smoke. No existing REST endpoint/start command should be invented.

## Playing settings and honest copy

Suggested initial demo: standard chess, 10+0, human chooses color; sampled move policy temperature1, sampled feasible E05 timing, no engine search/strength boost. 10+0 is the most represented training control (7,185 positions; 5+0 has5,265). Default rating context can use training medians self491/opponent529 as a documented historical context, not a claimed current or playing Elo; allow an opponent context choice if useful. Production RNG should vary across games; keep fixed seeds for reproducible tests. Avoid bullet as the initial showcase because low-clock gains remain uncertain and runtime floors matter. Disable clock progression until model loading completes.

Suggested UI: “Play Liemfish — a chess model personalized on Liem’s games, with a learned move-timing model. It imitates patterns in those games; it isn’t a perfect copy or a strength-rated engine.” After browser integration is verified: “Runs locally in your browser after a ~92 MB model download.” Do not claim current behavior (training ends November3,2024), human indistinguishability, exact Elo, flawless clock-pressure mimicry or complete gameplay verification. No raw PGNs, training rows, reports with personal positions, or .pt files need shipping to visitors. Ship only the required browser assets/notices once publication terms are settled.
