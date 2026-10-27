# Maia3 ONNX browser contract

`export_model(checkpoint: Path, output: Path) -> dict` loads a finished checkpoint on CPU and writes a single FP32 ONNX file (opset 18), `<stem>.moves.json`, and `<stem>.json`. `output` is a filename, not a directory. The report records SHA-256 hashes, dependency versions, tensor shapes, bucket centers, and CPU ONNX Runtime comparisons. A failed comparison writes the report and raises; never deploy an artifact whose report says `parity.passed: false`. Install the project's `dev` extra for ONNX and ONNX Runtime. Finish checkpoint downloads before calling; the exporter does not download or poll files.

The implementation uses the pinned upstream source commit recorded in the report. Export disables PyTorch MHA fastpath temporarily and decomposes RMSNorm into FP32 square, mean, epsilon, reciprocal square root, and multiplication. Verification uses the original native RMSNorm, not the export decomposition. No quantization or external weight files are used.

## Inputs

The Python CLI accepts a position JSON containing `fen`, `history_fens` (oldest to current, one to eight FEN strings), `elo_self`, `elo_oppo`, `initial_seconds`, `increment_seconds`, and `remaining_seconds`. See `example-position.json` for a generic example. These fields are encoded into the tensors below. `move_uci` and `elapsed_seconds` are optional training labels and are unnecessary for prediction.

All three inputs are contiguous float32 tensors with the same positive batch size B:

| Name | Shape | Meaning |
| --- | --- | --- |
| `tokens` | `[B,64,99]` | 96 history channels and 3 clock channels |
| `self_elos` | `[B]` | Raw rating of the current player |
| `oppo_elos` | `[B]` | Raw rating of the opponent |

Ratings are clamped to [0,5000] inside the graph and interpolated exactly as upstream. Do not normalize them in the client.

Squares are rank-major: a1=0, b1=1, …, h8=63. Supply the last one to eight actual pre-move board positions in chronological order, ending with the current position. For short histories, repeat the earliest available board on the left until there are eight. Concatenate their twelve channels, oldest first, current last. Do not replace available history with repeated copies of the current board.

Each historical board is oriented independently by **that board's own side to move**. If black is to move in that historical board, mirror ranks (a1 ↔ a8) and swap piece colors, as `python-chess Board.mirror()` does. Within each twelve-channel block, channels are white pawn, knight, bishop, rook, queen, king, then black pawn, knight, bishop, rook, queen, king after orientation. Each occupied square has one piece channel set to 1; empty squares have twelve zeros. Castling rights, en-passant, and counters are not extra input planes; the client still needs them to compute legal moves.

Channels 96, 97, 98 repeat across all 64 squares: initial clock seconds / 100, increment seconds / 100, and current player's remaining seconds **before the move** / 100. Elapsed move time is a training target and must NEVER appear in this input. No padding mask or legal-move mask is supplied to the graph.

## Outputs and move indexing

| Name | Shape | Meaning |
| --- | --- | --- |
| `move_logits` | `[B,4352]` | Raw unmasked move scores |
| `timing_logits` | `[B,30]` | Raw position-level timing scores |
| `timing_probabilities` | `[B,30]` | Softmax of timing logits |
| `expected_seconds` | `[B]` | Probability-weighted timing bucket centers |

The adjacent `.moves.json` is the exact ordered UCI vocabulary. Indices 0–4095 are `64 * from_square + to_square`, including impossible moves and same-square entries. Indices 4096–4351 are promotions: `4096 + 32 * from_file + 4 * to_file + piece_index`, with files a=0 through h=7 and pieces q=0, r=1, b=2, n=3. Promotion squares are always rank 7 to rank 8 in the oriented vocabulary.

For the current black-to-move position, mirror ranks of each vocabulary UCI move back to the original board, preserving the promotion suffix. Compute legality using the complete original board state. Before softmax or sampling, mask illegal move logits to negative infinity, optionally divide the remaining logits by a strictly positive temperature, and normalize only legal moves. A terminal position has no prediction to sample; handle it outside the model. Never softmax all 4352 entries and then merely discard illegal moves without renormalizing.

Timing bin i has center `10**(-1 + (i + 0.5)*0.1) - 0.1` seconds, i=0…29. Boundaries are `10**(-1 + i*0.1) - 0.1`; training coverage is [0,99.9) seconds. The report includes the actual FP32 centers. The expectation is the sum of each probability times its center. This is a **position-level** timing distribution conditioned on history, clocks, and ratings. It is not a duration conditioned on a selected candidate move, a guaranteed response delay, or an engine search budget. Do not change it when a different move is sampled or imply that it models arbitrarily long thinks.

## Future browser worker integration

A future Web Worker can load the ONNX asset with ONNX Runtime Web, build named float32 tensors, run the session, and return the four outputs with a request/position identifier. Keep session loading and inference off the main thread, discard stale results after a position changes, and surface load/inference errors. Cache assets by the recorded ONNX hash. Transfer typed-array buffers when useful; retain the actual history and full legal board state in the caller. Prefer WASM as an initial compatibility target and verify any chosen WebGPU execution path separately.

This implementation provides no browser UI or worker. CPU ONNX Runtime parity does not establish browser WASM/WebGPU compatibility or performance. The saved verification suite covers white and black, promotions for both colors, short and full actual histories, low clocks, and batches of 1 and 2, comparing all four outputs with `atol=2e-4, rtol=2e-4`. It records maximum and mean absolute errors for every case/output.
