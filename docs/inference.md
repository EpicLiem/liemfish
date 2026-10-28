# Inference

Liemfish predicts a legal move and a target turn duration. For JavaScript and TypeScript applications, use the [browser package](../packages/browser/README.md). For native Python applications, use `TimingPredictor`.

## Selected artifacts

Paths are relative to the repository root.

| Artifact | Bytes | Role |
| --- | ---: | --- |
| models/export/recent-policy-features.onnx | 91,339,625 | Recent-game epoch-6 move policy and frozen backbone representations |
| models/export/e05-recent-policy.onnx | 418,125 | E05 epoch-5 timing correction, bound to the recent policy |
| models/personalized/recent-policy-epoch6.pt | 91,864,375 | Native personalized backbone checkpoint |
| models/personalized/e05-recent-policy.pt | 465,309 | Native selected timing head |

Browser models total 91,757,750 bytes (91.76 MB), before transfer compression. Use the backbone and timing head together.

Adjacent export JSON files contain hashes, exact inputs and feature order. Backbone ONNX SHA256: `bc97f3882bf51be3599eca78f178a9066bfc0c19d196321c3bd8c2bc779a168a`. Side ONNX SHA256: `85b5dc3ab33dfde62ca511ea0c5a4f9e8ca1ece0846fa2bc2a1aaadca56a1921`. Underlying backbone checkpoint SHA256: `52f9339db166dc07e61fc8c01448ba0b0a83befbf435fd864ac6a510f5422427`.

The move policy adapts on 99 games from 2024–2025, with epoch selection on 39 games from 2026. The original test games remain reserved. Timing weights and backbone representations are unchanged, but the selected move’s policy surprisal can change timing predictions. [Validation and gameplay results](../reports/recent-policy-01/RESULTS.md).

## Native Python API

`TimingPredictor(checkpoint, head, mode='sample', feasible=True, reserve_seconds=0., device='cpu', seed=42)` loads once. Call `predictor(chess.Board(row['fen']), row)` sequentially: extractor hooks and RNG are stateful; use a queue or one instance per game, not concurrent forward calls.

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "history_fens": ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"],
  "elo_self": 680,
  "elo_oppo": 680,
  "initial_seconds": 600,
  "increment_seconds": 0,
  "remaining_seconds": 600,
  "opponent_remaining_seconds": 600,
  "own_time_history": [],
  "opponent_time_history": [],
  "previous_move_was_capture": false
}
```

FEN is complete standard-chess state before the AI move. history_fens contains the last 1–8 actual positions after successive plies, oldest to current, ending exactly at fen; do not include future positions or manually pad the Python input. Ratings are raw, side-to-move-relative (self=AI when AI moves). Clocks are seconds before the move, excluding the upcoming increment. Own/opponent histories each contain up to eight completed durations in chronological order, relative to the current player; null means unknown, and must not be replaced with a guessed duration. The immediately preceding move's capture flag comes from the board before that move. Opponent clock may be null if truly unknown; base/increment/current clock and ratings must be finite, nonnegative values. Check terminal positions in the host before inference; the encoder rejects terminal boards.

Example return:

```json
{"move_uci":"g1f3","target_seconds":2.4,"removed_probability_mass":0.01,"inference_seconds":0.053}
```

Each prediction samples a legal move and a duration conditioned on that move. `target_seconds` is total target turn duration, not additional sleep. Normal responses omit timeout. Exhausted budget returns `{"timeout":true,"target_seconds":0.0}` without a move. Exceptions are Python errors. Applications should discard stale replies after restart, takeback, or position changes.


```python
from pathlib import Path
import chess, torch
from chess_self.timing_predict import TimingPredictor
torch.set_num_threads(1)
predictor = TimingPredictor(
    Path('models/personalized/recent-policy-epoch6.pt'),
    Path('models/personalized/e05-recent-policy.pt'),
    device='cpu', mode='sample', feasible=True, seed=42)
# row follows the position schema above:
result = predictor(chess.Board(row['fen']), row)
```

## Model interfaces

The [backbone export contract](browser-contract.md) describes board encoding and move indexing. The [timing model contract](timing-model-contract.md) describes timing features, distributions, and the side-head interface. Export metadata beside each ONNX file records tensor names, shapes, and hashes.
