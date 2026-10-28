# @epicliem/liemfish-browser

Framework-independent Liemfish move and timing inference. This package contains the client API, prebuilt module worker, pinned ONNX Runtime WASM files, trained time-control metadata, and an asset preparation CLI.

```ts
import { createModel, DEFAULT_TIME_CONTROL, ratingContextForControl } from '@epicliem/liemfish-browser';
import type { ModelPosition, ModelProgress, ModelPrediction } from '@epicliem/liemfish-browser';

const model = createModel({ assetBaseUrl: '/chess/engine/' });
await model.loadModel(progress => console.log(progress));
const ratings = ratingContextForControl(position.initialSeconds, position.incrementSeconds);
const result = await model.predict({ ...position, ...ratings }, { signal: abortController.signal });
// result: { uci: string, targetSeconds: number }
model.disposeModel();
```

`createModel` performs no browser work until `loadModel`. Each instance owns one worker and queue. Load calls share a promise; the first caller receives progress. Disposing terminates the worker and rejects pending requests with `AbortError`; a later load creates a new session. Abort rejects an individual prediction and ignores its eventual reply; it does not interrupt an already-running ONNX operation. Inference uses single-threaded WASM.

`assetBaseUrl` defaults to `/chess/engine/`. Relative values resolve against the page URL, and a trailing slash is optional. Host the worker and assets on the same origin. The asset CLI supplies the compiled worker.

`ModelPosition` accepts: `fen`, `historyFens`, `remainingSeconds`, `initialSeconds`, `incrementSeconds`, `eloSelf`, `eloOpponent`, `opponentRemainingSeconds`, `ownTimeHistory`, `opponentTimeHistory`, and `previousMoveWasCapture`. Clocks are seconds before the move; history ends at the current FEN. Unknown opponent clocks or completed durations are null. The host checks terminal positions, owns chess-game state, and rejects stale results. `targetSeconds` is total desired turn duration, including inference time, not additional sleep.

The package exports `TRAINED_TIME_CONTROLS` and `DEFAULT_TIME_CONTROL` (10+0) as metadata.

`ratingContextForControl(base, increment)` supplies optional historical player context: 230 for bullet, 220 for blitz, and 680 for rapid or longer games. Categories use `base + 40 × increment` seconds, with boundaries at 180 and 600 seconds. Both returned ratings use the same player rating; pass a known opponent rating instead when available. The model never overrides ratings supplied in `ModelPosition`.

`RECENT_PLAYER_RATINGS` records the underlying 2026 Chess.com medians and game counts: bullet 232 (1 game), blitz 224 (15), rapid 675 (23). Rounded inputs are historical context, not measured bot Elo; the single bullet game is particularly sparse.

## Consumer assets

Install the published version with `npm install --save-exact @epicliem/liemfish-browser@0.2.0`, then run:

```sh
npx liemfish-assets --source /path/to/chess-self --out public/chess/engine
npx liemfish-assets --verify --out public/chess/engine
```

Preparation verifies the recent-game epoch-6 move policy and the rebound E05 epoch-5 timing head before writing. The policy was adapted on 99 games from 2024–2025 and selected on 39 development games from 2026. E05 timing weights remain unchanged; the head is bound to the new policy after conditional timing validation. Model reports, move vocabulary, license notices, worker and WASM files are staged beside a deterministic `manifest.json`. Verification needs only the installed package and staged directory. It rejects missing files, changed hashes, incorrect model identity and a worker/runtime that differs from the installed package.

## Development

From `packages/browser`:

```sh
npm ci
npm run build
npm test
npm run test:package
npm pack
```

The client entry supports both ESM import and CommonJS require without browser globals; the worker remains ESM. Build bundles chess.js and ONNX Runtime into the worker and emits public declarations. The package includes runtime files and dependency notices. Build/test tools are pinned in package-lock.json. `npm pack` supports local release verification before publishing.

The source license is AGPL-3.0; chess.js and ONNX Runtime notices retain their respective terms. Model weight licensing is separate, as documented in the root repository.
