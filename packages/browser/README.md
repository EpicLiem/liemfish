# @epicliem/liemfish-browser

Framework-independent Liemfish move and timing inference. This package contains the client API, prebuilt module worker, pinned ONNX Runtime WASM files, trained time-control metadata, and an asset preparation CLI. It contains no weights, training data, React components, board UI, clock scheduler, sounds, or premove logic.

```ts
import { createModel, DEFAULT_TIME_CONTROL } from '@epicliem/liemfish-browser';
import type { ModelPosition, ModelProgress, ModelPrediction } from '@epicliem/liemfish-browser';

const model = createModel({ assetBaseUrl: '/chess/engine/' });
await model.loadModel(progress => console.log(progress));
const result = await model.predict(position, { signal: abortController.signal });
// result: { uci: string, targetSeconds: number }
model.disposeModel();
```

`createModel` performs no browser work until `loadModel`. Each instance owns one worker and queue. Load calls share a promise; the first caller receives progress. Disposing terminates the worker and rejects pending requests with `AbortError`; a later load creates a new session. Abort rejects an individual prediction and ignores its eventual reply; it does not interrupt an already-running ONNX operation. All inference uses single-threaded WASM, preserving the portfolio implementation's baseline.

`assetBaseUrl` defaults to `/chess/engine/`. Relative values resolve against the page URL, and a trailing slash is optional. Host the worker and assets on the same origin. No consumer bundler must compile or discover the worker.

`ModelPosition` preserves the portfolio contract: `fen`, `historyFens`, `remainingSeconds`, `initialSeconds`, `incrementSeconds`, `eloSelf`, `eloOpponent`, `opponentRemainingSeconds`, `ownTimeHistory`, `opponentTimeHistory`, and `previousMoveWasCapture`. Clocks are seconds before the move; history ends at the current FEN. Unknown opponent clocks or completed durations are null. The host checks terminal positions, owns chess-game state, and rejects stale results. `targetSeconds` is total desired turn duration, including inference time, not additional sleep.

The package exports `TRAINED_TIME_CONTROLS` and `DEFAULT_TIME_CONTROL` (10+0) as metadata. It does not implement game-clock behavior.

## Consumer assets

Install the published version with `npm install --save-exact @epicliem/liemfish-browser@0.1.0`, then run:

```sh
npx liemfish-assets --source /path/to/chess-self --out public/chess/engine
npx liemfish-assets --verify --out public/chess/engine
```

Preparation verifies the exact personalized backbone and E05 epoch-5 timing head before writing. Model reports, move vocabulary, license notices, worker and WASM files are staged beside a deterministic `manifest.json`. Verification needs only the installed package and staged directory. It rejects missing files, changed hashes, incorrect model identity and a worker/runtime that differs from the installed package. A manifest records integrity, not provenance authenticity. Keep generated assets separate from GUI pieces. No asset download or network access happens in this CLI.

The selected hashes and feature algorithms remain owned here. The consumer should not copy engine algorithms or its tests. Its responsibilities are a thin adapter, preparing/verifying assets before builds, and full gameplay QA.

## Development

From `packages/browser`:

```sh
npm ci
npm run build
npm test
npm run test:package
npm pack
```

The client entry supports both ESM import and CommonJS require without browser globals; the worker remains ESM. Build bundles chess.js and ONNX Runtime into the worker and emits public declarations. Runtime files and dependency notices are included in the tarball; no dependencies need resolving in the consumer. Build/test tools are pinned in package-lock.json. The registry package is public. `npm pack` supports local release verification before publishing.

Source: extracted from the authorized portfolio implementation; source hashes are recorded in `SOURCE-PROVENANCE.json`. Existing encoding, legal-move sampling and continuous timing algorithms were preserved. The source license is AGPL-3.0; chess.js and ONNX Runtime notices retain their respective terms. Model weight licensing is separate, as documented in the root repository.
