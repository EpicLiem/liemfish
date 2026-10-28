import test from 'node:test';
import assert from 'node:assert/strict';
import { createModel } from './model';
const { disposeModel, loadModel, predict } = createModel();
Object.defineProperty(globalThis, 'location', { value: { href: 'https://example.com/' }, configurable: true });
import { Chess } from 'chess.js';

class FakeWorker {
  static instances: FakeWorker[] = [];
  onmessage?: (event: { data: unknown }) => void;
  onerror?: (event: { message: string }) => void;
  onmessageerror?: () => void;
  messages: { id: number; type: string }[] = [];
  terminated = false;
  constructor() { FakeWorker.instances.push(this); }
  postMessage(message: { id: number; type: string }) { this.messages.push(message); }
  terminate() { this.terminated = true; }
  reply(index: number, data: object) { this.onmessage?.({ data: { id: this.messages[index].id, ...data } }); }
}
const fen = new Chess().fen();
const position = { fen, historyFens: [fen], opponentRemainingSeconds: 300, ownTimeHistory: [], opponentTimeHistory: [], previousMoveWasCapture: false, initialSeconds: 300, incrementSeconds: 0, remainingSeconds: 300, eloSelf: 1500, eloOpponent: 1500 };

test('worker lazy load, abort, stale reply, disposal, failure and immediate retry', async () => {
  const original = globalThis.Worker;
  globalThis.Worker = FakeWorker as unknown as typeof Worker;
  try {
    assert.equal(FakeWorker.instances.length, 0);
    const firstLoad = loadModel();
    const first = FakeWorker.instances.at(-1)!;
    first.reply(0, { result: null });
    await firstLoad;
    const controller = new AbortController();
    const aborted = predict(position, { signal: controller.signal });
    await Promise.resolve();
    controller.abort();
    await assert.rejects(aborted, { name: 'AbortError' });
    first.reply(1, { result: { uci: 'e2e4', targetSeconds: 1 } });
    const move = predict(position);
    await Promise.resolve();
    first.reply(2, { result: { uci: 'd2d4', targetSeconds: 1 } });
    assert.deepEqual(await move, { uci: 'd2d4', targetSeconds: 1 });
    disposeModel();
    assert.equal(first.terminated, true);
    const disposedLoad = loadModel();
    const disposedCheck = assert.rejects(disposedLoad, { name: 'AbortError' });
    disposeModel();
    const retryLoad = loadModel();
    const retry = FakeWorker.instances.at(-1)!;
    await disposedCheck;
    assert.equal(retry.terminated, false, 'old rejected load must not dispose a new session');
    retry.reply(0, { result: null });
    await retryLoad;
    const failedMove = predict(position);
    await Promise.resolve();
    retry.onerror?.({ message: 'WASM failed' });
    await assert.rejects(failedMove, /WASM failed/);
    assert.equal(retry.terminated, true);
  } finally {
    disposeModel();
    globalThis.Worker = original;
  }
});

test('factories isolate sessions and resolve configurable public worker URLs', async () => {
  const original = globalThis.Worker;
  const urls: string[] = [];
  class LocatedWorker extends FakeWorker {
    constructor(url: URL) { super(); urls.push(url.href); }
  }
  globalThis.Worker = LocatedWorker as unknown as typeof Worker;
  const a = createModel({ assetBaseUrl: '/nested/engine' });
  const b = createModel();
  try {
    const la = a.loadModel();
    const wa = FakeWorker.instances.at(-1)!;
    const lb = b.loadModel();
    const wb = FakeWorker.instances.at(-1)!;
    wa.reply(0, { result: null }); wb.reply(0, { result: null });
    await Promise.all([la, lb]);
    assert.deepEqual(urls, ['https://example.com/nested/engine/model.worker.js', 'https://example.com/chess/engine/model.worker.js']);
    a.disposeModel();
    assert.equal(wb.terminated, false);
    const aborted = new AbortController(); aborted.abort();
    await assert.rejects(b.predict(position, { signal: aborted.signal }), { name: 'AbortError' });
    assert.equal(wb.messages.length, 1);
  } finally { a.disposeModel(); b.disposeModel(); globalThis.Worker = original; }
});
