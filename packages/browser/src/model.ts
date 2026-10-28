import type { ModelPosition } from './model-encoding';
export type { ModelPosition } from './model-encoding';
export interface ModelProgress { phase: 'downloading' | 'initializing' | 'ready'; loaded: number; total?: number }
export interface ModelPrediction { uci: string; targetSeconds: number }
type Pending = { resolve: (value: ModelPrediction | null) => void; reject: (error: Error) => void; progress?: (progress: ModelProgress) => void; cleanup?: () => void };
export interface ModelOptions { assetBaseUrl?: string }

export function createModel(options: ModelOptions = {}) {
let worker: Worker | undefined;
let sequence = 0;
let loading: Promise<void> | undefined;
const pending = new Map<number, Pending>();

function disposeModel() {
  worker?.terminate();
  worker = undefined;
  loading = undefined;
  for (const request of pending.values()) {
    request.cleanup?.();
    request.reject(new DOMException('Model session disposed.', 'AbortError'));
  }
  pending.clear();
}

function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('model.worker.js', new URL((options.assetBaseUrl ?? '/chess/engine/').replace(/\/?$/, '/'), globalThis.location.href)), { type: 'module' });
  worker.onmessage = ({ data }) => {
    const request = pending.get(data.id);
    if (!request) return;
    if (data.progress) { request.progress?.(data.progress); return; }
    pending.delete(data.id);
    request.cleanup?.();
    if (data.error) request.reject(new Error(data.error));
    else request.resolve(data.result);
  };
  worker.onerror = (event) => {
    const error = new Error(event.message || 'The model worker failed. Retry loading the model.');
    for (const request of pending.values()) { request.cleanup?.(); request.reject(error); }
    pending.clear();
    disposeModel();
  };
  worker.onmessageerror = () => disposeModel();
  return worker;
}

function request(type: 'load' | 'predict', position?: ModelPosition, progress?: Pending['progress'], signal?: AbortSignal) {
  if (signal?.aborted) return Promise.reject(new DOMException('Prediction cancelled.', 'AbortError'));
  return new Promise<ModelPrediction | null>((resolve, reject) => {
    const id = ++sequence;
    const abort = () => {
      pending.delete(id);
      signal?.removeEventListener('abort', abort);
      reject(new DOMException('Prediction cancelled.', 'AbortError'));
    };
    pending.set(id, { resolve, reject, progress, cleanup: () => signal?.removeEventListener('abort', abort) });
    signal?.addEventListener('abort', abort, { once: true });
    try { getWorker().postMessage({ id, type, position }); }
    catch (error) { pending.delete(id); signal?.removeEventListener('abort', abort); reject(error); }
  });
}

function loadModel(onProgress?: (progress: ModelProgress) => void): Promise<void> {
  if (!loading) {
    const operation = request('load', undefined, onProgress);
    const activeWorker = worker;
    loading = operation.then(() => undefined).catch(error => {
      if (worker === activeWorker) disposeModel();
      throw error;
    });
  }
  return loading;
}

async function predict(position: ModelPosition, options: { signal?: AbortSignal } = {}): Promise<ModelPrediction> {
  if (!loading) throw new Error('Load the model before requesting a move.');
  await loading;
  return await request('predict', position, undefined, options.signal) as ModelPrediction;
}

return { loadModel, predict, disposeModel };
}
