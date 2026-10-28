import * as ort from 'onnxruntime-web/wasm';
import { encodePosition, sampleLegalMove, type ModelPosition } from './model-encoding';
import { timingFeatures, selectedMoveInputs, sampleTiming } from './model-timing';

const MODEL_HASH = '8109a8d96e19e2d36de56018b8429b41a99c7c1adae6b8a4a2bdee1454cd6e7a';
const TIMING_HASH = '85b5dc3ab33dfde62ca511ea0c5a4f9e8ca1ece0846fa2bc2a1aaadca56a1921';
let session: ort.InferenceSession | undefined;
let timingSession: ort.InferenceSession | undefined;
let queue = Promise.resolve();

async function load(id: number) {
  if (session && timingSession) return;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = new URL('./runtime/', self.location.href).href;
  const response = await fetch(new URL(`./personalized-${MODEL_HASH}.onnx`, self.location.href), { cache: 'force-cache' });
  if (!response.ok) throw new Error(`Model download failed (${response.status}).`);
  const total = Number(response.headers.get('content-length')) || 91339625;
  const reader = response.body?.getReader();
  if (!reader) throw new Error('Streaming model download is unavailable.');
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    self.postMessage({ id, progress: { phase: 'downloading', loaded, total } });
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  chunks.length = 0;
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const hash = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('');
  if (hash !== MODEL_HASH) throw new Error('Model checksum mismatch. Please clear the browser cache and retry.');
  self.postMessage({ id, progress: { phase: 'initializing', loaded, total } });
  session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  const timingResponse = await fetch(new URL(`./timing-${TIMING_HASH}.onnx`, self.location.href), { cache: 'force-cache' });
  if (!timingResponse.ok) throw new Error(`Timing model download failed (${timingResponse.status}).`);
  const timingBytes = await timingResponse.arrayBuffer();
  const timingDigest = await crypto.subtle.digest('SHA-256', timingBytes);
  const timingHash = Array.from(new Uint8Array(timingDigest), value => value.toString(16).padStart(2, '0')).join('');
  if (timingHash !== TIMING_HASH) throw new Error('Timing model checksum mismatch.');
  timingSession = await ort.InferenceSession.create(timingBytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  self.postMessage({ id, progress: { phase: 'ready', loaded: loaded + timingBytes.byteLength, total: total + timingBytes.byteLength } });
}

self.onmessage = (event: MessageEvent<{ id: number; type: 'load' | 'predict'; position?: ModelPosition }>) => {
  const { id, type, position } = event.data;
  queue = queue.then(async () => {
    try {
      if (type === 'load') { await load(id); self.postMessage({ id, result: null }); return; }
      if (!session || !timingSession || !position) throw new Error('Load the model before requesting a move.');
      const feeds = {
        tokens: new ort.Tensor('float32', encodePosition(position), [1, 64, 99]),
        self_elos: new ort.Tensor('float32', Float32Array.of(position.eloSelf), [1]),
        oppo_elos: new ort.Tensor('float32', Float32Array.of(position.eloOpponent), [1]),
      };
      let outputs: ort.InferenceSession.OnnxValueMapType | undefined;
      try {
        if (position.remainingSeconds <= 0) throw new Error('No clock budget remains.');
        outputs = await session.run(feeds);
        const logits = outputs.move_logits.data as Float32Array;
        const uci = sampleLegalMove(position.fen, logits);
        const selected = selectedMoveInputs(position.fen, uci, logits);
        const side = {
          side_features: new ort.Tensor('float32', timingFeatures(position), [1, 50]),
          move_indices: new ort.Tensor('int64', selected.indices, [1, 2]),
          move_attributes: new ort.Tensor('float32', selected.attributes, [1, 8]),
        };
        let timingOutputs: ort.InferenceSession.OnnxValueMapType | undefined;
        try {
          timingOutputs = await timingSession.run({ ...side, pooled: outputs.pooled, base_logits: outputs.base_logits, squares: outputs.squares });
          const targetSeconds = sampleTiming(timingOutputs.timing_raw.data as Float32Array, position.remainingSeconds);
          self.postMessage({ id, result: { uci, targetSeconds } });
        } finally {
          Object.values(side).forEach(tensor => tensor.dispose());
          if (timingOutputs) Object.values(timingOutputs).forEach(tensor => tensor.dispose());
        }
      } finally {
        Object.values(feeds).forEach(tensor => tensor.dispose());
        if (outputs) Object.values(outputs).forEach(tensor => tensor.dispose());
      }
    } catch (error) {
      self.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
    }
  });
};
