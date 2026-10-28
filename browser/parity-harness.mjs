import * as ort from './node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs';
import { validateFixture, evaluateCase, latencySummary } from './parity-core.mjs';
import { TurnScheduler, executeTurn } from './timing-runtime.mjs';

const byId = id => document.getElementById(id);
const form = byId('run-form');
let report = null;
let cancelled = false;
let activeScheduler = null;
let running = false;

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL('./node_modules/onnxruntime-web/dist/', import.meta.url).href;

function showReport() { byId('report').textContent = JSON.stringify(report, null, 2); }
function checkCancellation() { if (cancelled) throw new Error('Cancelled by user'); }
async function digest(buffer) {
  const value = await crypto.subtle.digest('SHA-256', buffer);
  return [...new Uint8Array(value)].map(x => x.toString(16).padStart(2, '0')).join('');
}
function resultRow(c) {
  const tr = document.createElement('tr');
  const outputs = c.cold.outputs;
  const maximum = Math.max(...Object.values(outputs).map(x => x.max_abs_error ?? Infinity));
  for (const text of [c.name, c.batch_size, c.pass ? 'Pass' : 'Fail', Number.isFinite(maximum) ? maximum.toExponential(3) : 'Shape mismatch', `${c.cold.latency_ms.pipeline.toFixed(2)} ms`, `${c.warm_latency.median_ms.toFixed(2)} ms`]) {
    const td = document.createElement('td'); td.textContent = String(text); tr.append(td);
  }
  byId('results').append(tr);
}

async function runScheduler(fixture, feature, side) {
  const s = fixture.scheduler;
  if (!s || !byId('scheduler').checked) return { status: 'not_run', reason: s ? 'disabled' : 'no_scheduler_fixture' };
  const c = fixture.cases.find(x => x.name === s.case_name);
  if (!c || !Array.isArray(s.legal_moves) || !s.legal_moves.includes(s.move_uci) || typeof s.position_key !== 'string') {
    throw new Error('Invalid scheduler fixture: case, position_key and selected legal move are required');
  }
  if (!Number.isFinite(s.target_seconds) || s.target_seconds < 0 || s.target_seconds > 10) {
    throw new Error('Scheduler diagnostic target must be between 0 and 10 seconds');
  }
  const scheduler = new TurnScheduler({ reserveSeconds: s.reserve_seconds ?? 0 });
  activeScheduler = scheduler;
  const token = scheduler.begin(s.position_key, s.remaining_seconds);
  let dummyCommits = 0;
  const event = await executeTurn({
    scheduler, token,
    infer: async () => {
      await evaluateCase(ort, feature, side, c, fixture.tolerances);
      checkCancellation();
      return { move_uci: s.move_uci, target_seconds: s.target_seconds };
    },
    wait: seconds => new Promise(resolve => setTimeout(resolve, seconds * 1000)),
    currentPosition: () => s.position_key,
    validateMove: move => s.legal_moves.includes(move),
    commitMove: () => { dummyCommits++; },
    incrementSeconds: 0,
  });
  activeScheduler = null;
  return { status: event.status, dummy_commits: dummyCommits,
    mode: 'local_fixture_only', selected_move_source: 'native_fixture',
    inference_includes_output_readback_and_parity_comparison: true, events: scheduler.events };
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (running) return;
  running = true; cancelled = false; report = null;
  byId('run').disabled = true; byId('cancel').disabled = false; byId('download').disabled = true;
  byId('results').replaceChildren(); byId('summary').textContent = 'Running…'; byId('report').textContent = '';
  let featureSession, sideSession;
  try {
    const files = { feature: byId('feature-model').files[0], side: byId('side-model').files[0], fixture: byId('fixture').files[0] };
    if (Object.values(files).some(x => !x)) throw new Error('Choose both models and the fixture');
    const repetitions = Number(byId('repetitions').value);
    if (!Number.isInteger(repetitions) || repetitions < 1 || repetitions > 20) throw new Error('Warm runs must be an integer from 1 to 20');
    const provider = byId('provider').value;
    if (!['wasm', 'webgpu'].includes(provider)) throw new Error('Choose WASM or WebGPU');
    if (provider === 'webgpu' && !navigator.gpu) throw new Error('WebGPU is unavailable in this browser/context. Run WASM or use a WebGPU-capable browser on localhost.');
    byId('status').textContent = 'Reading local files…';
    const fixtureBytes = await files.fixture.arrayBuffer();
    const fixture = validateFixture(JSON.parse(new TextDecoder().decode(fixtureBytes)));
    const featureBytes = await files.feature.arrayBuffer();
    const sideBytes = await files.side.arrayBuffer();
    report = {
      schema_version: 1, status: 'running', created_at: new Date().toISOString(),
      execution_provider_requested: provider, operator_placement_verified: false,
      onnxruntime_web_version: '1.31.1', user_agent: navigator.userAgent,
      secure_context: window.isSecureContext, cross_origin_isolated: window.crossOriginIsolated,
      wasm_threads: 1, warm_runs_per_case: repetitions,
      first_run_semantics: 'First call per case in one shared session; only the first case is session-cold',
      tolerances: fixture.tolerances ?? { atol: 1e-4, rtol: 1e-3 }, native_config: fixture.config ?? null,
      files: {
        feature: { name: files.feature.name, bytes: files.feature.size, sha256: await digest(featureBytes) },
        side: { name: files.side.name, bytes: files.side.size, sha256: await digest(sideBytes) },
        fixture: { name: files.fixture.name, bytes: files.fixture.size, sha256: await digest(fixtureBytes) },
      }, cases: [], scheduler: { status: 'not_run' },
    };
    checkCancellation();
    byId('status').textContent = `Loading models with ${provider.toUpperCase()}…`;
    const loadStart = performance.now();
    featureSession = await ort.InferenceSession.create(featureBytes, { executionProviders: [provider] });
    sideSession = await ort.InferenceSession.create(sideBytes, { executionProviders: [provider] });
    report.session_load_ms = performance.now() - loadStart;
    report.model_interfaces = { feature_inputs: featureSession.inputNames, feature_outputs: featureSession.outputNames,
      side_inputs: sideSession.inputNames, side_outputs: sideSession.outputNames };
    for (const c of fixture.cases) {
      checkCancellation();
      byId('status').textContent = `Checking ${c.name}, batch ${c.inputs.tokens.shape[0]}…`;
      const cold = await evaluateCase(ort, featureSession, sideSession, c, report.tolerances);
      const warm = [];
      for (let i = 0; i < repetitions; i++) {
        checkCancellation();
        warm.push(await evaluateCase(ort, featureSession, sideSession, c, report.tolerances));
      }
      const result = { name: c.name, batch_size: cold.batch_size, pass: cold.pass && warm.every(x => x.pass), cold, warm,
        warm_latency: latencySummary(warm.map(x => x.latency_ms.pipeline)) };
      report.cases.push(result); resultRow(result); showReport();
    }
    checkCancellation();
    byId('status').textContent = 'Checking scheduler instrumentation…';
    report.scheduler = await runScheduler(fixture, featureSession, sideSession);
    checkCancellation();
    report.status = report.cases.every(x => x.pass) ? 'parity_passed' : 'parity_failed';
    byId('summary').textContent = `${report.cases.filter(x => x.pass).length} of ${report.cases.length} cases passed. Scheduler: ${report.scheduler.status}.`;
    byId('status').textContent = report.status === 'parity_passed' ? 'Checks complete. Save the measured report.' : 'Parity failed. Inspect per-output errors before deployment.';
  } catch (error) {
    report ??= { schema_version: 1, created_at: new Date().toISOString(), cases: [] };
    report.status = cancelled ? 'cancelled' : 'error'; report.error = String(error);
    if (activeScheduler) report.scheduler = { status: 'cancelled_or_failed', events: activeScheduler.events };
    byId('status').textContent = report.error;
    byId('summary').textContent = 'Run did not complete. The report retains completed measurements and the error.';
  } finally {
    activeScheduler = null;
    const cleanup = await Promise.allSettled([featureSession?.release(), sideSession?.release()]);
    const cleanupErrors = cleanup.filter(x => x.status === 'rejected').map(x => String(x.reason));
    if (cleanupErrors.length && report) report.cleanup_errors = cleanupErrors;
    running = false; byId('run').disabled = false; byId('cancel').disabled = true;
    byId('download').disabled = report === null; showReport();
  }
});

byId('cancel').addEventListener('click', () => {
  cancelled = true; activeScheduler?.cancel('user_cancelled');
  byId('status').textContent = 'Cancelling after the current inference call…';
});
byId('download').addEventListener('click', () => {
  if (!report) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2) + '\n'], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url;
  a.download = `chess-timing-browser-${report.execution_provider_requested ?? 'error'}-${Date.now()}.json`;
  a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
