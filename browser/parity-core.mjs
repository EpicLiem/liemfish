export const FEATURE_INPUTS = ['tokens', 'self_elos', 'oppo_elos'];
export const FEATURE_OUTPUTS = ['move_logits', 'base_logits', 'pooled', 'squares'];
export const SIDE_INPUTS = ['side_features', 'move_indices', 'move_attributes', 'history'];
export const EXPECTED_OUTPUTS = [...FEATURE_OUTPUTS, 'timing_raw'];

function tensorSpec(spec, name) {
  if (!spec || !Array.isArray(spec.shape) || !Array.isArray(spec.data)) throw new Error(`${name}: expected data and shape arrays`);
  if (spec.shape.length === 0 || !spec.shape.every(x => Number.isInteger(x) && x >= 0) || spec.shape[0] < 1) throw new Error(`${name}: shape requires a positive batch and nonnegative dimensions`);
  const count = spec.shape.reduce((a, b) => a * b, 1);
  if (!Number.isSafeInteger(count) || count !== spec.data.length) throw new Error(`${name}: data length does not match shape`);
  if (!spec.data.every(Number.isFinite)) throw new Error(`${name}: fixture values must be finite`);
  const dtype = spec.dtype ?? 'float32';
  if (!['float32', 'int64'].includes(dtype)) throw new Error(`${name}: unsupported dtype ${dtype}`);
  if (dtype === 'int64' && !spec.data.every(Number.isSafeInteger)) throw new Error(`${name}: int64 data must contain safe integers`);
}

export function validateFixture(fixture) {
  if (!fixture || fixture.schema_version !== 1) throw new Error('Expected fixture schema_version 1');
  if (!Array.isArray(fixture.cases) || fixture.cases.length === 0) throw new Error('Fixture has no cases');
  const names = new Set();
  for (const c of fixture.cases) {
    if (typeof c.name !== 'string' || !c.name || names.has(c.name)) throw new Error('Each case needs a unique name');
    names.add(c.name);
    for (const name of FEATURE_INPUTS) tensorSpec(c.inputs?.[name], `${c.name}.inputs.${name}`);
    for (const name of SIDE_INPUTS) tensorSpec(c.side_inputs?.[name], `${c.name}.side_inputs.${name}`);
    for (const name of EXPECTED_OUTPUTS) tensorSpec(c.expected?.[name], `${c.name}.expected.${name}`);
    const batch = c.inputs.tokens.shape[0];
    for (const spec of [...Object.values(c.inputs), ...Object.values(c.side_inputs), ...Object.values(c.expected)]) {
      if (spec.shape[0] !== batch) throw new Error(`${c.name}: inconsistent batch sizes`);
    }
    if (c.side_inputs.move_indices.dtype !== 'int64') throw new Error(`${c.name}: move_indices dtype must be int64`);
  }
  const tolerances = fixture.tolerances ?? { atol: 1e-4, rtol: 1e-3 };
  for (const name of ['atol', 'rtol']) {
    if (!Number.isFinite(tolerances[name]) || tolerances[name] < 0) throw new Error(`Invalid ${name}`);
  }
  return fixture;
}

export function compareTensor(actual, shape, expected, { atol = 1e-4, rtol = 1e-3 } = {}) {
  if (shape.length !== expected.shape.length || shape.some((n, i) => n !== expected.shape[i]) || actual.length !== expected.data.length) {
    return { pass: false, reason: 'shape_mismatch', actual_shape: [...shape], expected_shape: expected.shape };
  }
  let maxAbs = 0, maxRel = 0, failures = 0, nonfinite = 0;
  for (let i = 0; i < actual.length; i++) {
    const a = Number(actual[i]), e = expected.data[i];
    if (!Number.isFinite(a) || !Number.isFinite(e)) { nonfinite++; failures++; continue; }
    const absolute = Math.abs(a - e);
    maxAbs = Math.max(maxAbs, absolute);
    maxRel = Math.max(maxRel, absolute / Math.max(Math.abs(e), 1e-12));
    if (absolute > atol + rtol * Math.abs(e)) failures++;
  }
  return { pass: failures === 0, elements: actual.length, failed_elements: failures,
    nonfinite_elements: nonfinite, max_abs_error: maxAbs, max_rel_error: maxRel };
}

export function makeTensor(ort, spec) {
  const dtype = spec.dtype ?? 'float32';
  const values = dtype === 'int64' ? BigInt64Array.from(spec.data, x => BigInt(x)) : Float32Array.from(spec.data);
  return new ort.Tensor(dtype, values, spec.shape);
}

export function filterInputs(names, available) {
  return Object.fromEntries(names.map(name => {
    if (!(name in available)) throw new Error(`Model requires unknown input: ${name}`);
    return [name, available[name]];
  }));
}

/** Both model calls and output readback are timed, but fixture comparisons are not. */
export async function evaluateCase(ort, featureSession, sideSession, c, tolerances, now = () => performance.now()) {
  const tensors = new Set();
  const track = tensor => { tensors.add(tensor); return tensor; };
  try {
    const featureInputs = Object.fromEntries(FEATURE_INPUTS.map(name => [name, track(makeTensor(ort, c.inputs[name]))]));
    const extraInputs = Object.fromEntries(SIDE_INPUTS.map(name => [name, track(makeTensor(ort, c.side_inputs[name]))]));
    const start = now();
    const features = await featureSession.run(filterInputs(featureSession.inputNames, featureInputs));
    Object.values(features).forEach(track);
    const featureEnd = now();
    for (const name of FEATURE_OUTPUTS) if (!(name in features)) throw new Error(`Feature model missing output: ${name}`);
    const side = await sideSession.run(filterInputs(sideSession.inputNames, { ...features, ...extraInputs }));
    Object.values(side).forEach(track);
    const sideEnd = now();
    if (!side.timing_raw) throw new Error('Timing model missing timing_raw output');
    const outputs = { ...features, timing_raw: side.timing_raw };
    const data = {};
    for (const name of EXPECTED_OUTPUTS) {
      data[name] = typeof outputs[name].getData === 'function' ? await outputs[name].getData() : outputs[name].data;
    }
    const readEnd = now();
    const comparisons = Object.fromEntries(EXPECTED_OUTPUTS.map(name => [name, compareTensor(data[name], outputs[name].dims, c.expected[name], tolerances)]));
    return {
      name: c.name, batch_size: c.inputs.tokens.shape[0],
      pass: Object.values(comparisons).every(x => x.pass), outputs: comparisons,
      latency_ms: { feature: featureEnd - start, timing: sideEnd - featureEnd,
        output_readback: readEnd - sideEnd, pipeline: readEnd - start },
    };
  } finally {
    for (const tensor of tensors) if (typeof tensor.dispose === 'function') tensor.dispose();
  }
}

export function latencySummary(values) {
  if (!values.length || !values.every(x => Number.isFinite(x) && x >= 0)) throw new Error('Invalid latency samples');
  const sorted = [...values].sort((a, b) => a - b);
  const quantile = q => {
    const index = (sorted.length - 1) * q;
    const low = Math.floor(index), high = Math.ceil(index);
    return sorted[low] + (sorted[high] - sorted[low]) * (index - low);
  };
  return { samples: values.length, mean_ms: values.reduce((a, b) => a + b, 0) / values.length,
    median_ms: quantile(.5), p90_ms: quantile(.9), min_ms: sorted[0], max_ms: sorted.at(-1) };
}
