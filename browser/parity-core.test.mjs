import test from 'node:test';
import assert from 'node:assert/strict';
import { validateFixture, compareTensor, filterInputs, evaluateCase, latencySummary } from './parity-core.mjs';

function spec(data = [1, 2], dtype = 'float32') { return { data, shape: [1, 2], dtype }; }
function fixture() {
  return { schema_version: 1, cases: [{ name: 'tiny',
    inputs: { tokens: spec(), self_elos: spec(), oppo_elos: spec() },
    side_inputs: { side_features: spec(), move_indices: spec([0, 1], 'int64'), move_attributes: spec(), history: spec() },
    expected: { move_logits: spec(), base_logits: spec(), pooled: spec(), squares: spec(), timing_raw: spec() },
  }] };
}

test('native fixture validates shape and integer move contract', () => {
  validateFixture(fixture());
  const f = fixture(); f.cases[0].side_inputs.move_indices.data[0] = .5;
  assert.throws(() => validateFixture(f), /safe integers/);
  const g = fixture(); g.cases[0].inputs.tokens.shape = [2, 2];
  assert.throws(() => validateFixture(g), /data length/);
  const h = fixture(); h.cases[0].inputs.tokens.data[0] = Infinity;
  assert.throws(() => validateFixture(h), /finite/);
});

test('fixture does not silently accept missing outputs or duplicate cases', () => {
  const f = fixture(); delete f.cases[0].expected.timing_raw;
  assert.throws(() => validateFixture(f), /timing_raw/);
  const g = fixture(); g.cases.push(g.cases[0]);
  assert.throws(() => validateFixture(g), /unique/);
});

test('allclose uses absolute plus relative tolerance on each element', () => {
  assert.equal(compareTensor([0.00001, 100.01], [1, 2], spec([0, 100]), { atol: .0001, rtol: .001 }).pass, true);
  const failed = compareTensor([.002, 101], [1, 2], spec([0, 100]), { atol: .0001, rtol: .001 });
  assert.equal(failed.pass, false); assert.equal(failed.failed_elements, 2);
  assert.equal(failed.max_abs_error, 1);
});

test('nonfinite actual values and wrong shapes fail parity', () => {
  const result = compareTensor([NaN, Infinity], [1, 2], spec());
  assert.equal(result.pass, false); assert.equal(result.nonfinite_elements, 2);
  assert.equal(compareTensor([1, 2], [2, 1], spec()).reason, 'shape_mismatch');
});

test('ONNX-pruned inputs are omitted and unexpected inputs fail clearly', () => {
  assert.deepEqual(filterInputs(['pooled'], { pooled: 1, history: 2 }), { pooled: 1 });
  assert.throws(() => filterInputs(['unknown'], {}), /unknown input/);
});

test('pipeline wires feature outputs to side model and checks all outputs', async () => {
  const tensors = [];
  class Tensor {
    constructor(dtype, data, dims) { Object.assign(this, { dtype, data, dims, disposed: false }); tensors.push(this); }
    async getData() { return this.data; }
    dispose() { this.disposed = true; }
  }
  const ort = { Tensor };
  const feature = {
    inputNames: ['tokens', 'self_elos', 'oppo_elos'],
    async run(inputs) {
      assert.equal(inputs.tokens.dtype, 'float32');
      return Object.fromEntries(['move_logits', 'base_logits', 'pooled', 'squares'].map(name => [name, new Tensor('float32', Float32Array.from([1, 2]), [1, 2])]));
    },
  };
  const side = {
    inputNames: ['pooled', 'base_logits', 'move_indices'],
    async run(inputs) {
      assert.deepEqual(Object.keys(inputs), this.inputNames);
      assert.equal(inputs.move_indices.data[1], 1n);
      return { timing_raw: new Tensor('float32', Float32Array.from([1, 2]), [1, 2]) };
    },
  };
  let clock = 0;
  const result = await evaluateCase(ort, feature, side, fixture().cases[0], undefined, () => ++clock);
  assert.equal(result.pass, true);
  assert.equal(result.batch_size, 1);
  assert.equal(result.latency_ms.pipeline, 3);
  assert.ok(tensors.every(x => x.disposed));
});

test('latency summary retains measured counts and quantiles', () => {
  const result = latencySummary([1, 4, 2]);
  assert.equal(result.samples, 3); assert.equal(result.median_ms, 2);
  assert.equal(result.min_ms, 1); assert.equal(result.max_ms, 4);
  assert.throws(() => latencySummary([]));
});

test('disabled side features may have a zero-width dimension', () => {
  const f = fixture(); f.cases[0].side_inputs.side_features = { data: [], shape: [1, 0] };
  validateFixture(f);
});
