#!/usr/bin/env node
import { readFile, writeFile, mkdir, realpath } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const MODEL_HASH = '8109a8d96e19e2d36de56018b8429b41a99c7c1adae6b8a4a2bdee1454cd6e7a';
export const TIMING_HASH = '85b5dc3ab33dfde62ca511ea0c5a4f9e8ca1ece0846fa2bc2a1aaadca56a1921';
const CHECKPOINT_HASH = '8758ea33f911233b28c2c134192a3f127b5c7669e22c08178417e41393c40427';
const packageVersion = JSON.parse(await readFile(resolve(root, 'package.json'), 'utf8')).version;
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const modelFiles = {
  [`personalized-${MODEL_HASH}.onnx`]: 'models/export/personalized-timing-features.onnx',
  [`timing-${TIMING_HASH}.onnx`]: 'models/export/e05-selected-epoch5.onnx',
  'personalized-timing-features.json': 'models/export/personalized-timing-features.json',
  'personalized-timing-features.moves.json': 'models/export/personalized-timing-features.moves.json',
  'e05-selected-epoch5.json': 'models/export/e05-selected-epoch5.json',
  'MAIA3-LICENSE': 'LICENSE',
};
const packageFiles = {
  'model.worker.js': 'dist/model.worker.js',
  'runtime/ort-wasm-simd-threaded.wasm': 'dist/runtime/ort-wasm-simd-threaded.wasm',
  'runtime/ort-wasm-simd-threaded.mjs': 'dist/runtime/ort-wasm-simd-threaded.mjs',
  'runtime/LICENSE': 'notices/ONNXRUNTIME-LICENSE',
  'CHESS-JS-LICENSE': 'notices/CHESS-JS-LICENSE',
};
async function validateModels(files) {
  const report = JSON.parse(files['personalized-timing-features.json']);
  const timingReport = JSON.parse(files['e05-selected-epoch5.json']);
  if (!report.parity?.passed || report.checkpoint_sha256 !== CHECKPOINT_HASH || report.onnx_sha256 !== MODEL_HASH || digest(files[`personalized-${MODEL_HASH}.onnx`]) !== MODEL_HASH) throw new Error('Unexpected or unverified personalized checkpoint.');
  if (!timingReport.parity?.passed || timingReport.base_checkpoint_sha256 !== CHECKPOINT_HASH || timingReport.onnx_sha256 !== TIMING_HASH || digest(files[`timing-${TIMING_HASH}.onnx`]) !== TIMING_HASH) throw new Error('Unexpected or unverified timing head.');
}
export async function prepareAssets(source, out) {
  const files = {};
  for (const [name, path] of Object.entries(modelFiles)) files[name] = await readFile(resolve(source, path));
  for (const [name, path] of Object.entries(packageFiles)) files[name] = await readFile(resolve(root, path));
  await validateModels(files);
  const manifest = { schemaVersion: 1, package: '@epicliem/liemfish-browser', version: packageVersion, runtime: 'onnxruntime-web@1.31.1', files: {} };
  for (const [name, bytes] of Object.entries(files)) {
    await mkdir(dirname(resolve(out, name)), { recursive: true });
    await writeFile(resolve(out, name), bytes);
    manifest.files[name] = { bytes: bytes.length, sha256: digest(bytes) };
  }
  await writeFile(resolve(out, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  await verifyAssets(out);
}
export async function verifyAssets(out) {
  const manifest = JSON.parse(await readFile(resolve(out, 'manifest.json'), 'utf8'));
  const expected = [...Object.keys(modelFiles), ...Object.keys(packageFiles)].sort();
  if (manifest.schemaVersion !== 1 || manifest.package !== '@epicliem/liemfish-browser' || manifest.version !== packageVersion || JSON.stringify(Object.keys(manifest.files).sort()) !== JSON.stringify(expected)) throw new Error('Invalid engine asset manifest.');
  const files = {};
  for (const name of expected) {
    const bytes = await readFile(resolve(out, name));
    const entry = manifest.files[name];
    if (bytes.length !== entry.bytes || digest(bytes) !== entry.sha256) throw new Error(`Asset checksum mismatch: ${name}`);
    files[name] = bytes;
  }
  await validateModels(files);
  // Verify packaged runtime and worker independently of the consumer manifest.
  for (const [name, path] of Object.entries(packageFiles)) {
    if (digest(files[name]) !== digest(await readFile(resolve(root, path)))) throw new Error(`Asset differs from installed package: ${name}`);
  }
  return manifest;
}
if (process.argv[1] && await realpath(process.argv[1]).catch(() => '') === fileURLToPath(import.meta.url)) {
  try {
    const { values } = parseArgs({ options: { source: { type: 'string' }, out: { type: 'string' }, verify: { type: 'boolean' } } });
    if (!values.out || (!values.verify && !values.source)) throw new Error('Usage: liemfish-assets --source /path/to/liemfish --out public/chess/engine OR --verify --out public/chess/engine');
    if (values.verify) await verifyAssets(resolve(values.out));
    else await prepareAssets(resolve(values.source), resolve(values.out));
    console.log(`Verified Liemfish engine assets: ${resolve(values.out)}`);
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
