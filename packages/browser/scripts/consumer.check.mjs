import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { execFileSync } from 'node:child_process';

test('packed package imports in ESM and requires in CommonJS without browser globals', async () => {
  const root = await mkdtemp(join(tmpdir(), 'liemfish-consumer-'));
  try {
    const packed = JSON.parse(execFileSync('npm', ['pack', '--ignore-scripts', '--json', '--pack-destination', root], { encoding: 'utf8' }))[0];
    assert.ok(packed.files.some(file => file.path === 'dist/index.cjs'));
    assert.ok(!packed.files.some(file => /\.(onnx|pt|pgn|jsonl)$/.test(file.path)));
    await writeFile(join(root, 'package.json'), JSON.stringify({ name: 'liemfish-clean-consumer', private: true }));
    execFileSync('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund', '--offline', resolve(root, packed.filename)], { cwd: root, stdio: 'pipe' });
    const installed = JSON.parse(await (await import('node:fs/promises')).readFile(join(root, 'node_modules/@epicliem/liemfish-browser/package.json'), 'utf8'));
    assert.equal(installed.bin['liemfish-assets'], 'scripts/assets.mjs');
    assert.throws(() => execFileSync(join(root, 'node_modules/.bin/liemfish-assets'), [], { cwd: root, stdio: 'pipe' }), error => error.status === 1 && error.stderr.toString().includes('Usage:'));
    const checks = `
      if (typeof globalThis.Worker !== 'undefined' || typeof globalThis.location !== 'undefined') throw Error('Expected no browser globals');
      if (api.DEFAULT_TIME_CONTROL.id !== '10+0') throw Error('Metadata missing');
      const model = api.createModel();
      for (const name of ['loadModel', 'predict', 'disposeModel']) if (typeof model[name] !== 'function') throw Error(name);
      model.disposeModel();
      console.log('consumer passed');
    `;
    for (const [name, prefix] of [['consumer.mjs', "import * as api from '@epicliem/liemfish-browser';"], ['consumer.cjs', "const api = require('@epicliem/liemfish-browser');"]]) {
      await writeFile(join(root, name), prefix + checks);
      assert.match(execFileSync(process.execPath, [name], { cwd: root, encoding: 'utf8' }), /consumer passed/);
    }
  } finally { await rm(root, { recursive: true, force: true }); }
});
