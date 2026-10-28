import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile, readFile, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';
import { prepareAssets, verifyAssets } from './assets.mjs';
test('stages exact selected pair and detects tampered or missing assets without source', async () => {
  const out = await mkdtemp(join(tmpdir(), 'liemfish-assets-'));
  try {
    await prepareAssets('../..', out);
    const bin = join(out, 'liemfish-assets');
    await symlink(resolve('scripts/assets.mjs'), bin);
    assert.match(execFileSync(process.execPath, [bin, '--verify', '--out', out], { encoding: 'utf8' }), /Verified Liemfish engine assets/);
    assert.match(execFileSync(process.execPath, [bin, '--source', '../..', '--out', out], { encoding: 'utf8' }), /Verified Liemfish engine assets/);
    const manifest = await verifyAssets(out);
    assert.equal(Object.keys(manifest.files).length, 11);
    await writeFile(join(out, 'model.worker.js'), 'tampered');
    await assert.rejects(verifyAssets(out), /checksum mismatch/);
    assert.throws(() => execFileSync(process.execPath, [bin, '--verify', '--out', out], { stdio: 'pipe' }));
    const data = JSON.parse(await readFile(join(out, 'manifest.json')));
    delete data.files['model.worker.js'];
    await writeFile(join(out, 'manifest.json'), JSON.stringify(data));
    await assert.rejects(verifyAssets(out), /Invalid engine asset manifest/);
  } finally { await rm(out, { recursive: true, force: true }); }
});
