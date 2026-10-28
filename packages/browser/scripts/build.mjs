import { build } from 'esbuild';
import { mkdir, copyFile } from 'node:fs/promises';
await mkdir('dist/runtime', { recursive: true });
await build({ entryPoints: ['src/index.ts'], outfile: 'dist/index.js', bundle: true, format: 'esm', platform: 'browser', target: 'es2022' });
await build({ entryPoints: ['src/index.ts'], outfile: 'dist/index.cjs', bundle: true, format: 'cjs', platform: 'browser', target: 'es2022' });
await build({ entryPoints: ['src/model.worker.ts'], outfile: 'dist/model.worker.js', bundle: true, format: 'esm', platform: 'browser', target: 'es2022' });
for (const name of ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs']) {
  await copyFile(`node_modules/onnxruntime-web/dist/${name}`, `dist/runtime/${name}`);
}
await copyFile('node_modules/chess.js/LICENSE', 'notices/CHESS-JS-LICENSE');
