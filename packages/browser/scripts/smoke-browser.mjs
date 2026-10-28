const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
const root = resolve('.test-consumer');
const server = createServer(async (req, res) => {
  try {
    if (req.url === '/') { res.setHeader('content-type','text/html'); res.end('<!doctype html><title>Liemfish engine smoke</title>'); return; }
    const path = resolve(root, '.' + decodeURIComponent(req.url.split('?')[0]));
    if (!path.startsWith(root + '/')) throw Error('path');
    const bytes = await readFile(path);
    res.setHeader('content-type', ({'.js':'text/javascript','.mjs':'text/javascript','.wasm':'application/wasm','.json':'application/json'})[extname(path)] || 'application/octet-stream');
    res.end(bytes);
  } catch { res.statusCode = 404; res.end(); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
let browser;
try {
  browser = await chromium.launch({executablePath:process.env.CHROME_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result = await page.evaluate(async () => {
    const {createModel} = await import('/node_modules/@epicliem/liemfish-browser/dist/index.js');
    const m = createModel({assetBaseUrl:'/public/chess/engine/'});
    const progress=[];
    try {
      await m.loadModel(p=>progress.push(p.phase));
      const fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
      const prediction=await m.predict({fen,historyFens:[fen],remainingSeconds:600,initialSeconds:600,incrementSeconds:0,eloSelf:491,eloOpponent:529,opponentRemainingSeconds:600,ownTimeHistory:[],opponentTimeHistory:[],previousMoveWasCapture:false});
      if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(prediction.uci) || !(prediction.targetSeconds>=0 && prediction.targetSeconds<600)) throw Error('Invalid prediction');
      return {prediction,phases:[...new Set(progress)]};
    } finally {m.disposeModel();}
  });
  console.log(JSON.stringify(result));
} finally { await browser?.close(); await new Promise(r=>server.close(r)); }
