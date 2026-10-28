/** Run the existing parity page in a fresh browser profile; no live chess account is used. */
import { createServer } from 'node:http';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
const root=resolve(fileURLToPath(new URL('..',import.meta.url)));
const browserRoot=resolve(root,'browser');
const output=resolve(root,'reports/timing-browser-selected');
const playwrightPath=process.env.PLAYWRIGHT_MODULE ?? '/Users/liem/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const executablePath=process.env.CHROME_EXECUTABLE ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const {chromium}=await import(pathToFileURL(playwrightPath));
const mime={'.html':'text/html','.mjs':'text/javascript','.js':'text/javascript','.wasm':'application/wasm','.css':'text/css','.json':'application/json'};
const server=createServer(async(req,res)=>{
 try {
  const url=new URL(req.url,'http://localhost');
  const path=resolve(browserRoot,'.'+decodeURIComponent(url.pathname==='/'?'/index.html':url.pathname));
  if(!path.startsWith(browserRoot+sep)){res.writeHead(403).end();return;}
  res.setHeader('Content-Type',mime[extname(path)]??'application/octet-stream');
  res.end(await readFile(path));
 }catch{res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
await mkdir(output,{recursive:true});
const url=`http://127.0.0.1:${server.address().port}/`;
let browser;
try {
 browser=await chromium.launch({executablePath,headless:true});
 for(const provider of ['wasm','webgpu']){
  const page=await browser.newPage();
  const messages=[];
  page.on('console',m=>messages.push({type:m.type(),text:m.text()}));
  page.on('pageerror',e=>messages.push({type:'pageerror',text:String(e)}));
  try{
   await page.goto(url);
   await page.locator('#feature-model').setInputFiles(resolve(root,'models/export/personalized-timing-features.onnx'));
   await page.locator('#side-model').setInputFiles(resolve(root,'models/export/e05-selected-epoch5.onnx'));
   await page.locator('#fixture').setInputFiles(resolve(root,'reports/e05-selected-browser-fixtures.json'));
   await page.locator('#provider').selectOption(provider);
   await page.locator('#repetitions').fill('3');
   await page.locator('#run').click();
   await page.waitForFunction(()=>{const s=document.querySelector('#report').textContent;return s&&JSON.parse(s).status!=='running';},{},{timeout:180000});
   const result=JSON.parse(await page.locator('#report').textContent());
   result.browser_test={headless:true,executable:executablePath,browser_version:browser.version(),page:url};
   await writeFile(resolve(output,`${provider}.json`),JSON.stringify(result,null,2)+'\n');
   console.log(provider,result.status,result.error??'',result.cases?.length??0);
   await page.screenshot({path:resolve(output,`${provider}.png`),fullPage:true});
  }catch(error){
   await writeFile(resolve(output,`${provider}-runner-error.json`),JSON.stringify({error:String(error),browser:browser.version(),messages},null,2)+'\n');
   console.log(provider,'RUNNER ERROR',String(error));
  }finally{
   await writeFile(resolve(output,`${provider}-console.json`),JSON.stringify(messages,null,2)+'\n');
   await page.close();
  }
 }
}finally{
 await browser?.close();
 await new Promise(resolve=>server.close(resolve));
}
