// 算首次下載量與 v1 → v2 換版重抓量（gzip level 6 為估計值；GitHub Pages 實際壓縮率部署後用 measure-live.sh 量）。
// 用法：node prototypes/offline/measure.mjs [--v1 out/v1] [--v2 out/v2] [--urls out/cached-urls-v1.json] [--season 2026-27]
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

const here = fileURLToPath(new URL('.', import.meta.url));
const args = process.argv.slice(2);
const opt = (name, def) => (args.includes(`--${name}`) ? args[args.indexOf(`--${name}`) + 1] : def);
const v1 = opt('v1', here + 'out/v1');
const v2 = opt('v2', here + 'out/v2');
const urlsFile = opt('urls', here + 'out/cached-urls-v1.json');
const season = opt('season', '2026-27');

const cached = JSON.parse(readFileSync(urlsFile, 'utf8'));
const groupOf = (prefix) => Object.entries(cached).find(([k]) => k.startsWith(prefix))?.[1] ?? [];
const toPath = (u) => decodeURIComponent(new URL(u).pathname).replace(/^\/[^/]+\//, ''); // 去掉 /<base>/
const pre1 = groupOf('precache-').map(toPath);
const data1 = groupOf('data-').map(toPath);

const info = new Map();
function stat(dir, rel) {
	const key = `${dir}\0${rel}`;
	if (!info.has(key)) {
		const buf = readFileSync(`${dir}/${rel}`);
		info.set(key, { raw: buf.length, gz: gzipSync(buf, { level: 6 }).length, hash: createHash('sha1').update(buf).digest('hex') });
	}
	return info.get(key);
}
function walk(dir, sub = '') {
	return readdirSync(`${dir}/${sub}`).flatMap((n) => {
		const rel = sub ? `${sub}/${n}` : n;
		return statSync(`${dir}/${rel}`).isDirectory() ? walk(dir, rel) : [rel];
	});
}

const kind = (p) => (p === 'shell.html' ? '空殼頁' : p.startsWith('_app/') ? '程式檔' : '隊徽與其他靜態檔');
function sum(dir, paths) {
	return paths.reduce((a, p) => {
		const s = stat(dir, p);
		return { n: a.n + 1, raw: a.raw + s.raw, gz: a.gz + s.gz };
	}, { n: 0, raw: 0, gz: 0 });
}
const kb = (n) => (n / 1024).toFixed(1).padStart(9) + ' KB';
const row = (name, s) => console.log(`  ${name.padEnd(26)}${String(s.n).padStart(6)} 檔 ${kb(s.raw)} ${kb(s.gz)}`);
const head = (title) => console.log(`\n${title}\n  ${'項目'.padEnd(24)}${'檔數'.padStart(7)}${'原始大小'.padStart(13)}${'gzip(6)'.padStart(13)}`);

// 用 build 目錄重建「預存內容」清單，並與 v1 實際快取的清單核對
const pre2 = (() => {
	const all = walk(v2);
	const skip = /^(robots\.txt|og-image\.(png|svg))$/;
	return all.filter((p) => p.startsWith('_app/immutable/') || p === 'shell.html' || p === 'manifest.webmanifest' || p === 'favicon.svg' || p.startsWith('icons/') || p.startsWith('logos/')).filter((p) => !skip.test(p));
})();
const set1 = new Set(pre1);
const mismatch = pre2.filter((p) => !p.startsWith('_app/') && !set1.has(p));
if (mismatch.length) console.log(`注意：v2 重建的靜態檔清單有 ${mismatch.length} 個不在 v1 快取清單，例：${mismatch.slice(0, 3)}`);

head(`首次下載（瀏覽器分頁訪客；快取清單 ${urlsFile.replace(here, '')}）`);
const byKind = {};
for (const p of pre1) (byKind[kind(p)] ??= []).push(p);
let first = { n: 0, raw: 0, gz: 0 };
for (const k of ['程式檔', '隊徽與其他靜態檔', '空殼頁']) {
	const s = sum(v1, byKind[k] ?? []);
	row(k, s);
	first = { n: first.n + s.n, raw: first.raw + s.raw, gz: first.gz + s.gz };
}
row('合計', first);
console.log(`  → gzip 後 ${(first.gz / 1048576).toFixed(2)} MB，${first.gz <= 5 * 1048576 ? '在 5 MB 以內' : '超過 5 MB'}`);

head(`獨立視窗多出來的（${season} 單場頁與球員頁的 __data.json，v1）`);
const seasonData1 = data1.filter((p) => p.startsWith(`${season}/`) && p.endsWith('__data.json'));
const sd1 = sum(v1, seasonData1);
row('當季資料檔', sd1);
const otherData = data1.length - seasonData1.length;
console.log(`  （另有 ${otherData} 個看過就存的其他資料檔，不計入）`);
console.log(`  → 獨立視窗首次合計 gzip ${((first.gz + sd1.gz) / 1048576).toFixed(2)} MB`);

// ---- v1 → v2 ----
head('v1 → v2 換版重抓（預存內容：網址變了或內容變了的檔）');
const changedPre = pre2.filter((p) => !set1.has(p) || stat(v1, p).hash !== stat(v2, p).hash);
const cp = {};
for (const p of changedPre) (cp[kind(p)] ??= []).push(p);
let upd = { n: 0, raw: 0, gz: 0 };
for (const k of ['程式檔', '隊徽與其他靜態檔', '空殼頁']) {
	const s = sum(v2, cp[k] ?? []);
	row(k, s);
	upd = { n: upd.n + s.n, raw: upd.raw + s.raw, gz: upd.gz + s.gz };
}
row('預存內容小計', upd);
const codeV1 = pre1.filter((p) => p.startsWith('_app/'));
const codeKept = codeV1.filter((p) => pre2.includes(p));
console.log(`  v1 與 v2 之間程式檔沒變的：${codeKept.length}／${codeV1.length}（v2 共 ${pre2.filter((p) => p.startsWith('_app/')).length} 個程式檔，新增 ${pre2.filter((p) => p.startsWith('_app/') && !set1.has(p)).length}）`);

console.log('\n  變動最大的前 10 個檔（依 gzip 大小）');
for (const p of changedPre.map((p) => [p, stat(v2, p)]).sort((a, b) => b[1].gz - a[1].gz).slice(0, 10)) {
	console.log(`    ${kb(p[1].gz)} gz ${kb(p[1].raw)} raw  ${p[0]}${set1.has(p[0]) ? '' : '  (新網址)'}`);
}

head(`獨立視窗多出來的（當季資料檔，v2）`);
const seasonData2 = walk(v2, season).filter((p) => /\/(matches|players)\/[^/]+\/__data\.json$/.test(p));
const all2 = sum(v2, seasonData2);
const changedData = seasonData2.filter((p) => !seasonData1.includes(p) || stat(v1, p).hash !== stat(v2, p).hash);
const chg2 = sum(v2, changedData);
row('全部重抓', all2);
row('只算內容有變的', chg2);
console.log('  （SW 用 cache:no-cache 重新驗證；伺服器有 ETag 時沒變的回 304，實際接近「只算內容有變的」，沒有 ETag 就是「全部重抓」）');

head('每日更新一次的總量（v1 → v2）');
row('瀏覽器分頁訪客', upd);
row('獨立視窗（資料全部重抓）', { n: upd.n + all2.n, raw: upd.raw + all2.raw, gz: upd.gz + all2.gz });
row('獨立視窗（只算有變）', { n: upd.n + chg2.n, raw: upd.raw + chg2.raw, gz: upd.gz + chg2.gz });
const worst = upd.gz + all2.gz;
console.log(`  → 最壞情況 gzip ${(worst / 1048576).toFixed(2)} MB，${worst <= 3 * 1048576 ? '在 3 MB 以內' : '超過 3 MB'}`);
