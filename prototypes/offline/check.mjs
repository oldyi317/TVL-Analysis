// 離線原型的自動檢查（Playwright Chromium、手機尺寸）。
// 用法：node prototypes/offline/check.mjs [--v1 out/v1] [--v2 out/v2] [--base /tvl-shell-proto] [--label v1] [--skip-swap]
// 逐項印 ✓／✗；有 ✗ 時結束碼 1。
import { createRequire } from 'node:module';
import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { startServer } from './serve.mjs';

const here = fileURLToPath(new URL('.', import.meta.url));
const require = createRequire(here + '../../web/package.json');
const { chromium, devices } = require('playwright');

const args = process.argv.slice(2);
const opt = (name, def) => (args.includes(`--${name}`) ? args[args.indexOf(`--${name}`) + 1] : def);
const v1Dir = opt('v1', here + 'out/v1');
const v2Dir = opt('v2', here + 'out/v2');
const base = opt('base', '/tvl-shell-proto');
const label = opt('label', 'v1');
const skipSwap = args.includes('--skip-swap');
const SEASON = '2026-27';
const MISS = '這一頁離線時看不到，連上網路後再試';

const results = [];
async function step(name, fn) {
	try {
		const note = await fn();
		results.push(true);
		console.log(`✓ ${name}${note ? ` — ${note}` : ''}`);
	} catch (e) {
		results.push(false);
		console.log(`✗ ${name} — ${String(e.message ?? e).split('\n')[0]}`);
	}
}
function assert(cond, msg) {
	if (!cond) throw new Error(msg);
}
const norm = (t) => t.replace(/\s+/g, ' ').trim();
const mainText = async (page) => norm(await page.locator('main').innerText());

async function waitText(page, pred, desc, timeout = 20000) {
	const deadline = Date.now() + timeout;
	let last = '';
	while (Date.now() < deadline) {
		try {
			last = await mainText(page);
			if (pred(last)) return last;
		} catch {
			/* main 尚未出現 */
		}
		await page.waitForTimeout(150);
	}
	throw new Error(`${desc}：逾時，頁面文字「${last.slice(0, 80)}」`);
}

async function openPanel(page) {
	if (!(await page.locator('[data-testid=proto-panel]').count())) await page.locator('[data-testid=proto-fab]').click();
	await page.locator('[data-testid=proto-panel]').waitFor();
}
const panelVal = async (page, testid) => {
	await openPanel(page);
	return (await page.locator(`[data-testid=${testid}]`).innerText()).trim();
};
const clickSetting = async (page, row, button) => {
	await openPanel(page);
	await page.locator(`[data-testid=proto-panel] .row:has-text("${row}") button:text-is("${button}")`).click();
};
async function waitControlled(page) {
	await page.waitForFunction(() => !!navigator.serviceWorker.controller, null, { timeout: 60000 });
}
async function waitPredl(page, timeout = 240000) {
	await openPanel(page);
	await page.waitForFunction(
		() => {
			const t = document.querySelector('[data-testid=predl]')?.textContent ?? '';
			const m = t.match(/(\d+)／(\d+)/);
			return !!m && m[1] === m[2] && Number(m[2]) > 0 && !t.includes('進行中') && !t.includes('舊版');
		},
		null,
		{ timeout }
	);
	return (await page.locator('[data-testid=predl]').innerText()).trim();
}
const versionOf = (dir) => JSON.parse(readFileSync(`${dir}/_app/version.json`, 'utf8')).version;

const server = await startServer({ dir: v1Dir, base });
const browser = await chromium.launch();
const device = devices['Pixel 7'];
const url = (p) => server.url + p;

try {
	// ---- 事前：另開一個乾淨 context 抓「線上的真實內容」當比對基準（不碰主 context 的快取）----
	const truth = {};
	{
		const ctx = await browser.newContext({ ...device, serviceWorkers: 'block' });
		const p = await ctx.newPage();
		const link = async (listPath, sel, idx) => {
			await p.goto(url(listPath));
			await p.waitForSelector('[data-hydrated]');
			const hrefs = await p.locator(sel).evaluateAll((as) => as.map((a) => a.getAttribute('href')));
			return hrefs[idx];
		};
		truth.match0 = await link(`${SEASON}/men/matches/`, `main a[href*="/${SEASON}/men/matches/"]`, 0);
		truth.match1 = await link(`${SEASON}/men/matches/`, `main a[href*="/${SEASON}/men/matches/"]`, 1);
		truth.player0 = await link(`${SEASON}/men/players/`, `main a[href*="/${SEASON}/men/players/"]`, 0);
		truth.old0 = await link('2020-21/men/matches/', 'main a[href*="/2020-21/men/matches/"]', 0);
		truth.old1 = await link('2020-21/men/matches/', 'main a[href*="/2020-21/men/matches/"]', 1);
		truth.old2 = await link('2020-21/men/matches/', 'main a[href*="/2020-21/men/matches/"]', 2);
		for (const k of ['match0', 'match1', 'player0', 'old2']) {
			assert(truth[k], `找不到連結 ${k}`);
			await p.goto(server.origin + truth[k]);
			await p.waitForSelector('[data-hydrated]');
			truth[k + 'Text'] = await mainText(p);
		}
		await ctx.close();
		console.log(`（基準頁面）單場 ${truth.match0}、${truth.match1}；球員 ${truth.player0}；歷史單場 ${truth.old0}、${truth.old1}、${truth.old2}`);
	}

	const ctx = await browser.newContext({ ...device });
	const page = await ctx.newPage();

	// ---- 1 ----
	await step('1. 線上開首頁：service worker 裝好並接手', async () => {
		await page.goto(url(''));
		await waitControlled(page);
		await page.waitForSelector('[data-hydrated]');
		const how = await panelVal(page, 'nav-how');
		return `SW ${await panelVal(page, 'sw-version')}，這次導覽：${how}`;
	});
	await step('1. 打開「當成獨立視窗」，當季預先下載跑完', async () => {
		await clickSetting(page, '當成獨立視窗', '開');
		const t = Date.now();
		const text = await waitPredl(page);
		return `${text}，${Math.round((Date.now() - t) / 1000)} 秒`;
	});
	await page.locator('[data-testid=proto-close]').click();

	// ---- 2 離線冷啟動 ----
	await ctx.setOffline(true);
	const cold = async (path, token, desc) => {
		const p = await ctx.newPage();
		await p.goto(url(path), { waitUntil: 'commit' });
		await p.waitForSelector('[data-hydrated]', { timeout: 20000 });
		const text = await waitText(p, (t) => t.includes(token), desc);
		await p.close();
		return text.length;
	};
	const cases = [
		['首頁', '', '最新比分'],
		[`${SEASON} 賽事列表`, `${SEASON}/men/matches/`, '三福產業'],
		[`${SEASON} 戰績`, `${SEASON}/men/standings/`, '雲林美津濃'],
		[`${SEASON} 球隊`, `${SEASON}/men/teams/`, '屏東台電'],
		[`${SEASON} 排行榜`, `${SEASON}/men/leaders/`, '本季四王'],
		[`${SEASON} 球員列表`, `${SEASON}/men/players/`, '張祐晨'],
		['歷史紀錄', 'records/men/', '單季紀錄'],
		['歷史賽季 2020-21 賽事列表', '2020-21/men/matches/', '桃園臺產']
	];
	for (const [name, path, token] of cases) {
		await step(`2. 離線冷啟動：${name}`, async () => `含「${token}」，內容 ${await cold(path, token, name)} 字`);
	}
	for (const [name, key] of [
		['現行賽季單場頁', 'match0'],
		['現行賽季球員頁', 'player0']
	]) {
		await step(`2. 離線冷啟動：${name}`, async () => {
			const p = await ctx.newPage();
			await p.goto(server.origin + truth[key], { waitUntil: 'commit' });
			await p.waitForSelector('[data-hydrated]', { timeout: 20000 });
			const want = truth[key + 'Text'].slice(0, 60);
			const text = await waitText(p, (t) => t.startsWith(want), name);
			const how = await panelVal(p, 'nav-how');
			await p.close();
			return `與線上內容一致（${text.length} 字），導覽：${how}`;
		});
	}

	// ---- 3 離線站內點進 ----
	await step('3. 離線站內點進現行賽季單場頁', async () => {
		const p = await ctx.newPage();
		await p.goto(url(`${SEASON}/men/matches/`), { waitUntil: 'commit' });
		await p.waitForSelector('[data-hydrated]');
		await p.locator(`main a[href="${truth.match1}"]`).first().click();
		const want = truth.match1Text.slice(0, 60);
		const text = await waitText(p, (t) => t.startsWith(want), '單場頁');
		await p.close();
		return `與線上內容一致（${text.length} 字）`;
	});

	// ---- 4 離線沒存的頁 ----
	const shellOk = async (p) => {
		assert((await p.locator('header').count()) > 0, '頂部列不在');
		assert((await p.locator('nav').count()) > 0, '分頁列不在');
	};
	await step('4. 離線直接開歷史賽季沒看過的單場頁：站內說明頁、殼層在', async () => {
		const p = await ctx.newPage();
		const reqs = [];
		p.on('request', (r) => r.isNavigationRequest() && r.frame() === p.mainFrame() && reqs.push(r.url()));
		await p.goto(server.origin + truth.old1, { waitUntil: 'commit' });
		await p.waitForSelector('[data-hydrated]', { timeout: 20000 });
		await waitText(p, (t) => t.includes(MISS), '說明頁');
		await shellOk(p);
		await p.waitForTimeout(1500);
		assert(reqs.length === 1, `疑似重載：整頁導覽請求 ${reqs.length} 次`);
		const back = p.locator('[data-testid=offline-miss] button');
		assert((await back.count()) === 1, '沒有返回鈕');
		await p.close();
		return `出現「${MISS}」，整頁導覽請求 ${reqs.length} 次，沒有重載`;
	});
	await step('4. 離線站內點進歷史賽季沒看過的單場頁：站內說明頁、殼層在', async () => {
		const p = await ctx.newPage();
		await p.goto(url('2020-21/men/matches/'), { waitUntil: 'commit' });
		await p.waitForSelector('[data-hydrated]');
		let navs = 0;
		p.on('request', (r) => r.isNavigationRequest() && r.frame() === p.mainFrame() && navs++);
		await p.locator(`main a[href="${truth.old0}"]`).first().click();
		await waitText(p, (t) => t.includes(MISS), '說明頁');
		await shellOk(p);
		await p.waitForTimeout(1500);
		// 返回鈕：回到列表
		await p.locator('[data-testid=offline-miss] button').click();
		await waitText(p, (t) => t.includes('桃園臺產'), '返回後列表');
		assert(navs === 0, `點擊後發生整頁導覽 ${navs} 次`);
		await p.close();
		return `出現說明頁，返回鈕回到列表（點擊後整頁導覽請求 ${navs} 次）`;
	});
	await step('4. 離線時頂部有「離線中，資料更新至 M/D」細條', async () => {
		const p = await ctx.newPage();
		await p.goto(url(`${SEASON}/men/standings/`), { waitUntil: 'commit' });
		await p.waitForSelector('[data-hydrated]');
		const bar = await p.locator('[data-testid=proto-offbar]').innerText({ timeout: 10000 });
		await p.close();
		return bar;
	});

	// ---- 5 看過就存 ----
	await ctx.setOffline(false);
	let seenText = '';
	await step('5. 線上直接開歷史單場頁，再離線重開同一頁：有內容（看過就存）', async () => {
		const p = await ctx.newPage();
		await p.goto(server.origin + truth.old2);
		await p.waitForSelector('[data-hydrated]');
		seenText = await mainText(p);
		await p.waitForFunction(
			async (path) => (await (await caches.open('data-v1')).keys()).some((r) => r.url.includes(path)),
			truth.old2,
			{ timeout: 15000 }
		);
		await p.close();
		await ctx.setOffline(true);
		const q = await ctx.newPage();
		await q.goto(server.origin + truth.old2, { waitUntil: 'commit' });
		await q.waitForSelector('[data-hydrated]', { timeout: 20000 });
		const text = await waitText(q, (t) => t.startsWith(seenText.slice(0, 60)), '離線重開');
		await q.close();
		await ctx.setOffline(false);
		return `離線重開與線上內容一致（${text.length} 字）`;
	});

	// ---- 6 快取清單 ----
	await step(`6. 寫出所有 Cache 的網址清單（cached-urls-${label}.json）`, async () => {
		const p = await ctx.newPage();
		await p.goto(url(''));
		await p.waitForSelector('[data-hydrated]');
		const groups = await p.evaluate(async () => {
			const out = {};
			for (const name of await caches.keys()) out[name] = (await (await caches.open(name)).keys()).map((r) => r.url);
			return out;
		});
		await p.close();
		mkdirSync(here + 'out', { recursive: true });
		const file = `${here}out/cached-urls-${label}.json`;
		writeFileSync(file, JSON.stringify(groups, null, 1));
		return Object.entries(groups).map(([k, v]) => `${k}:${v.length}`).join('、');
	});
	await step('6b. 人為延遲 5 秒、等待上限 2 秒：導覽回空殼頁（逾時）約 2 秒，且頁面有內容', async () => {
		const p = await ctx.newPage();
		await p.goto(url(`${SEASON}/men/standings/`));
		await p.waitForSelector('[data-hydrated]');
		await clickSetting(p, '導覽等待上限', '2 秒');
		await clickSetting(p, '人為延遲導覽回應', '5 秒');
		await p.waitForTimeout(500);
		await p.goto(url(`${SEASON}/men/standings/`), { waitUntil: 'commit' });
		await p.waitForSelector('[data-hydrated]', { timeout: 20000 });
		await waitText(p, (t) => t.includes('雲林美津濃'), '空殼頁內容');
		let how = '';
		for (let i = 0; i < 12 && !how.includes('逾時'); i++) {
			how = await panelVal(p, 'nav-how');
			if (!how.includes('逾時')) await p.waitForTimeout(500);
		}
		assert(how.includes('逾時'), `導覽結果：${how}`);
		const ms = Number(how.match(/(\d+) ms/)?.[1]);
		assert(ms >= 1900 && ms < 2600, `耗時 ${ms} ms`);
		await clickSetting(p, '導覽等待上限', '3 秒');
		await clickSetting(p, '人為延遲導覽回應', '0 秒');
		await p.close();
		return how;
	});
	await ctx.close();

	// ---- 7 換版 ----
	if (skipSwap || !existsSync(v2Dir)) {
		console.log('（略過換版檢查）');
	} else {
		const ver1 = versionOf(v1Dir);
		const ver2 = versionOf(v2Dir);
		console.log(`（換版）v1=${ver1}，v2=${ver2}`);
		// Chromium 模擬背景：覆寫 document.visibilityState 後發 visibilitychange
		const setVis = (p, state) =>
			p.evaluate((s) => {
				Object.defineProperty(document, 'visibilityState', { get: () => s, configurable: true });
				document.dispatchEvent(new Event('visibilitychange'));
			}, state);
		const fresh = async (path) => {
			server.setDir(v1Dir);
			const c = await browser.newContext({ ...device });
			const p = await c.newPage();
			await p.goto(url(path));
			await waitControlled(p);
			await p.waitForSelector('[data-hydrated]');
			assert((await panelVal(p, 'page-version')) === ver1, '頁面不是 v1');
			assert((await panelVal(p, 'sw-version')) === ver1, 'SW 不是 v1');
			await p.locator('[data-testid=proto-close]').click();
			return [c, p];
		};

		await step('7a. 瀏覽器分頁：離開再回來，出現「有新資料，點此更新」，點了換成 v2', async () => {
			const [c, p] = await fresh(`${SEASON}/men/standings/`);
			server.setDir(v2Dir);
			await setVis(p, 'hidden');
			await p.waitForTimeout(500);
			await setVis(p, 'visible');
			const banner = p.locator('[data-testid=proto-update]');
			await banner.waitFor({ timeout: 40000 });
			assert((await banner.innerText()).includes('有新資料，點此更新'), '條文字不對');
			// 等 12 秒（超過 10 秒門檻也不會自動換：瀏覽器分頁不管離開多久）
			await p.waitForTimeout(500);
			assert((await panelVal(p, 'page-version')) === ver1, '分頁不該自己換版');
			await p.locator('[data-testid=proto-close]').click();
			await Promise.all([p.waitForNavigation({ timeout: 30000 }), banner.click()]);
			await p.waitForSelector('[data-hydrated]');
			const pv = await panelVal(p, 'page-version');
			const sv = await panelVal(p, 'sw-version');
			assert(pv === ver2 && sv === ver2, `換版後頁面 ${pv}、SW ${sv}`);
			assert(p.url().endsWith(`${SEASON}/men/standings/`), `沒停在原頁：${p.url()}`);
			await c.close();
			return `頁面與 SW 都是 v2，停在 ${SEASON}/men/standings/`;
		});

		await step('7b. 瀏覽器分頁：離開超過門檻（10 秒）也只出條，不自動換', async () => {
			const [c, p] = await fresh(`${SEASON}/men/standings/`);
			await clickSetting(p, '自動換新門檻', '10 秒');
			await p.locator('[data-testid=proto-close]').click();
			server.setDir(v2Dir);
			await setVis(p, 'hidden');
			await p.waitForTimeout(11000);
			await setVis(p, 'visible');
			await p.locator('[data-testid=proto-update]').waitFor({ timeout: 40000 });
			assert((await panelVal(p, 'page-version')) === ver1, '分頁不該自己換版');
			await c.close();
			return '出現提示條，頁面仍是 v1';
		});

		await step('7c. 獨立視窗（除錯開關）門檻 10 秒：離開 11 秒後自動換成 v2，停在原頁', async () => {
			const [c, p] = await fresh(`${SEASON}/men/standings/`);
			await clickSetting(p, '當成獨立視窗', '開');
			await clickSetting(p, '自動換新門檻', '10 秒');
			await waitPredl(p);
			await p.locator('[data-testid=proto-close]').click();
			server.setDir(v2Dir);
			await setVis(p, 'hidden');
			await p.waitForTimeout(11000);
			await Promise.all([p.waitForNavigation({ timeout: 40000 }), setVis(p, 'visible')]);
			await p.waitForSelector('[data-hydrated]');
			const pv = await panelVal(p, 'page-version');
			const sv = await panelVal(p, 'sw-version');
			assert(pv === ver2 && sv === ver2, `換版後頁面 ${pv}、SW ${sv}`);
			assert(p.url().endsWith(`${SEASON}/men/standings/`), `沒停在原頁：${p.url()}`);
			const log = await p.locator('[data-testid=log]').innerText();
			await c.close();
			return `頁面與 SW 都是 v2，停在原頁；紀錄含「自動換新」：${log.includes('自動換新')}`;
		});

		await step('7d. 獨立視窗：離開 3 秒（低於門檻）只出條，不自動換', async () => {
			const [c, p] = await fresh(`${SEASON}/men/standings/`);
			await clickSetting(p, '當成獨立視窗', '開');
			await clickSetting(p, '自動換新門檻', '10 秒');
			await waitPredl(p);
			await p.locator('[data-testid=proto-close]').click();
			server.setDir(v2Dir);
			await setVis(p, 'hidden');
			await p.waitForTimeout(3000);
			await setVis(p, 'visible');
			await p.locator('[data-testid=proto-update]').waitFor({ timeout: 40000 });
			assert((await panelVal(p, 'page-version')) === ver1, '不該自己換版');
			await c.close();
			return '出現提示條，頁面仍是 v1';
		});

		await step('7e. 頁面本身已是新版（導覽 3 秒內拿到新 HTML）：新 SW 直接接手，不重載、不提示', async () => {
			const [c, p] = await fresh(`${SEASON}/men/standings/`);
			server.setDir(v2Dir);
			let loads = 0;
			p.on('load', () => loads++);
			await p.goto(url(`${SEASON}/men/teams/`));
			await p.waitForSelector('[data-hydrated]');
			assert((await panelVal(p, 'page-version')) === ver2, '導覽應拿到 v2 HTML');
			await p.waitForFunction(() => document.querySelector('[data-testid=sw-version]')?.textContent?.trim() !== '', null);
			await p.waitForFunction(
				(v) => document.querySelector('[data-testid=sw-version]')?.textContent?.trim() === v,
				ver2,
				{ timeout: 40000 }
			);
			assert((await p.locator('[data-testid=proto-update]').count()) === 0, '不該出現提示條');
			assert(loads === 1, `不該重載，load 事件 ${loads} 次`);
			await c.close();
			return 'SW 換成 v2，無提示條，無重載';
		});
	}
} finally {
	await browser.close();
	await server.close();
}

const ok = results.filter(Boolean).length;
console.log(`\n總結：${ok}／${results.length} 通過，${results.length - ok} 項 ✗`);
process.exit(ok === results.length ? 0 : 1);
