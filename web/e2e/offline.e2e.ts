import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { type Page } from '@playwright/test';
import {
	BASE,
	OLD_SEASON,
	asStandalone,
	currentMatches,
	currentSeason,
	expect,
	findSeason,
	loadAndWaitControlled,
	openMoreSheet,
	readJson,
	test
} from './helpers.js';

test.use({ serviceWorkers: 'allow' });

const shellHtml = readFileSync('build/shell.html', 'utf8');
const { version } = JSON.parse(readFileSync('build/_app/version.json', 'utf8'));
const MISS = '這一頁離線時看不到，連上網路後再試';
const playedSeason = findSeason('played')?.season;
const playerSeason = findSeason('players');

const noKitErrorPage = (page: Page) => expect(page.getByText('Internal Error')).toHaveCount(0);

function playerUrl() {
	const player = readJson('players').players.find(
		(p: { group: string; seasons: { cup_id: number }[] }) =>
			p.group === 'men' && p.seasons.some((s) => s.cup_id === playerSeason!.cupId)
	);
	return `${BASE}/${playerSeason!.season}/men/players/${player.player_id}/`;
}

test('註冊的路徑與範圍從 base 推導', async ({ page }) => {
	await loadAndWaitControlled(page, `${BASE}/${currentSeason}/men/`);
	const { scope, script } = await page.evaluate(async () => {
		const registration = await navigator.serviceWorker.ready;
		return { scope: registration.scope, script: registration.active!.scriptURL };
	});
	expect(new URL(scope).pathname).toBe(`${BASE}/`);
	expect(new URL(script).pathname).toBe(`${BASE}/service-worker.js`);
});

test('預存以這一版的版本命名，啟用時清掉舊版的預存', async ({ page }) => {
	await page.addInitScript(() => void caches.open('precache-old'));
	await loadAndWaitControlled(page, `${BASE}/${currentSeason}/men/`);
	expect(await page.evaluate(() => caches.keys())).toEqual([`precache-${version}`]);
});

test('線上且已被接手：整頁載入列表頁、用戶端導覽到單場頁都正常', async ({ page }) => {
	test.skip(!playedSeason, '所有賽季的男子組都沒有已完賽場次');
	const list = `${BASE}/${playedSeason}/men/matches/`;
	await loadAndWaitControlled(page, list);
	await page.reload();
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${playedSeason} 男子組 賽事`);
	const card = page.locator('.card-link').first();
	const href = (await card.getAttribute('href'))!;
	await card.click();
	await expect(page).toHaveURL(href);
	await expect(page.getByRole('heading', { level: 1 })).toContainText('對');
});

test.describe('桌面寬度', () => {
	test.use({ viewport: { width: 1280, height: 800 } });

	test('線上且已被接手：整頁載入與側欄導覽都正常', async ({ page }) => {
		await loadAndWaitControlled(page, `${BASE}/${currentSeason}/men/standings/`);
		await page.reload();
		await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${currentSeason} 男子組 戰績`);
		await page.getByRole('navigation', { name: '頁面' }).getByRole('link', { name: '排行榜', exact: true }).click();
		await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${currentSeason} 男子組 排行榜`);
	});
});

const listPages = [
	{ name: '首頁', path: `/${currentSeason}/men/`, h1: `${currentSeason} 男子組` },
	{ name: '賽事列表', path: `/${currentSeason}/men/matches/`, h1: `${currentSeason} 男子組 賽事` },
	{ name: '戰績', path: `/${currentSeason}/men/standings/`, h1: `${currentSeason} 男子組 戰績` },
	{ name: '球隊', path: `/${currentSeason}/men/teams/`, h1: `${currentSeason} 男子組 球隊` },
	{ name: '排行榜', path: `/${currentSeason}/men/leaders/`, h1: `${currentSeason} 男子組 排行榜` },
	{ name: '球員列表', path: `/${currentSeason}/men/players/`, h1: '球員' },
	{ name: '歷史紀錄', path: '/records/men/', h1: '男子組 歷史紀錄' },
	{ name: '加到主畫面', path: '/notifications/', h1: '加到主畫面' },
	{ name: '非當季賽季的戰績', path: `/${OLD_SEASON}/men/standings/`, h1: `${OLD_SEASON} 男子組 戰績` }
];

for (const { name, path, h1 } of listPages) {
	test(`離線也開得起來：${name}（由空殼頁畫出）`, async ({ page, context }) => {
		await loadAndWaitControlled(page, `${BASE}${path}`);
		await context.setOffline(true);
		const response = await page.goto(`${BASE}${path}`);
		expect(response!.fromServiceWorker()).toBe(true);
		expect(await response!.text()).toBe(shellHtml);
		await expect(page.getByRole('heading', { level: 1 })).toHaveText(h1);
	});
}

test('離線點到沒存的頁：顯示說明頁與返回鈕，返回回到賽事列表', async ({ page, context }) => {
	test.skip(!playedSeason, '所有賽季的男子組都沒有已完賽場次');
	const list = `${BASE}/${playedSeason}/men/matches/`;
	await loadAndWaitControlled(page, list);
	await context.setOffline(true);
	await page.locator('.card-link').first().click();
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(MISS);
	await noKitErrorPage(page);
	await page.getByRole('button', { name: '返回' }).click();
	await expect(page).toHaveURL(list);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${playedSeason} 男子組 賽事`);
});

test('離線直接開沒存的球員頁：由空殼頁啟動，顯示說明頁，返回到上一頁', async ({ page, context }) => {
	test.skip(!playerSeason, '所有賽季的男子組都沒有球員');
	const home = `${BASE}/${currentSeason}/men/`;
	await loadAndWaitControlled(page, home);
	await context.setOffline(true);
	const response = await page.goto(playerUrl());
	expect(response!.fromServiceWorker()).toBe(true);
	expect(await response!.text()).toBe(shellHtml);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(MISS);
	await noKitErrorPage(page);
	await page.getByRole('button', { name: '返回' }).click();
	await expect(page).toHaveURL(home);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${currentSeason} 男子組`);
});

test('離線直接開沒存的球員頁、沒有上一頁：返回到當季首頁', async ({ page, context }) => {
	test.skip(!playerSeason, '所有賽季的男子組都沒有球員');
	await loadAndWaitControlled(page, `${BASE}/${OLD_SEASON}/men/standings/`);
	await context.setOffline(true);
	// window.open 開出的分頁沒有上一頁，像從主畫面或連結直接開到這一頁（newPage 再 goto 會留一筆 about:blank）
	const [fresh] = await Promise.all([
		context.waitForEvent('page'),
		page.evaluate((url) => void window.open(url), playerUrl())
	]);
	await expect(fresh.getByRole('heading', { level: 1 })).toHaveText(MISS);
	await fresh.getByRole('button', { name: '返回' }).click();
	await expect(fresh).toHaveURL(`${BASE}/${currentSeason}/men/`);
	await expect(fresh.getByRole('heading', { level: 1 })).toHaveText(`${currentSeason} 男子組`);
});

const SLOW = `${BASE}/${currentSeason}/men/standings/`;
const marker = '<html data-hydrated><body><h1>network-marker</h1></body></html>';
const markerResponse = { status: 200, contentType: 'text/html', body: marker };

test('網路 3 秒內回來就用網路上的新頁面', async ({ page, context }) => {
	await loadAndWaitControlled(page, `${BASE}/${currentSeason}/men/`);
	await context.route(`**${SLOW}`, (route) => route.fulfill(markerResponse));
	await page.goto(SLOW);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText('network-marker');
});

test('網路慢於 3 秒就改用存好的空殼頁畫出頁面', async ({ page, context }) => {
	await loadAndWaitControlled(page, `${BASE}/${currentSeason}/men/`);
	// 網路 4.5 秒才回：畫面來自空殼頁，表示等待上限短於 4.5 秒；下面再斷言至少等了約 3 秒
	await context.route(`**${SLOW}`, async (route) => {
		await new Promise((resolve) => setTimeout(resolve, 4500));
		await route.fulfill(markerResponse).catch(() => {});
	});
	const start = Date.now();
	const response = await page.goto(SLOW);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${currentSeason} 男子組 戰績`, {
		timeout: 15_000
	});
	expect(Date.now() - start).toBeGreaterThanOrEqual(2900);
	expect(response!.fromServiceWorker()).toBe(true);
	expect(await response!.text()).toBe(shellHtml);
});

test('離線開根路徑會轉到當季首頁', async ({ page, context }) => {
	await loadAndWaitControlled(page, `${BASE}/`);
	await context.setOffline(true);
	const response = await page.goto(`${BASE}/`);
	expect(response!.fromServiceWorker()).toBe(true);
	expect(await response!.text()).toBe(shellHtml);
	await expect(page).toHaveURL(`${BASE}/${currentSeason}/men/`);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${currentSeason} 男子組`);
});

test('離線開 /matches/#d-… 轉到記住的組別的賽事頁並帶著錨點', async ({ page, context }) => {
	const matches = currentMatches('women');
	test.skip(matches.length === 0, '現行賽季女子組還沒有賽事');
	const date = matches.at(-1)!.date;
	await loadAndWaitControlled(page, `${BASE}/${currentSeason}/women/`);
	await context.setOffline(true);
	const response = await page.goto(`${BASE}/matches/#d-${date}`);
	expect(response!.fromServiceWorker()).toBe(true);
	await expect(page).toHaveURL(`${BASE}/${currentSeason}/women/matches/#d-${date}`);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${currentSeason} 女子組 賽事`);
	await expect(page.locator(`#d-${date}`)).toBeInViewport();
});

// ---- 看過就存、預先下載、離線狀態顯示（#238） ----

const updatedMonthDay = new Intl.DateTimeFormat('en-US', {
	timeZone: 'Asia/Taipei',
	month: 'numeric',
	day: 'numeric'
}).format(new Date(readJson('site').generated_at));
const READY = `可離線使用，資料更新至 ${updatedMonthDay}`;
const PREPARING = '離線資料準備中';
const BAR = `離線中，資料更新至 ${updatedMonthDay}`;

// 從建置產物列出某賽季的單場頁與球員頁路徑（與預先下載的對象同一批）
function detailPaths(season: string, kind?: 'matches' | 'players') {
	return ['men', 'women'].flatMap((group) =>
		(kind ? [kind] : (['matches', 'players'] as const)).flatMap((k) => {
			const dir = `build/${season}/${group}/${k}`;
			if (!existsSync(dir)) return [];
			return readdirSync(dir)
				.filter((id) => existsSync(`${dir}/${id}/__data.json`))
				.map((id) => `${BASE}/${season}/${group}/${k}/${id}/`);
		})
	);
}

const cachedData = (page: Page) =>
	page.evaluate(async () => (await (await caches.open('data')).keys()).map((r) => new URL(r.url).pathname));

const hasData = (page: Page, path: string) => async () => (await cachedData(page)).includes(`${path}__data.json`);

async function waitStatus(page: Page, text: string, timeout = 80_000) {
	await expect(page.getByText(text, { exact: true })).toBeVisible({ timeout });
}

async function openDetailOffline(page: Page, path: string, kind: 'matches' | 'players') {
	const response = await page.goto(path);
	expect(response!.fromServiceWorker()).toBe(true);
	const h1 = page.getByRole('heading', { level: 1 });
	await expect(h1).not.toHaveText(MISS);
	if (kind === 'matches') await expect(h1).toContainText('對');
	else await expect(h1).toHaveText(/\S/);
	await noKitErrorPage(page);
}

for (const season of [currentSeason, OLD_SEASON]) {
	test(`看過就存（整頁載入）：${season} 的單場頁與球員頁離線再開得起來`, async ({ page, context }) => {
		const match = detailPaths(season, 'matches')[0];
		const player = detailPaths(season, 'players')[0];
		test.skip(!match || !player, `${season} 沒有單場頁或球員頁`);
		await loadAndWaitControlled(page, `${BASE}/${season}/men/`);
		await page.goto(match);
		await page.goto(player);
		await expect.poll(hasData(page, match)).toBe(true);
		await expect.poll(hasData(page, player)).toBe(true);
		await context.setOffline(true);
		await openDetailOffline(page, match, 'matches');
		await openDetailOffline(page, player, 'players');
	});
}

test('看過就存（站內導覽）：由列表點進單場頁與球員頁，離線再開得起來', async ({ page, context }) => {
	test.skip(!playedSeason || !playerSeason, '沒有已完賽場次或球員');
	await loadAndWaitControlled(page, `${BASE}/${playedSeason}/men/matches/`);
	const card = page.locator('.card-link').first();
	const matchPath = (await card.getAttribute('href'))!;
	await card.click();
	await expect(page).toHaveURL(matchPath);
	await page.goto(`${BASE}/${playerSeason!.season}/men/players/`);
	const link = page.locator('a[href*="/players/"]:not([href$="/players/"])').first();
	const playerPath = (await link.getAttribute('href'))!;
	await link.click();
	await expect(page).toHaveURL(playerPath);
	await expect.poll(hasData(page, matchPath)).toBe(true);
	await expect.poll(hasData(page, playerPath)).toBe(true);
	await context.setOffline(true);
	await openDetailOffline(page, matchPath, 'matches');
	await openDetailOffline(page, playerPath, 'players');
});

test.describe('桌面寬度也會存看過的頁', () => {
	test.use({ viewport: { width: 1280, height: 800 } });

	test('單場頁與球員頁離線再開得起來', async ({ page, context }) => {
		const match = detailPaths(currentSeason, 'matches')[0];
		const player = detailPaths(currentSeason, 'players')[0];
		await loadAndWaitControlled(page, `${BASE}/${currentSeason}/men/`);
		await page.goto(match);
		await page.goto(player);
		await expect.poll(hasData(page, match)).toBe(true);
		await expect.poll(hasData(page, player)).toBe(true);
		await context.setOffline(true);
		await openDetailOffline(page, match, 'matches');
		await openDetailOffline(page, player, 'players');
	});
});

test('第一次進站直接落在單場頁：service worker 接手後補存，離線重開同一頁看得到資料', async ({ page, context }) => {
	const match = detailPaths(currentSeason, 'matches')[0];
	await loadAndWaitControlled(page, match);
	await expect.poll(hasData(page, match)).toBe(true);
	await context.setOffline(true);
	await page.reload();
	await expect(page.getByRole('heading', { level: 1 })).toContainText('對');
	await expect(page.getByRole('heading', { level: 1 })).not.toHaveText(MISS);
});

test('獨立視窗：當季預先下載完成後，離線時沒看過的單場頁與球員頁都開得起來', async ({ page, context }) => {
	test.slow();
	await asStandalone(page);
	await loadAndWaitControlled(page, `${BASE}/${currentSeason}/men/`);
	await openMoreSheet(page);
	await waitStatus(page, READY);
	const expected = detailPaths(currentSeason);
	expect(expected.length).toBeGreaterThan(0);
	const cached = await cachedData(page);
	for (const path of expected) expect(cached).toContain(`${path}__data.json`);
	await page.keyboard.press('Escape');
	await context.setOffline(true);
	await openDetailOffline(page, detailPaths(currentSeason, 'matches').at(-1)!, 'matches');
	await openDetailOffline(page, detailPaths(currentSeason, 'players').at(-1)!, 'players');
});

test('下載中顯示「離線資料準備中」，下載完變成可離線使用', async ({ page, context }) => {
	test.slow();
	let release!: () => void;
	const gate = new Promise<void>((resolve) => (release = resolve));
	await context.route('**/__data.json', async (route) => {
		await gate;
		await route.continue().catch(() => {});
	});
	await asStandalone(page);
	await loadAndWaitControlled(page, `${BASE}/${currentSeason}/men/`);
	await openMoreSheet(page);
	await waitStatus(page, PREPARING);
	await expect(page.getByText(READY, { exact: true })).toHaveCount(0);
	release();
	await waitStatus(page, READY);
});

test('瀏覽器分頁不預先下載：沒看過的當季單場頁離線時顯示說明頁', async ({ page, context }) => {
	await loadAndWaitControlled(page, `${BASE}/${currentSeason}/men/`);
	await openMoreSheet(page);
	await waitStatus(page, READY);
	const unseen = detailPaths(currentSeason, 'matches')[0];
	expect(await cachedData(page)).not.toContain(`${unseen}__data.json`);
	await page.keyboard.press('Escape');
	await context.setOffline(true);
	await page.goto(unseen);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(MISS);
});

test('離線資料被清掉：下次連網開啟自動重抓，沒有任何通知', async ({ page, context }) => {
	// 預存加當季下載要跑兩輪；整套並行又機器忙時超過預設的 30 秒
	test.slow();
	await asStandalone(page);
	await loadAndWaitControlled(page, `${BASE}/${currentSeason}/men/`);
	await openMoreSheet(page);
	await waitStatus(page, READY);
	await page.keyboard.press('Escape');
	await page.evaluate(async () => {
		for (const key of await caches.keys()) await caches.delete(key);
	});
	await page.reload();
	await expect(page.getByRole('alert')).toHaveCount(0);
	await expect(page.getByRole('dialog')).toHaveCount(0);
	await openMoreSheet(page);
	await waitStatus(page, READY);
	await page.keyboard.press('Escape');
	await context.setOffline(true);
	await openDetailOffline(page, detailPaths(currentSeason, 'matches').at(-1)!, 'matches');
	const list = await page.goto(`${BASE}/${currentSeason}/men/standings/`);
	expect(list!.fromServiceWorker()).toBe(true);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${currentSeason} 男子組 戰績`);
	await expect(page.getByRole('alert')).toHaveCount(0);
	await expect(page.getByRole('dialog')).toHaveCount(0);
});

// iPhone 的獨立視窗從背景回來不會重新載入，回到前景就是下一次「開啟」
test('獨立視窗回到前景：離線資料在背景時被清掉，回來就重抓', async ({ page }) => {
	test.slow();
	await asStandalone(page);
	await loadAndWaitControlled(page, `${BASE}/${currentSeason}/men/`);
	await openMoreSheet(page);
	await waitStatus(page, READY);
	await page.evaluate(() => caches.delete('data'));
	expect(await cachedData(page)).toEqual([]);
	await page.evaluate(() => document.dispatchEvent(new Event('visibilitychange')));
	// 清掉後狀態行還留著就緒的字，所以先等資料檔真的抓齊，再看狀態行
	const expected = detailPaths(currentSeason).map((path) => `${path}__data.json`);
	const missing = async () => {
		const cached = await cachedData(page);
		return expected.filter((url) => !cached.includes(url)).length;
	};
	await expect.poll(missing, { timeout: 80_000 }).toBe(0);
	await waitStatus(page, READY);
});

test('離線狀態細條：線上沒有、頁尾照舊；離線出現「離線中，資料更新至 M/D」且日期同頁尾', async ({ page, context }) => {
	await loadAndWaitControlled(page, `${BASE}/${currentSeason}/men/`);
	await expect(page.getByText(/^離線中/)).toHaveCount(0);
	const footer = page.getByText(/^資料更新時間/);
	await expect(footer).toBeVisible();
	const [, month, day] = (await page.locator('footer time').textContent())!.match(/-(\d+)-(\d+) /)!;
	await context.setOffline(true);
	await expect(page.getByText(BAR, { exact: true })).toBeVisible();
	expect(BAR).toBe(`離線中，資料更新至 ${Number(month)}/${Number(day)}`);
	await expect(footer).toBeVisible();
});

test('離線判斷不只看 navigator.onLine：探測失敗、站內導覽後細條出現', async ({ page, context }) => {
	await loadAndWaitControlled(page, `${BASE}/${currentSeason}/men/`);
	await expect(page.getByText(/^離線中/)).toHaveCount(0);
	await context.route('**/_app/version.json', (route) => route.abort());
	await page.getByRole('navigation', { name: '主要分頁' }).getByRole('link', { name: '戰績' }).click();
	await expect(page).toHaveURL(`${BASE}/${currentSeason}/men/standings/`);
	await expect(page.getByText(BAR, { exact: true })).toBeVisible();
	expect(await page.evaluate(() => navigator.onLine)).toBe(true);
});
