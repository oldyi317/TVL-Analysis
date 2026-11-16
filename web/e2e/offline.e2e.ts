import { readFileSync } from 'node:fs';
import { type Page } from '@playwright/test';
import { BASE, OLD_SEASON, currentSeason, expect, findSeason, readJson, test } from './helpers.js';

test.use({ serviceWorkers: 'allow' });

const shellHtml = readFileSync('build/shell.html', 'utf8');
const { version } = JSON.parse(readFileSync('build/_app/version.json', 'utf8'));
const MISS = '這一頁離線時看不到，連上網路後再試';
const playedSeason = findSeason('played')?.season;
const playerSeason = findSeason('players');

// 線上載入一頁，等 service worker 預存完、啟用並接手
async function loadAndWaitControlled(page: Page, url: string) {
	await page.goto(url);
	await page.waitForFunction(() => navigator.serviceWorker.controller !== null);
}

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
