import { readFileSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';
import { BASE, currentSeason as current, readJson, seasonsData, topBar } from './helpers.js';
import { isPlayed } from '../src/lib/match.js';

const cupId: number = seasonsData.current_cup_id;
const matchesData = JSON.parse(readFileSync('src/lib/data/matches.json', 'utf8'));
const menMatches: { date: string; game_id: number; home_team_id: number; sets: unknown[] }[] = matchesData.matches.find(
	(m: { cup_id: number; group: string }) => m.cup_id === cupId && m.group === 'men'
).matches;
const game = menMatches.find(isPlayed);
const month = game?.date.slice(0, 7);
const player = readJson('players').players.find(
	(p: { group: string; seasons: { cup_id: number }[] }) =>
		p.group === 'men' && p.seasons.some((s) => s.cup_id === cupId)
);

const group = `${BASE}/${current}/men`;
const gameUrl = `${group}/matches/${game?.game_id}/`;
const teamUrl = `${group}/teams/${game?.home_team_id}/`;
const playerUrl = `${group}/players/${player?.player_id}/`;

test.beforeEach(() => {
	test.skip(!game || !player, '本季男子組還沒有已完賽場次或報名球員');
});

const backButton = (page: Page) => topBar(page).getByRole('link', { name: '返回' });
const brand = (page: Page) => topBar(page).getByRole('link', { name: 'TVL 數據' });

// 進頁後第一個互動可能早於 hydration；等到載入結束再點，才是站內導頁
async function open(page: Page, url: string) {
	await page.goto(url);
	await page.waitForLoadState('networkidle');
}

test.describe('手機頂部列左側', () => {
	for (const [name, url] of [
		['單場頁', gameUrl],
		['球隊頁', teamUrl],
		['球員頁', playerUrl]
	]) {
		test(`${name}是返回鈕，沒有站名，觸控目標至少 44px`, async ({ page }) => {
			await page.goto(url);
			await expect(backButton(page)).toBeVisible();
			await expect(brand(page)).toHaveCount(0);
			const box = (await backButton(page).boundingBox())!;
			expect(box.width).toBeGreaterThanOrEqual(44);
			expect(box.height).toBeGreaterThanOrEqual(44);
		});
	}

	for (const [name, url] of [
		['賽季首頁', `${group}/`],
		['賽事列表', `${group}/matches/`],
		['球隊列表', `${group}/teams/`],
		['歷史紀錄', `${BASE}/records/men/`]
	]) {
		test(`${name}是站名，沒有返回鈕`, async ({ page }) => {
			await page.goto(url);
			await expect(brand(page)).toBeVisible();
			await expect(backButton(page)).toHaveCount(0);
		});
	}
});

test.describe('返回鈕去哪裡', () => {
	test('從賽事列表點進單場頁，返回到賽事列表', async ({ page }) => {
		await open(page, `${group}/matches/`);
		await page.locator(`.card-link[href$="/matches/${game?.game_id}/"]`).click();
		await expect(page).toHaveURL(gameUrl);
		await backButton(page).click();
		await expect(page).toHaveURL(`${group}/matches/`);
	});

	test('從球隊頁點進單場頁，返回到那個球隊頁', async ({ page }) => {
		await open(page, teamUrl);
		await page.locator(`.card-link[href$="/matches/${game?.game_id}/"]`).click();
		await expect(page).toHaveURL(gameUrl);
		await backButton(page).click();
		await expect(page).toHaveURL(teamUrl);
	});

	for (const [name, url, target] of [
		['單場頁到賽事列表的該月份', gameUrl, `${group}/matches/#m-${month}`],
		['球隊頁到球隊列表', teamUrl, `${group}/teams/`],
		['球員頁到球員列表', playerUrl, `${group}/players/`]
	]) {
		// 沒有站內上一頁時返回鈕走原生 href，hydration 前後點擊都到同一處，不必重試
		test(`直接開啟${name}`, async ({ page }) => {
			await page.goto(url);
			await backButton(page).click();
			await expect(page).toHaveURL(target);
		});
	}

	test('單場頁重新整理後返回到賽事列表的該月份', async ({ page }) => {
		await open(page, `${group}/matches/`);
		await page.locator(`.card-link[href$="/matches/${game?.game_id}/"]`).click();
		await expect(page).toHaveURL(gameUrl);
		await page.reload();
		await backButton(page).click();
		await expect(page).toHaveURL(`${group}/matches/#m-${month}`);
	});
});

test.describe('內文的列表連結', () => {
	const links: [string, string, string][] = [
		['賽事列表', gameUrl, `${group}/matches/#m-${month}`],
		['球隊列表', teamUrl, `${group}/teams/`],
		['球員列表', playerUrl, `${group}/players/`]
	];

	for (const [name, url] of links) {
		test(`手機看不到「${name}」`, async ({ page }) => {
			await page.goto(url);
			await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
			await expect(page.getByRole('link', { name, exact: true })).toHaveCount(0);
		});
	}

	for (const [name, url, href] of links) {
		test(`1280 寬看得到「${name}」，連到 ${href}`, async ({ page }) => {
			await page.setViewportSize({ width: 1280, height: 900 });
			await page.goto(url);
			const link = page.getByRole('link', { name, exact: true });
			await expect(link).toBeVisible();
			await expect(link).toHaveAttribute('href', href);
		});
	}
});
