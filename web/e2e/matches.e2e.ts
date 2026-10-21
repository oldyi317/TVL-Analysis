import { expect, test } from '@playwright/test';
import { BASE, currentSeason, findSeason, readJson, seasonsData } from './helpers.js';

// 開季初當季還沒有已完賽場次，退回最近一個有已完賽場次的賽季
const picked = findSeason('played');
const current = picked?.season ?? currentSeason;
const URL = `${BASE}/${current}/men/matches/`;
const menMatches: { date: string; game_id: number; sets: unknown[]; golden_set: number[] | null }[] =
	picked
		? readJson('matches').matches.find(
				(g: { cup_id: number; group: string }) => g.cup_id === picked.cupId && g.group === 'men'
			).matches
		: [];
const noPlayed = '所有賽季的男子組都沒有已完賽場次';
const monthCount = new Set(menMatches.map((m) => m.date.slice(0, 7))).size;

test('列表依月份分段、每張卡是連結、無橫向捲動', async ({ page }) => {
	test.skip(!picked, noPlayed);
	await page.goto(URL);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${current} 男子組 賽事`);
	const months = page.getByRole('navigation', { name: '月份' }).getByRole('link');
	expect(await months.count()).toBe(monthCount);
	const first = menMatches[0];
	await expect(page.getByRole('heading', { level: 2 }).first()).toHaveText(
		`${first.date.slice(0, 4)} 年 ${Number(first.date.slice(5, 7))} 月`
	);
	const cards = page.locator('.card-link');
	expect(await cards.count()).toBe(menMatches.length);
	await expect(cards.first()).toContainText(first.date);

	const overflow = await page.evaluate(
		() => document.documentElement.scrollWidth - document.documentElement.clientWidth
	);
	expect(overflow).toBeLessThanOrEqual(0);
});

test('月份跳轉：點最後一個月，該段標題進入畫面且 chip 標為當前', async ({ page }) => {
	test.skip(!picked, noPlayed);
	await page.goto(URL);
	const nav = page.getByRole('navigation', { name: '月份' });
	const last = nav.getByRole('link').last();
	const key = (await last.getAttribute('href'))!.replace('#m-', '');
	await last.click();
	await expect(page).toHaveURL(new RegExp(`#m-${key}$`));
	await expect(last).toHaveAttribute('aria-current', 'location');
	const heading = page.locator(`#h-${key}`);
	await expect(heading).toBeInViewport();
	const navBottom = (await nav.boundingBox())!.y + (await nav.boundingBox())!.height;
	expect((await heading.boundingBox())!.y).toBeGreaterThanOrEqual(navBottom);
});

test('黃金決勝局在卡片與詳情都標明', async ({ page }) => {
	test.skip(!picked, noPlayed);
	const golden = menMatches.find((m) => m.golden_set);
	test.skip(!golden, '用於比對的賽季男子組沒有黃金決勝局');
	await page.goto(URL);
	const card = page.locator(`.card-link[href$="/matches/${golden!.game_id}/"]`);
	await expect(card).toContainText(`黃金決勝局 ${golden!.golden_set![0]}-${golden!.golden_set![1]}`);
	await card.click();
	await expect(page).toHaveURL(new RegExp(`/matches/${golden!.game_id}/$`));
	await expect(page.getByRole('rowheader', { name: '黃金決勝局' })).toBeVisible();
});

test('點卡片進詳情：逐局比分表、返回連結', async ({ page }) => {
	test.skip(!picked, noPlayed);
	const first = menMatches.find((m) => m.sets.length > 0)!;
	await page.goto(URL);
	await page.locator(`.card-link[href$="/matches/${first.game_id}/"]`).click();
	await expect(page).toHaveURL(/\/matches\/\d+\/$/);
	await expect(page.getByRole('heading', { level: 1 })).toContainText('對');
	await expect(page.getByRole('rowheader', { name: '第 1 局' })).toBeVisible();
	await expect(page.getByRole('rowheader', { name: '合計' })).toBeVisible();
	expect(await page.getByRole('row').count()).toBeGreaterThanOrEqual(first.sets.length + 2);
	await page.getByRole('link', { name: '賽事列表' }).click();
	await expect(page).toHaveURL(new RegExp(`/matches/#m-${first.date.slice(0, 7)}$`));
});

test('詳情頁的組別與賽季切換退回賽事列表', async ({ page }) => {
	test.skip(!picked, noPlayed);
	const first = menMatches[0];
	await page.goto(`${URL}${first.game_id}/`);
	await expect(
		page.getByRole('navigation', { name: '組別' }).getByRole('link', { name: '女子組' })
	).toHaveAttribute('href', `${BASE}/${current}/women/matches/`);
	const other = seasonsData.seasons.find((s: { name: string }) => s.name !== current);
	await expect(
		page.getByRole('navigation', { name: '賽季' }).getByRole('link', { name: other.name })
	).toHaveAttribute('href', `${BASE}/${other.name}/men/matches/`);
});

test('歷史賽季賽事卡與賽事頁顯示當季隊名', async ({ page }) => {
	await page.goto(`${BASE}/2025-26/men/matches/`);
	const main = page.getByRole('main');
	await expect(main).toContainText('臺北國北獅');
	await expect(main).not.toContainText('臺北鳳凰');
	await page.locator('.card-link').filter({ hasText: '臺北國北獅' }).first().click();
	await expect(page).toHaveURL(/\/2025-26\/men\/matches\/\d+\/$/);
	await expect(page.getByRole('heading', { level: 1 })).toContainText('臺北國北獅');
	await expect(page).toHaveTitle(/臺北國北獅/);
});
