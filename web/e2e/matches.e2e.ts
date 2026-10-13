import { readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import { BASE, currentSeason as current, seasonsData } from './helpers.js';

const URL = `${BASE}/${current}/men/matches/`;
const matchesData = JSON.parse(readFileSync('src/lib/data/matches.json', 'utf8'));
const currentCup = seasonsData.current_cup_id;
const menMatches = matchesData.matches.find(
	(g: { cup_id: number; group: string }) => g.cup_id === currentCup && g.group === 'men'
).matches;

test('列表依月份分段、每張卡是連結、無橫向捲動', async ({ page }) => {
	await page.goto(URL);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${current} 男子組 賽事`);
	const months = page.getByRole('navigation', { name: '月份' }).getByRole('link');
	expect(await months.count()).toBeGreaterThanOrEqual(2);
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
	const golden = menMatches.find((m: { golden_set: unknown }) => m.golden_set);
	test.skip(!golden, '當季男子組沒有黃金決勝局');
	await page.goto(URL);
	const card = page.locator(`.card-link[href$="/matches/${golden.game_id}/"]`);
	await expect(card).toContainText(`黃金決勝局 ${golden.golden_set[0]}-${golden.golden_set[1]}`);
	await card.click();
	await expect(page).toHaveURL(new RegExp(`/matches/${golden.game_id}/$`));
	await expect(page.getByRole('rowheader', { name: '黃金決勝局' })).toBeVisible();
});

test('點卡片進詳情：逐局比分表、返回連結', async ({ page }) => {
	await page.goto(URL);
	await page.locator('.card-link').first().click();
	await expect(page).toHaveURL(/\/matches\/\d+\/$/);
	const first = menMatches[0];
	await expect(page.getByRole('heading', { level: 1 })).toContainText('對');
	await expect(page.getByRole('rowheader', { name: '第 1 局' })).toBeVisible();
	await expect(page.getByRole('rowheader', { name: '合計' })).toBeVisible();
	expect(await page.getByRole('row').count()).toBeGreaterThanOrEqual(first.sets.length + 2);
	await page.getByRole('link', { name: '賽事列表' }).click();
	await expect(page).toHaveURL(new RegExp(`/matches/#m-${first.date.slice(0, 7)}$`));
});

test('詳情頁的組別與賽季切換退回賽事列表', async ({ page }) => {
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

test('球隊列表頁的賽事連結', async ({ page }) => {
	await page.goto(`${BASE}/${current}/men/`);
	await page.getByRole('link', { name: '賽事', exact: true }).click();
	await expect(page).toHaveURL(/\/matches\/$/);
});
