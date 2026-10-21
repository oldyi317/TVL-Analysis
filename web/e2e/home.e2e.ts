import { expect, test } from '@playwright/test';
import { BASE, currentSeason as current, findSeason, seasonsData } from './helpers.js';
import { readFileSync } from 'node:fs';

const cupId: number = seasonsData.current_cup_id;
const matchesData = JSON.parse(readFileSync('src/lib/data/matches.json', 'utf8'));
const menMatches: { date: string; sets: unknown[] }[] = matchesData.matches.find(
	(m: { cup_id: number; group: string }) => m.cup_id === cupId && m.group === 'men'
).matches;
const played = menMatches.filter((m) => m.sets.length > 0);
const latestDate = played.map((m) => m.date).sort().at(-1);
const hasUpcoming = menMatches.some((m) => m.sets.length === 0);

test('根路徑首次導向當季男子組首頁', async ({ page }) => {
	await page.goto(`${BASE}/`);
	await expect(page).toHaveURL(`${BASE}/${current}/men/`);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${current} 男子組`);
});

test('首頁三段：最新比分、戰績、下一場', async ({ page }) => {
	await page.goto(`${BASE}/${current}/men/`);

	const latest = page.getByRole('region', { name: '最新比分' });
	await expect(latest).toBeVisible();
	await expect(latest.locator('.card-link')).toHaveCount(played.filter((m) => m.date === latestDate).length);

	const standings = page.getByRole('region', { name: '戰績' });
	await expect(standings).toBeVisible();
	const rows = await standings.getByRole('row').count();
	expect(rows).toBeGreaterThan(1);
	expect(rows - 1).toBeLessThanOrEqual(4);
	await expect(standings.getByRole('link', { name: '完整戰績' })).toHaveAttribute(
		'href',
		`${BASE}/${current}/men/standings/`
	);

	const next = page.getByRole('region', { name: '下一場' });
	if (hasUpcoming) {
		await expect(next).toBeVisible();
		await expect(next.getByRole('group', { name: '賽前勝率' })).toHaveText(/\d+% – \d+%|資料不足/);
		await expect(next.getByRole('note')).toBeVisible();
	} else await expect(next).toHaveCount(0);
});

test('最新比分有「全部賽事」連結', async ({ page }) => {
	// 開季初當季沒有已完賽場次，退回最近一個有已完賽場次的賽季
	const picked = findSeason('played');
	test.skip(!picked, '所有賽季的男子組都沒有已完賽場次');
	await page.goto(`${BASE}/${picked!.season}/men/`);
	await expect(
		page.getByRole('region', { name: '最新比分' }).getByRole('link', { name: '全部賽事' })
	).toHaveAttribute('href', `${BASE}/${picked!.season}/men/matches/`);
});

test('記住上次看的組別', async ({ page }) => {
	await page.goto(`${BASE}/${current}/women/`);
	await expect.poll(() => page.evaluate(() => localStorage.getItem('tvl.group'))).toBe('women');
	await page.goto(`${BASE}/`);
	await expect(page).toHaveURL(`${BASE}/${current}/women/`);
});

test('全站導覽五個項目，點球隊到球隊頁', async ({ page }) => {
	await page.goto(`${BASE}/${current}/men/`);
	const nav = page.getByRole('navigation', { name: '頁面' });
	for (const name of ['首頁', '賽事', '戰績', '球隊', '球員']) {
		await expect(nav.getByRole('link', { name, exact: true })).toBeVisible();
	}
	await expect(nav.getByRole('link', { name: '首頁' })).toHaveAttribute('aria-current', 'page');
	await nav.getByRole('link', { name: '球隊', exact: true }).click();
	await expect(page).toHaveURL(`${BASE}/${current}/men/teams/`);
	await expect(page.getByRole('heading', { level: 1 })).toContainText('球隊');
	await expect(nav.getByRole('link', { name: '球隊', exact: true })).toHaveAttribute(
		'aria-current',
		'page'
	);
});

test('首頁沒有橫向溢位', async ({ page }) => {
	await page.goto(`${BASE}/${current}/men/`);
	const overflow = await page.evaluate(
		() => document.documentElement.scrollWidth - document.documentElement.clientWidth
	);
	expect(overflow).toBeLessThanOrEqual(0);
});
