import { expect, test } from '@playwright/test';
import { BASE, currentSeason as current, findSeason, readJson, seasonsData } from './helpers.js';

// 開季初當季球隊還沒登錄齊，退回最近一個有球隊的賽季
const teamSeason = findSeason('teams');
const teamCount: number = teamSeason
	? readJson('teams').teams.filter(
			(t: { group: string; cup_ids: number[] }) =>
				t.group === 'men' && t.cup_ids.includes(teamSeason.cupId)
		).length
	: 0;

test('球隊頁列出帶隊色的圓形隊標', async ({ page }) => {
	test.skip(!teamSeason, '所有賽季的男子組都沒有球隊');
	await page.goto(`${BASE}/${teamSeason!.season}/men/teams/`);
	const rows = page.getByRole('list').getByRole('listitem');
	await expect(rows.first()).toBeVisible();
	expect(await rows.count()).toBe(teamCount);
	const badge = rows.first().locator('.badge');
	await expect(badge).toHaveCSS('border-radius', '50%');
	expect(await badge.evaluate((el) => getComputedStyle(el).backgroundColor)).not.toBe(
		'rgba(0, 0, 0, 0)'
	);
});

test('組別切換連結正確且停留在同一頁面類型', async ({ page }) => {
	await page.goto(`${BASE}/${current}/men/teams/`);
	const women = page.getByRole('navigation', { name: '組別' }).getByRole('link', { name: '女子組' });
	await expect(women).toHaveAttribute('href', `${BASE}/${current}/women/teams/`);
	await women.click();
	await expect(page).toHaveURL(`${BASE}/${current}/women/teams/`);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${current} 女子組 球隊`);
	await expect(
		page.getByRole('navigation', { name: '組別' }).getByRole('link', { name: '女子組' })
	).toHaveAttribute('aria-current', 'page');
});

test('賽季連結指向同組別的另一季', async ({ page }) => {
	await page.goto(`${BASE}/${current}/women/teams/`);
	const other = seasonsData.seasons.find((s: { name: string }) => s.name !== current);
	await expect(
		page.getByRole('navigation', { name: '賽季' }).getByRole('link', { name: other.name })
	).toHaveAttribute('href', `${BASE}/${other.name}/women/teams/`);
});
