import { expect, test } from '@playwright/test';
import { BASE, currentSeason as current, seasonsData } from './helpers.js';

test('根路徑導向當季男子組', async ({ page }) => {
	await page.goto(`${BASE}/`);
	await expect(page).toHaveURL(`${BASE}/${current}/men/`);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${current} 男子組 球隊`);
});

test('球隊頁列出帶隊色的圓形隊標', async ({ page }) => {
	await page.goto(`${BASE}/${current}/men/`);
	const rows = page.getByRole('list').getByRole('listitem');
	await expect(rows.first()).toBeVisible();
	expect(await rows.count()).toBeGreaterThanOrEqual(4);
	const badge = rows.first().locator('.badge');
	await expect(badge).toHaveCSS('border-radius', '50%');
	expect(await badge.evaluate((el) => getComputedStyle(el).backgroundColor)).not.toBe(
		'rgba(0, 0, 0, 0)'
	);
});

test('組別切換連結正確且停留在同一頁面類型', async ({ page }) => {
	await page.goto(`${BASE}/${current}/men/`);
	const women = page.getByRole('navigation', { name: '組別' }).getByRole('link', { name: '女子組' });
	await expect(women).toHaveAttribute('href', `${BASE}/${current}/women/`);
	await women.click();
	await expect(page).toHaveURL(`${BASE}/${current}/women/`);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${current} 女子組 球隊`);
	await expect(
		page.getByRole('navigation', { name: '組別' }).getByRole('link', { name: '女子組' })
	).toHaveAttribute('aria-current', 'page');
});

test('賽季連結指向同組別的另一季', async ({ page }) => {
	await page.goto(`${BASE}/${current}/women/`);
	const other = seasonsData.seasons.find((s: { name: string }) => s.name !== current);
	await expect(
		page.getByRole('navigation', { name: '賽季' }).getByRole('link', { name: other.name })
	).toHaveAttribute('href', `${BASE}/${other.name}/women/`);
});
