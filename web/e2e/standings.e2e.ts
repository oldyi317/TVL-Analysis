import { expect, test } from '@playwright/test';
import { BASE, currentSeason as current } from './helpers.js';

const URL = `${BASE}/${current}/men/standings/`;

test.describe('手機', () => {
	test('精簡五欄、可展開、無橫向捲動', async ({ page }) => {
		await page.goto(URL);
		await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${current} 男子組 戰績`);

		const headers = page.getByRole('columnheader').filter({ visible: true });
		await expect(headers).toHaveText(['名次', '隊伍', '勝', '敗', '積分', '詳細']);
		await expect(page.getByRole('columnheader', { name: '近五場' })).toBeHidden();

		const button = page.getByRole('button', { name: /詳細數據/ }).first();
		const detailId = await button.getAttribute('aria-controls');
		await expect(async () => {
			await button.click();
			await expect(button).toHaveAttribute('aria-expanded', 'true', { timeout: 500 });
		}).toPass({ timeout: 10_000 });
		const detail = page.locator(`#${detailId}`);
		await expect(detail).toBeVisible();
		await expect(detail).toContainText('近五場');
		await expect(detail).toContainText('勝率');

		const overflow = await page.evaluate(
			() => document.documentElement.scrollWidth - document.documentElement.clientWidth
		);
		expect(overflow).toBeLessThanOrEqual(0);
	});
});

test.describe('桌機', () => {
	test.use({ viewport: { width: 1280, height: 800 } });

	test('完整欄位且沒有展開按鈕', async ({ page }) => {
		await page.goto(URL);
		for (const name of ['勝率', '勝局', '敗局', '得分', '失分', '3-0', '3-1', '3-2', '2-3', '1-3', '0-3', '近五場']) {
			await expect(page.getByRole('columnheader', { name, exact: true })).toBeVisible();
		}
		await expect(page.getByRole('button', { name: /詳細數據/ }).first()).toBeHidden();
	});
});

test('2024-25 戰績已補齊，不再顯示不完整註記', async ({ page }) => {
	await page.goto(`${BASE}/2024-25/women/standings/`);
	await expect(page.getByRole('main')).toContainText('臺北鯨華');
	await expect(page.getByRole('note').filter({ hasText: '不完整' })).toHaveCount(0);
});

test('歷史賽季戰績表顯示當季隊名', async ({ page }) => {
	await page.goto(`${BASE}/2025-26/men/standings/`);
	const main = page.getByRole('main');
	await expect(main).toContainText('臺北國北獅');
	await expect(main).not.toContainText('臺北鳳凰');
});

test('歷史賽季戰績表顯示當季隊徽', async ({ page }) => {
	await page.goto(`${BASE}/2025-26/men/standings/`);
	const logo = page.locator('tr', { hasText: '臺北國北獅' }).locator('.badge img').first();
	await expect(logo).toHaveAttribute('src', /men-4-21\.png$/);
	expect(await logo.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0);
});
