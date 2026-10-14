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

test('資料不完整的組別顯示註記', async ({ page }) => {
	await page.goto(`${BASE}/2024-25/women/standings/`);
	const note = page.getByRole('note');
	await expect(note).toBeVisible();
	await expect(note).toContainText('不完整');
});
