import { readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';

const BASE = '/TVL-Analysis';
const URL = `${BASE}/styleguide/`;
const teamNames: string[] = JSON.parse(readFileSync('src/lib/data/teams.json', 'utf8')).teams.map(
	(t: { name: string }) => t.name
);

test.describe('淺色', () => {
	test.use({ colorScheme: 'light' });

	test('標題、無橫向捲動、正文字級', async ({ page }) => {
		await page.goto(URL);
		await expect(page.getByRole('heading', { level: 1 })).toHaveText('樣式指南');
		const overflow = await page.evaluate(
			() => document.documentElement.scrollWidth - document.documentElement.clientWidth
		);
		expect(overflow).toBeLessThanOrEqual(0);
		const size = await page.evaluate(() => parseFloat(getComputedStyle(document.body).fontSize));
		expect(size).toBeGreaterThanOrEqual(16);
	});

	test('背景為淺色 token', async ({ page }) => {
		await page.goto(URL);
		await expect(page.locator('body')).toHaveCSS('background-color', 'rgb(243, 244, 249)');
	});

	test('戰績表第一列可展開', async ({ page }) => {
		await page.goto(URL);
		const button = page.locator('tbody tr.main button').first();
		await expect(button).toHaveAttribute('aria-expanded', 'false');
		await expect(async () => {
			await button.click();
			await expect(button).toHaveAttribute('aria-expanded', 'true', { timeout: 500 });
		}).toPass({ timeout: 10_000 });
		const detail = page.locator(`#${await button.getAttribute('aria-controls')}`);
		await expect(detail.getByText('近五場')).toBeVisible();
	});

	test('五局逐局比分在手機排成一行', async ({ page }) => {
		await page.goto(URL);
		const card = page.locator('article', { has: page.locator('.golden') });
		const tops = await card.locator('.set').evaluateAll((els) =>
			els.map((el) => el.getBoundingClientRect().top)
		);
		expect(tops).toHaveLength(5);
		expect(new Set(tops).size).toBe(1);
	});

	test('隊標文字不超出圓形', async ({ page }) => {
		await page.goto(URL);
		const overflowing = await page.locator('.badge').evaluateAll((badges) =>
			badges
				.filter((badge) =>
					[...badge.children].some(
						(child) => child.getBoundingClientRect().width > badge.getBoundingClientRect().width
					)
				)
				.map((badge) => badge.textContent)
		);
		expect(overflowing).toEqual([]);
	});

	test('圖表容器存在', async ({ page }) => {
		await page.goto(URL);
		await expect(page.getByRole('img', { name: /趨勢/ })).toBeVisible();
	});

	test('十四隊名稱都在隊色區塊', async ({ page }) => {
		await page.goto(URL);
		expect(teamNames).toHaveLength(14);
		const colors = page.locator('#colors');
		for (const name of teamNames) {
			await expect(colors.getByText(name, { exact: true }).first()).toBeVisible();
		}
	});
});

test.describe('深色', () => {
	test.use({ colorScheme: 'dark' });

	test('背景為深色 token', async ({ page }) => {
		await page.goto(URL);
		await expect(page.locator('body')).toHaveCSS('background-color', 'rgb(18, 20, 31)');
	});
});

test('未開打的賽事卡標示未開打並保留開賽時間', async ({ page }) => {
	await page.goto(`${BASE}/styleguide/`);
	const card = page.locator('article', { hasText: '未開打' });
	await expect(card).toContainText('19:00');
	await expect(card.locator('.score')).toHaveCount(0);
});
