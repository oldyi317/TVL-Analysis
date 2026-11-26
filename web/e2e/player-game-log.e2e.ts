import { BASE, expect, test } from './helpers.js';

// 三場都有接發球、防守與舉球次數，可與單場頁逐欄比對到位／總數。
const samples = [
	{ season: '2026-27', group: 'men', player: 197, name: '周胤政', game: 336 },
	{ season: '2026-27', group: 'women', player: 129, name: '江辰翊', game: 288 },
	{ season: '2024-25', group: 'women', player: 88, name: '陳潔', game: 227 }
];

test('日期連結在窄版與寬版的可及名稱都以畫面上的日期開頭', async ({ page }) => {
	await page.goto(`${BASE}/2026-27/women/players/129/`);
	const link = page.getByRole('table', { name: '逐場紀錄' }).locator('a[href$="/matches/288/"]');
	await page.setViewportSize({ width: 375, height: 812 });
	await expect(link).toHaveAccessibleName(/^11\/15\s+對臺北鯨華的單場頁$/);
	await page.setViewportSize({ width: 1440, height: 900 });
	await expect(link).toHaveAccessibleName(/^2026\/11\/15\s+對臺北鯨華的單場頁$/);
});

test('375px 對手隊名完整顯示在同一行', async ({ page }) => {
	await page.setViewportSize({ width: 375, height: 812 });
	await page.goto(`${BASE}/2026-27/women/players/129/`);
	const table = page.getByRole('table', { name: '逐場紀錄' });
	const team = table.getByRole('cell').filter({ hasText: '臺北鯨華' }).first();
	await expect(team).toBeVisible();
	const lines = await team.evaluate((cell) => {
		const walker = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT);
		while (walker.nextNode()) {
			if (walker.currentNode.textContent?.trim() !== '臺北鯨華') continue;
			const range = document.createRange();
			range.selectNodeContents(walker.currentNode);
			return range.getClientRects().length;
		}
		return 0;
	});
	expect(lines).toBe(1);
});

test('中間寬度可讀主客場、總決賽與黃金決勝局標記，整頁無橫向捲動', async ({ page }) => {
	await page.goto(`${BASE}/2025-26/men/players/10/`);
	const table = page.getByRole('table', { name: '逐場紀錄' });
	const row = table.getByRole('row').filter({ has: page.locator('a[href$="/matches/327/"]') });
	for (const width of [768, 900, 1024]) {
		await page.setViewportSize({ width, height: 900 });
		for (const tag of ['主場', '總決賽', '黃金決勝局']) {
			await expect(row.getByText(tag, { exact: true })).toBeVisible();
		}
		const sizes = await page.evaluate(() => [document.documentElement.clientWidth, document.documentElement.scrollWidth]);
		expect(sizes, `${width}px`).toEqual([width, width]);
	}
});

for (const { season, group, player, name, game } of samples) {
	test(`${season} ${name}的逐場紀錄可展開、連到單場頁，九項數字一致`, async ({ page }) => {
		const matchUrl = `${BASE}/${season}/${group}/matches/${game}/`;
		await page.setViewportSize({ width: 375, height: 812 });
		await page.goto(`${BASE}/${season}/${group}/players/${player}/`);
		const table = page.getByRole('table', { name: '逐場紀錄' });
		const row = table.getByRole('row').filter({ has: page.locator(`a[href="${matchUrl}"]`) });
		const dates = await table.locator('time').evaluateAll((times) => times.map((t) => t.getAttribute('datetime')));
		expect(dates.length).toBeGreaterThan(1);
		expect(dates).toEqual([...dates].sort().reverse());
		await expect(table.getByRole('columnheader')).toHaveText([
			'日期', '對手', '勝敗', '得分', '攻擊', '攔網', '發球', '詳細'
		]);
		const toggle = row.getByRole('button');
		const detail = page.locator(`[id="${await toggle.getAttribute('aria-controls')}"]`);
		await toggle.click();
		await expect(toggle).toHaveAttribute('aria-expanded', 'true');
		await expect(detail.locator('dt')).toHaveText(['攻擊次數', '發球次數', '接發球', '防守', '舉球']);
		const expandedValues = await detail.locator('dd').allTextContents();
		expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(375);
		await toggle.click();
		await expect(detail).toBeHidden();

		await page.setViewportSize({ width: 1440, height: 900 });
		await expect(table.getByRole('columnheader')).toHaveCount(12);
		const numbers = (await row.getByRole('cell').allTextContents()).slice(2);
		expect(numbers).toHaveLength(9);
		expect(numbers.slice(4)).toEqual(expandedValues);
		await row.click();
		await expect(page).toHaveURL(matchUrl);
		const playerRow = page.getByRole('table', { name: /單場數據$/ }).getByRole('row')
			.filter({ has: page.getByText(name, { exact: true }) });
		expect((await playerRow.getByRole('cell').allTextContents()).slice(2)).toEqual(numbers);
	});
}
