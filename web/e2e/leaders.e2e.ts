import { expect, test } from '@playwright/test';
import { BASE, readJson, seasonsData } from './helpers.js';

const entries = readJson('leaders').leaders;
const seasonName = (cup: number) => seasonsData.seasons.find((s: { cup_id: number }) => s.cup_id === cup).name;
const entry = entries.find((e: { group: string }) => e.group === 'women');
const url = `${BASE}/${seasonName(entry.cup_id)}/women/leaders/`;

test('七項跳轉、吸頂 chip 與捲動的當前項目', async ({ page }) => {
	await page.goto(url);
	const nav = page.getByRole('navigation', { name: '計數榜項目' });
	await expect(nav.getByRole('link')).toHaveCount(7);
	await page.setViewportSize({ width: 1440, height: 900 });
	for (const index of [0, 1]) {
		const chip = nav.getByRole('link').nth(index);
		await chip.click();
		await expect(chip).toHaveAttribute('aria-current', 'location');
	}
	const last = nav.getByRole('link').last();
	await last.click();
	await expect(last).toHaveAttribute('aria-current', 'location');
	await expect(page.locator('#b-set_excellent h2')).toBeInViewport();
	const bottom = (await nav.boundingBox())!.y + (await nav.boundingBox())!.height;
	expect((await page.locator('#b-set_excellent h2').boundingBox())!.y).toBeGreaterThanOrEqual(bottom);
	await page.evaluate(() => window.scrollTo(0, 0));
	await expect(nav.getByRole('link').first()).toHaveAttribute('aria-current', 'location');
});

test('缺單場數據時沒有四王卡或項目 chip', async ({ page }) => {
	const empty = seasonsData.seasons.flatMap((s: { cup_id: number; name: string }) =>
		['men', 'women'].map((group) => ({ ...s, group }))
	).find((s: { cup_id: number; group: string }) => !entries.some((e: { cup_id: number; group: string }) => e.cup_id === s.cup_id && e.group === s.group));
	test.skip(!empty, '每季每組皆有單場數據');
	await page.goto(`${BASE}/${empty.name}/${empty.group}/leaders/`);
	await expect(page.getByText('本季尚無單場數據')).toBeVisible();
	await expect(page.getByRole('region', { name: '本季四王' })).toHaveCount(0);
	await expect(page.getByRole('navigation', { name: '計數榜項目' })).toHaveCount(0);
});

test('男女組長姓名、四王首屏、深淺模式與桌面欄數', async ({ page }) => {
	for (const group of ['men', 'women']) {
		const data = entries.filter((e: { group: string }) => e.group === group)
			.sort((a: { cup_id: number }, b: { cup_id: number }) => b.cup_id - a.cup_id)[0];
		const longest = data.boards[0].rows.reduce((a: { name: string }, b: { name: string }) => a.name.length >= b.name.length ? a : b);
		for (const width of [390, 768, 1440]) {
			await page.setViewportSize({ width, height: 900 });
			await page.emulateMedia({ colorScheme: width === 768 ? 'dark' : 'light' });
			await page.goto(`${BASE}/${seasonName(data.cup_id)}/${group}/leaders/`);
			const kings = page.getByRole('region', { name: '本季四王' });
			await expect(kings).toBeInViewport();
			const kingBox = (await kings.boundingBox())!;
			expect(kingBox.y + kingBox.height).toBeLessThanOrEqual(900);
			const first = page.locator('#b-total_points');
			const leaderHeight = (await first.locator('tbody tr').first().boundingBox())!.height;
			expect(leaderHeight).toBeLessThanOrEqual(49);
			const expand = first.getByRole('button');
			if (await expand.count()) await expand.click();
			await expect(first.getByRole('link', { name: longest.name, exact: true })).toBeVisible();
			expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(0);
			const columns = await page.locator('.boards').evaluate((el) => getComputedStyle(el).gridTemplateColumns.split(' ').length);
			expect(columns).toBe(width < 640 ? 1 : width < 1024 ? 2 : 3);
			await page.evaluate(() => window.scrollTo(0, 0));
		}
	}
});

test('鍵盤切分段、chip 與展開，切換不重新載入', async ({ page }) => {
	await page.goto(url);
	let navigations = 0;
	page.on('framenavigated', (frame) => { if (frame === page.mainFrame()) navigations++; });
	const rate = page.getByRole('button', { name: '比率榜', exact: true });
	await expect(async () => {
		await rate.focus();
		await page.keyboard.press('Enter');
		await expect(rate).toHaveAttribute('aria-pressed', 'true', { timeout: 500 });
	}).toPass({ timeout: 10_000 });
	await expect(page.locator('#b-total_points')).toHaveCount(0);
	const nav = page.getByRole('navigation', { name: '比率榜項目' });
	const rates = entry.boards.filter((b: { kind: string }) => b.kind === 'rate');
	await expect(nav.getByRole('link')).toHaveCount(rates.length);
	for (const board of rates) {
		const card = page.locator(`#b-${board.key}`);
		await expect(card.getByRole('heading', { name: board.label, exact: true })).toBeAttached();
		await expect(card.getByText(board.threshold, { exact: true })).toBeAttached();
		if (board.rows.length) {
			const row = board.rows[0];
			await expect(card.getByText(`${row.numerator}／${row.denominator}`, { exact: true }).first()).toBeAttached();
		} else await expect(card.getByText('尚無球員達門檻')).toBeAttached();
	}
	const expandable = rates.find((b: { rows: { rank: number }[] }) => b.rows.length > 10 && b.rows.at(-1)!.rank > b.rows[9].rank);
	if (expandable) {
		const chip = nav.getByRole('link', { name: expandable.label, exact: true });
		await chip.focus();
		await page.keyboard.press('Enter');
		await expect(page.locator(`#b-${expandable.key} h2`)).toBeInViewport();
		const button = page.locator(`#b-${expandable.key}`).getByRole('button');
		await button.focus();
		await page.keyboard.press('Enter');
		await expect(button).toHaveAttribute('aria-expanded', 'true');
		await page.keyboard.press('Enter');
		await expect(button).toHaveAttribute('aria-expanded', 'false');
	}
	// 錨點會觸發同文件導航；分段本身應維持相同 document。
	const before = navigations;
	await page.getByRole('button', { name: '計數榜', exact: true }).click();
	await expect(page.locator('#b-total_points')).toBeVisible();
	expect(navigations).toBe(before);
});

test('比率榜錨點進站與歷史返回會選對分段並捲到卡片', async ({ page }) => {
	await page.goto(`${url}#b-serve_point_rate`);
	await expect(page.getByRole('button', { name: '比率榜', exact: true })).toHaveAttribute('aria-pressed', 'true');
	await expect(page.locator('#b-serve_point_rate h2')).toBeInViewport();
	await page.getByRole('button', { name: '計數榜', exact: true }).click();
	await page.getByRole('navigation', { name: '計數榜項目' }).getByRole('link', { name: '總得分', exact: true }).click();
	await page.goBack();
	await expect(page.getByRole('button', { name: '比率榜', exact: true })).toHaveAttribute('aria-pressed', 'true');
	await expect(page.locator('#b-serve_point_rate h2')).toBeInViewport();
});

test('無局數的賽季缺每局攔網，空榜仍保留 chip', async ({ page }) => {
	const noSets = entries.find((e: typeof entry) => e.boards.some((b: { kind: string }) => b.kind === 'rate') && !e.boards.some((b: { key: string }) => b.key === 'blocks_per_set'));
	expect(noSets).toBeTruthy();
	await page.goto(`${BASE}/${seasonName(noSets.cup_id)}/${noSets.group}/leaders/#b-points_per_game`);
	await expect(page.getByRole('navigation', { name: '比率榜項目' }).getByRole('link')).toHaveCount(6);
	await expect(page.locator('#b-blocks_per_set')).toHaveCount(0);
	await expect(page.getByRole('link', { name: '每局攔網', exact: true })).toHaveCount(0);
	await expect(page.locator('#b-points_per_game').getByText('尚無球員達門檻')).toBeVisible();
	await expect(page.getByRole('navigation', { name: '比率榜項目' }).getByRole('link', { name: '場均得分', exact: true })).toBeVisible();
});
