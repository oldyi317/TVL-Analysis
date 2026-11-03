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
			await page.screenshot({ path: `/tmp/tvl109-${group}-${width}.png` });
		}
	}
});
