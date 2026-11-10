import { expect, test, type Page } from '@playwright/test';
import { BASE, currentSeason, openMoreSheet, readJson } from './helpers.js';

const entries = readJson('records').records;
const entry = (group: string) => entries.find((e: { group: string }) => e.group === group);
const men = entry('men');
const coverageText = (c: { first_season: string; season_count: number }) =>
	`資料自 ${c.first_season} 起，共 ${c.season_count} 季`;

test('男子組歷史紀錄頁：標題、涵蓋範圍、四個得分項與附註', async ({ page }) => {
	await page.goto(`${BASE}/records/men/`);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText('男子組 歷史紀錄');
	await expect(page.getByText(coverageText(men.coverage))).toBeVisible();
	await expect(page.getByRole('heading', { level: 2, name: '單季紀錄' })).toBeVisible();
	for (const board of men.boards) {
		const card = page.locator(`#r-${board.key}`);
		await expect(card.getByRole('heading', { name: board.label, exact: true })).toBeVisible();
		const table = card.getByRole('table', { name: `${board.label}單季紀錄` });
		await expect(table.locator('tbody tr')).toHaveCount(board.season_records.length);
		if (board.season_records.length) {
			const first = board.season_records[0];
			await expect(table.getByRole('link', { name: first.name }).first()).toHaveAttribute(
				'href', /\/players\/\d+\/$/
			);
			await expect(table.getByText(`${first.season} · ${first.team_short} · 出賽 ${first.games} 場`).first()).toBeVisible();
		}
	}
	await expect(page.getByRole('list', { name: '附註' })).toContainText('出賽場數以有技術統計的場次計');
	await expect(page.getByRole('list', { name: '附註' })).toContainText('不含黃金決勝局');
});

test('男子組累計紀錄頁：四個得分項、列說明、連到最近一季球員頁', async ({ page }) => {
	await page.goto(`${BASE}/records/men/cumulative/`);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText('男子組 歷史紀錄');
	await expect(page.getByRole('heading', { level: 2, name: '累計紀錄' })).toBeVisible();
	const kinds = page.getByRole('navigation', { name: '紀錄類型' });
	await expect(kinds.getByRole('link', { name: '累計紀錄' })).toHaveAttribute('aria-current', 'page');
	await expect(kinds.getByRole('link', { name: '單季紀錄' })).not.toHaveAttribute('aria-current', 'page');
	for (const board of men.boards) {
		const table = page.locator(`#r-${board.key}`).getByRole('table', { name: `${board.label}累計紀錄` });
		await expect(table.locator('tbody tr')).toHaveCount(board.cumulative_records.length);
		if (board.cumulative_records.length) {
			const first = board.cumulative_records[0];
			await expect(table.getByRole('link', { name: first.name }).first()).toHaveAttribute(
				'href', new RegExp(`/${first.season}/men/players/${first.player_id}/$`)
			);
			await expect(
				table.getByText(`${first.season_count} 季 · 出賽 ${first.games} 場 · 最近一季 ${first.team_short}`).first()
			).toBeVisible();
		}
	}
});

test('男子組單場紀錄頁：列說明、連到紀錄所在賽季的球員頁與賽事頁', async ({ page }) => {
	await page.goto(`${BASE}/records/men/match/`);
	await expect(page.getByRole('heading', { level: 2, name: '單場紀錄' })).toBeVisible();
	await expect(
		page.getByRole('navigation', { name: '紀錄類型' }).getByRole('link', { name: '單場紀錄' })
	).toHaveAttribute('aria-current', 'page');
	for (const board of men.boards) {
		const table = page.locator(`#r-${board.key}`).getByRole('table', { name: `${board.label}單場紀錄` });
		await expect(table.locator('tbody tr')).toHaveCount(board.match_records.length);
		if (!board.match_records.length) continue;
		const first = board.match_records[0];
		const row = table.locator('tbody tr').first();
		await expect(row.getByRole('link', { name: first.name })).toHaveAttribute(
			'href', new RegExp(`/${first.season}/men/players/${first.player_id}/$`)
		);
		await expect(row).toContainText(`${first.season} · ${first.team_short}`);
		await expect(row).toContainText(`對 ${first.opponent}`);
		if (first.game_id !== null) {
			await expect(row.getByRole('link', { name: new RegExp(`對 ${first.opponent}$`) })).toHaveAttribute(
				'href', new RegExp(`/${first.season}/men/matches/${first.game_id}/$`)
			);
		}
	}
});

test('點單場紀錄進到該場賽事頁', async ({ page }) => {
	const first = men.boards[0].match_records.find((r: { game_id: number | null }) => r.game_id !== null);
	test.skip(!first, '目前沒有可連到賽事頁的單場紀錄');
	await page.goto(`${BASE}/records/men/match/`);
	const table = page.locator(`#r-${men.boards[0].key}`).getByRole('table');
	await table.getByRole('link', { name: new RegExp(`對 ${first.opponent}$`) }).first().click();
	await expect(page).toHaveURL(`${BASE}/${first.season}/men/matches/${first.game_id}/`);
});

test('三種紀錄互相切換，切組別停在同一種紀錄', async ({ page }) => {
	await page.goto(`${BASE}/records/men/`);
	const kinds = page.getByRole('navigation', { name: '紀錄類型' });
	await kinds.getByRole('link', { name: '累計紀錄' }).click();
	await expect(page).toHaveURL(`${BASE}/records/men/cumulative/`);
	await expect(page.getByRole('heading', { level: 2, name: '累計紀錄' })).toBeVisible();
	await page.getByRole('navigation', { name: '組別' }).getByRole('link', { name: '女子組' }).click();
	await expect(page).toHaveURL(`${BASE}/records/women/cumulative/`);
	await expect(kinds.getByRole('link', { name: '累計紀錄' })).toHaveAttribute('aria-current', 'page');
	await expect(page.getByRole('navigation', { name: '主要分頁' }).getByRole('button', { name: '更多' }))
		.toHaveAttribute('aria-current', 'page');
	await kinds.getByRole('link', { name: '單場紀錄' }).click();
	await expect(page).toHaveURL(`${BASE}/records/women/match/`);
	await page.getByRole('navigation', { name: '組別' }).getByRole('link', { name: '男子組' }).click();
	await expect(page).toHaveURL(`${BASE}/records/men/match/`);
	await page.goto(`${BASE}/records/women/cumulative/`);
	await kinds.getByRole('link', { name: '單季紀錄' }).click();
	await expect(page).toHaveURL(`${BASE}/records/women/`);
	await expect(page.getByRole('heading', { level: 2, name: '單季紀錄' })).toBeVisible();
});

test('累計含進行中賽季的紀錄標示「含進行中賽季」', async ({ page }) => {
	const live = men.boards.flatMap((b: { cumulative_records: { in_progress: boolean }[] }) => b.cumulative_records)
		.filter((r: { in_progress: boolean }) => r.in_progress);
	test.skip(!live.length, '目前沒有含進行中賽季的累計紀錄');
	await page.goto(`${BASE}/records/men/cumulative/`);
	await expect(page.getByText('含進行中賽季').first()).toBeVisible();
});

test('進行中賽季的紀錄標示「進行中」', async ({ page }) => {
	const live = men.boards.flatMap((b: { season_records: { in_progress: boolean }[] }) => b.season_records)
		.filter((r: { in_progress: boolean }) => r.in_progress);
	test.skip(!live.length, '目前沒有進行中賽季的紀錄');
	await page.goto(`${BASE}/records/men/`);
	await expect(page.getByText('進行中').first()).toBeVisible();
});

test('進行中賽季的單場紀錄標示「進行中」', async ({ page }) => {
	const live = men.boards.flatMap((b: { match_records: { in_progress: boolean }[] }) => b.match_records)
		.filter((r: { in_progress: boolean }) => r.in_progress);
	test.skip(!live.length, '目前沒有進行中賽季的單場紀錄');
	await page.goto(`${BASE}/records/men/match/`);
	await expect(page.getByText('進行中').first()).toBeVisible();
});

async function switchToWomenOnRecords(page: Page) {
	await expect(page.getByRole('navigation', { name: '賽季' })).toHaveCount(0);
	const groups = page.getByRole('navigation', { name: '組別' });
	await groups.getByRole('link', { name: '女子組' }).click();
	await expect(page).toHaveURL(`${BASE}/records/women/`);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText('女子組 歷史紀錄');
	await expect(groups.getByRole('link', { name: '女子組' })).toHaveAttribute('aria-current', 'page');
	await expect(page.getByText(coverageText(entry('women').coverage))).toBeVisible();
}

test('桌面寬度從側欄進入歷史紀錄，組別切換停在歷史紀錄', async ({ page }) => {
	await page.setViewportSize({ width: 1280, height: 800 });
	await page.goto(`${BASE}/${currentSeason}/men/`);
	const nav = page.getByRole('navigation', { name: '頁面' });
	await nav.getByRole('link', { name: '歷史紀錄', exact: true }).click();
	await expect(page).toHaveURL(`${BASE}/records/men/`);
	await expect(nav.getByRole('link', { name: '歷史紀錄', exact: true })).toHaveAttribute('aria-current', 'page');
	await expect(nav.getByRole('link', { name: '首頁', exact: true })).not.toHaveAttribute('aria-current', 'page');
	await switchToWomenOnRecords(page);
});

test('手機寬度從「更多」面板進入歷史紀錄，組別切換停在歷史紀錄', async ({ page }) => {
	await page.goto(`${BASE}/${currentSeason}/men/`);
	await openMoreSheet(page);
	await page.getByRole('dialog', { name: '更多' }).getByRole('link', { name: '歷史紀錄', exact: true }).click();
	await expect(page).toHaveURL(`${BASE}/records/men/`);
	await expect(page.getByRole('navigation', { name: '主要分頁' }).getByRole('button', { name: '更多' }))
		.toHaveAttribute('aria-current', 'page');
	await switchToWomenOnRecords(page);
});

test('手機寬度沒有橫向溢位', async ({ page }) => {
	await page.setViewportSize({ width: 360, height: 800 });
	for (const path of ['men/', 'women/', 'men/cumulative/', 'women/cumulative/', 'men/match/', 'women/match/']) {
		await page.goto(`${BASE}/records/${path}`);
		expect(
			await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
		).toBeLessThanOrEqual(0);
	}
});
