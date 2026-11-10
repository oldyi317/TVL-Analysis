import { readFileSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';
import { BASE, currentSeason as current, openMoreSheet, readJson, seasonsData } from './helpers.js';
import { isPlayed } from '../src/lib/match.js';

const cupId: number = seasonsData.current_cup_id;
const matchesData = JSON.parse(readFileSync('src/lib/data/matches.json', 'utf8'));
const menMatches: { date: string; game_id: number; sets: unknown[] }[] = matchesData.matches.find(
	(m: { cup_id: number; group: string }) => m.cup_id === cupId && m.group === 'men'
).matches;
const anyGame = menMatches.find(isPlayed) ?? menMatches[0];
const team = readJson('teams').teams.find(
	(t: { group: string; is_active: boolean; cup_ids: number[] }) =>
		t.group === 'men' && t.is_active && t.cup_ids.includes(cupId)
);
const player = readJson('players').players.find(
	(p: { group: string; seasons: { cup_id: number }[] }) =>
		p.group === 'men' && p.seasons.some((s) => s.cup_id === cupId)
);

const tabs = (page: Page) => page.getByRole('navigation', { name: '主要分頁' });
const moreButton = (page: Page) => tabs(page).getByRole('button', { name: '更多' });
const sheet = (page: Page) => page.getByRole('dialog', { name: '更多' });
const tabLink = (page: Page, name: string) => tabs(page).getByRole('link', { name, exact: true });

const group = `${BASE}/${current}/men`;

test.describe('手機底部分頁列', () => {
	const cases: [string, () => string | Promise<string>, string | null][] = [
		['賽季首頁', () => `${group}/`, '首頁'],
		['賽事', () => `${group}/matches/`, '賽事'],
		['單場頁', () => `${group}/matches/${anyGame.game_id}/`, '賽事'],
		['戰績', () => `${group}/standings/`, '戰績'],
		['球隊', () => `${group}/teams/`, '球隊'],
		['球員', () => `${group}/players/`, '更多'],
		['排行榜', () => `${group}/leaders/`, '更多'],
		['歷史紀錄', () => `${BASE}/records/men/`, '更多'],
	];

	for (const [name, url, currentTab] of cases) {
		test(`${name}：五格依序、前四格連到現行賽季，目前格是「${currentTab}」`, async ({ page }) => {
			await page.goto(await url());
			await expect(tabs(page)).toBeVisible();
			const cells = tabs(page).locator('a, button');
			await expect(cells).toHaveText(['首頁', '賽事', '戰績', '球隊', '更多']);
			const hrefs = ['', 'matches/', 'standings/', 'teams/'];
			for (const [i, label] of ['首頁', '賽事', '戰績', '球隊'].entries()) {
				await expect(tabLink(page, label)).toHaveAttribute('href', `${group}/${hrefs[i]}`);
			}
			await expect(moreButton(page)).toHaveAttribute('aria-haspopup', 'dialog');
			for (const label of ['首頁', '賽事', '戰績', '球隊', '更多']) {
				const cell = label === '更多' ? moreButton(page) : tabLink(page, label);
				if (label === currentTab) await expect(cell).toHaveAttribute('aria-current', 'page');
				else await expect(cell).not.toHaveAttribute('aria-current', 'page');
			}
			await expect(page.getByRole('navigation', { name: '頁面' })).toHaveCount(0);
		});
	}

	test('球隊頁屬於球隊、球員頁屬於「更多」', async ({ page }) => {
		await page.goto(`${group}/teams/${team.team_id}/`);
		await expect(tabLink(page, '球隊')).toHaveAttribute('aria-current', 'page');
		await expect(moreButton(page)).not.toHaveAttribute('aria-current', 'page');

		await page.goto(`${group}/players/${player.player_id}/`);
		await expect(moreButton(page)).toHaveAttribute('aria-current', 'page');
		await expect(tabLink(page, '球隊')).not.toHaveAttribute('aria-current', 'page');
	});

	test('沒有賽季與組別的頁：分頁列連到現行賽季，沒有目前格', async ({ page }) => {
		await page.goto(`${BASE}/no-such-page/`);
		await expect(tabs(page)).toBeVisible();
		await expect(tabLink(page, '賽事')).toHaveAttribute('href', `${group}/matches/`);
		await expect(tabs(page).locator('[aria-current]')).toHaveCount(0);
	});

	test('歷史紀錄頁前四格連到現行賽季與目前組別', async ({ page }) => {
		await page.goto(`${BASE}/records/women/`);
		await expect(tabLink(page, '戰績')).toHaveAttribute('href', `${BASE}/${current}/women/standings/`);
	});

	test('「更多」面板只有球員、排行榜、歷史紀錄；點項目導頁並關閉', async ({ page }) => {
		await page.goto(`${group}/`);
		await openMoreSheet(page);
		await expect(sheet(page).getByRole('link')).toHaveText(['球員', '排行榜', '歷史紀錄']);
		await sheet(page).getByRole('link', { name: '排行榜' }).click();
		await expect(page).toHaveURL(`${group}/leaders/`);
		await expect(sheet(page)).toBeHidden();
		await expect(moreButton(page)).toHaveAttribute('aria-current', 'page');
		await openMoreSheet(page);
		await expect(sheet(page).getByRole('link', { name: '排行榜' })).toHaveAttribute('aria-current', 'page');
		await expect(sheet(page).getByRole('link', { name: '球員' })).not.toHaveAttribute('aria-current', 'page');
	});

	test('面板每個項目至少 44px 高', async ({ page }) => {
		await page.goto(`${group}/`);
		await openMoreSheet(page);
		for (const link of await sheet(page).getByRole('link').all()) {
			expect((await link.boundingBox())!.height).toBeGreaterThanOrEqual(44);
		}
	});

	test('Esc 關閉面板，焦點回到「更多」', async ({ page }) => {
		await page.goto(`${group}/`);
		await openMoreSheet(page);
		await page.keyboard.press('Escape');
		await expect(sheet(page)).toBeHidden();
		await expect(moreButton(page)).toBeFocused();
	});

	test('點遮罩關閉面板，焦點回到「更多」', async ({ page }) => {
		await page.goto(`${group}/`);
		await openMoreSheet(page);
		await page.mouse.click(180, 40);
		await expect(sheet(page)).toBeHidden();
		await expect(moreButton(page)).toBeFocused();
	});

	test('點面板內非連結區域不會關閉', async ({ page }) => {
		await page.goto(`${group}/`);
		await openMoreSheet(page);
		const box = (await sheet(page).getByRole('link').first().boundingBox())!;
		await page.mouse.click(box.x + box.width / 2, box.y - 2);
		await expect(sheet(page)).toBeVisible();
	});

	test('每格至少 44px，捲到底時頁尾最後一行不被分頁列遮住', async ({ page }) => {
		await page.goto(`${group}/`);
		await expect(page.locator('meta[name="viewport"]')).toHaveAttribute('content', /viewport-fit=cover/);
		for (const cell of await tabs(page).locator('a, button').all()) {
			const box = (await cell.boundingBox())!;
			expect(box.width).toBeGreaterThanOrEqual(44);
			expect(box.height).toBeGreaterThanOrEqual(44);
		}
		await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
		const last = (await page.getByRole('contentinfo').getByRole('paragraph').last().boundingBox())!;
		const bar = (await tabs(page).boundingBox())!;
		expect(last.y + last.height).toBeLessThanOrEqual(bar.y);
	});
});

test.describe('桌面寬度', () => {
	test.use({ viewport: { width: 1280, height: 800 } });

	test('沒有底部分頁列，側欄導覽仍在', async ({ page }) => {
		await page.goto(`${group}/`);
		await expect(page.getByRole('navigation', { name: '頁面' })).toBeVisible();
		await expect(page.getByRole('navigation', { name: '主要分頁' })).toHaveCount(0);
	});
});
