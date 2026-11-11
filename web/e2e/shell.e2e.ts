import { readFileSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';
import {
	BASE,
	currentSeason as current,
	groupLink,
	openMoreSheet,
	openSeasonSheet,
	readJson,
	seasonButton,
	seasonSheet,
	seasonsData,
	switchGroup,
	switchSeason,
	topBar
} from './helpers.js';
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

test.describe('手機頂部列', () => {
	test('站名連到站台根目錄，往下捲之後仍吸在畫面頂端', async ({ page }) => {
		await page.goto(`${group}/matches/`);
		await expect(topBar(page).getByRole('link', { name: 'TVL 數據' })).toHaveAttribute('href', `${BASE}/`);
		await page.evaluate(() => window.scrollTo(0, 1500));
		await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(1000);
		const box = (await topBar(page).boundingBox())!;
		expect(box.y).toBe(0);
		expect(box.height).toBeGreaterThanOrEqual(44);
	});

	test('組別是「男子｜女子」，點女子到同一種頁面、點男子回來，並記住組別', async ({ page }) => {
		await page.goto(`${group}/teams/`);
		await expect(topBar(page).getByRole('navigation', { name: '組別' }).getByRole('link')).toHaveText(['男子', '女子']);
		await expect(groupLink(page, '男子')).toHaveAttribute('aria-current', 'page');
		await expect(groupLink(page, '女子')).toHaveAttribute('href', `${BASE}/${current}/women/teams/`);
		await switchGroup(page, '女子', `${BASE}/${current}/women/teams/`);
		await expect(groupLink(page, '女子')).toHaveAttribute('aria-current', 'page');
		await expect.poll(() => page.evaluate(() => localStorage.getItem('tvl.group'))).toBe('women');
		await switchGroup(page, '男子', `${group}/teams/`);
		await expect.poll(() => page.evaluate(() => localStorage.getItem('tvl.group'))).toBe('men');
	});

	test('組別切換鈕至少 44px', async ({ page }) => {
		await page.goto(`${group}/`);
		for (const label of ['男子', '女子'] as const) {
			const box = (await groupLink(page, label).boundingBox())!;
			expect(box.width).toBeGreaterThanOrEqual(44);
			expect(box.height).toBeGreaterThanOrEqual(44);
		}
	});

	test('沒有頁首卡，也沒有「非官方・資料整理自聯賽官網」那一行；頁尾聲明仍在', async ({ page }) => {
		await page.goto(`${group}/`);
		await expect(page.getByText('非官方・資料整理自聯賽官網')).toBeHidden();
		await expect(page.getByRole('navigation', { name: '頁面' })).toHaveCount(0);
		await expect(page.getByRole('contentinfo')).toContainText('非官方');
	});

	test('沒有賽季與組別的頁：頂部列只有站名', async ({ page }) => {
		await page.goto(`${BASE}/no-such-page/`);
		await expect(topBar(page).getByRole('link', { name: 'TVL 數據' })).toBeVisible();
		await expect(topBar(page).getByRole('navigation')).toHaveCount(0);
		await expect(seasonButton(page)).toHaveCount(0);
	});

	test('320px 寬時頂部列不橫向溢出', async ({ page }) => {
		await page.setViewportSize({ width: 320, height: 700 });
		await page.goto(`${group}/`);
		await expect(topBar(page)).toBeVisible();
		expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(320);
		const last = (await seasonButton(page).boundingBox())!;
		expect(last.x + last.width).toBeLessThanOrEqual(320);
	});
});

test.describe('手機頂部列的賽季面板', () => {
	const other: { name: string } = seasonsData.seasons.find((s: { name: string }) => s.name !== current);
	const names: string[] = seasonsData.seasons.map((s: { name: string }) => s.name);

	test('賽季鈕顯示目前賽季，無障礙名稱說明用途與目前賽季', async ({ page }) => {
		await page.goto(`${group}/teams/`);
		await expect(seasonButton(page)).toHaveText(current);
		await expect(seasonButton(page)).toHaveAccessibleName(`切換賽季，目前 ${current}`);
		await expect(seasonButton(page)).toHaveAttribute('aria-haspopup', 'dialog');
		expect((await seasonButton(page).boundingBox())!.height).toBeGreaterThanOrEqual(44);
	});

	test('面板列出所有賽季並標出目前賽季；每項至少 44px', async ({ page }) => {
		await page.goto(`${group}/teams/`);
		await openSeasonSheet(page);
		await expect(seasonSheet(page).getByRole('link')).toHaveText(names);
		await expect(seasonSheet(page).getByRole('link', { name: current, exact: true })).toHaveAttribute(
			'aria-current',
			'page'
		);
		await expect(seasonSheet(page).getByRole('link', { name: other.name, exact: true })).not.toHaveAttribute(
			'aria-current',
			'page'
		);
		for (const link of await seasonSheet(page).getByRole('link').all()) {
			expect((await link.boundingBox())!.height).toBeGreaterThanOrEqual(44);
		}
	});

	test('選另一個賽季：到該賽季的同一種頁面，面板關閉，賽季鈕改成新賽季', async ({ page }) => {
		await page.goto(`${group}/teams/`);
		await switchSeason(page, other.name, `${BASE}/${other.name}/men/teams/`);
		await expect(seasonSheet(page)).toBeHidden();
		await expect(seasonButton(page)).toHaveText(other.name);
		await openSeasonSheet(page);
		await expect(seasonSheet(page).getByRole('link', { name: other.name, exact: true })).toHaveAttribute(
			'aria-current',
			'page'
		);
	});

	test('Esc 關閉面板，焦點回到賽季鈕', async ({ page }) => {
		await page.goto(`${group}/`);
		await openSeasonSheet(page);
		await page.keyboard.press('Escape');
		await expect(seasonSheet(page)).toBeHidden();
		await expect(seasonButton(page)).toBeFocused();
	});

	test('點遮罩關閉面板，焦點回到賽季鈕', async ({ page }) => {
		await page.goto(`${group}/`);
		await openSeasonSheet(page);
		await page.mouse.click(180, 20);
		await expect(seasonSheet(page)).toBeHidden();
		await expect(seasonButton(page)).toBeFocused();
	});

	test('歷史紀錄頁沒有賽季鈕，組別仍可切換', async ({ page }) => {
		await page.goto(`${BASE}/records/men/`);
		await expect(topBar(page)).toBeVisible();
		await expect(seasonButton(page)).toHaveCount(0);
		await expect(groupLink(page, '女子')).toHaveAttribute('href', `${BASE}/records/women/`);
	});
});

test.describe('手機頂部列讓出的位置', () => {
	const bottomOf = async (page: Page) => {
		const box = (await topBar(page).boundingBox())!;
		return box.y + box.height;
	};

	test('賽事頁的月份列吸在頂部列下方，點月份後標題不被蓋住', async ({ page }) => {
		await page.goto(`${group}/matches/`);
		const nav = page.getByRole('navigation', { name: '月份' });
		await expect(nav).toBeVisible();
		await page.evaluate(() => window.scrollTo(0, 600));
		await expect.poll(async () => Math.round((await nav.boundingBox())!.y)).toBe(Math.round(await bottomOf(page)));
		const last = nav.getByRole('link').last();
		const key = (await last.getAttribute('href'))!.replace('#m-', '');
		await last.click();
		await expect(last).toHaveAttribute('aria-current', 'location');
		const navBox = (await nav.boundingBox())!;
		expect((await page.locator(`#h-${key}`).boundingBox())!.y).toBeGreaterThanOrEqual(navBox.y + navBox.height);
	});

	test('從單場頁回賽事列表，月份標題不被頂部列蓋住', async ({ page }) => {
		const month = anyGame.date.slice(0, 7);
		await page.goto(`${group}/matches/${anyGame.game_id}/`);
		await expect(async () => {
			await topBar(page).getByRole('link', { name: '返回' }).click();
			await expect(page).toHaveURL(`${group}/matches/#m-${month}`, { timeout: 1000 });
		}).toPass({ timeout: 10_000 });
		const heading = page.locator(`#h-${month}`);
		await expect(heading).toBeInViewport();
		const nav = (await page.getByRole('navigation', { name: '月份' }).boundingBox())!;
		expect(Math.round(nav.y)).toBeGreaterThanOrEqual(Math.round(await bottomOf(page)));
		expect((await heading.boundingBox())!.y).toBeGreaterThanOrEqual(nav.y + nav.height);
	});

	test('排行榜的項目列吸在頂部列下方，跳轉後榜單標題不被蓋住', async ({ page }) => {
		const entry = readJson('leaders').leaders.find((e: { group: string }) => e.group === 'women');
		const season: string = seasonsData.seasons.find((s: { cup_id: number }) => s.cup_id === entry.cup_id).name;
		await page.goto(`${BASE}/${season}/women/leaders/`);
		const nav = page.getByRole('navigation', { name: /項目$/ });
		await expect(nav).toBeVisible();
		await page.evaluate(() => window.scrollTo(0, 800));
		await expect.poll(async () => Math.round((await nav.boundingBox())!.y)).toBe(Math.round(await bottomOf(page)));
		const last = nav.getByRole('link').last();
		await last.click();
		await expect(last).toHaveAttribute('aria-current', 'location');
		const navBox = (await nav.boundingBox())!;
		const href = (await last.getAttribute('href'))!.replace('#', '');
		expect((await page.locator(`#${href} h2`).boundingBox())!.y).toBeGreaterThanOrEqual(navBox.y + navBox.height);
	});
});

test.describe('桌面寬度', () => {
	test.use({ viewport: { width: 1280, height: 800 } });

	test('沒有底部分頁列，側欄導覽仍在', async ({ page }) => {
		await page.goto(`${group}/`);
		await expect(page.getByRole('navigation', { name: '頁面' })).toBeVisible();
		await expect(page.getByRole('navigation', { name: '主要分頁' })).toHaveCount(0);
	});

	test('沒有頂部列：側欄仍有組別、賽季 chip 與非官方註記', async ({ page }) => {
		await page.goto(`${group}/`);
		await expect(seasonButton(page)).toHaveCount(0);
		const side = page.getByRole('banner');
		await expect(side.getByRole('navigation', { name: '組別' }).getByRole('link')).toHaveText(['男子組', '女子組']);
		await expect(side.getByRole('navigation', { name: '賽季' }).getByRole('link')).toHaveText(
			seasonsData.seasons.map((s: { name: string }) => s.name)
		);
		await expect(side.getByText('非官方・資料整理自聯賽官網')).toBeVisible();
		expect((await side.boundingBox())!.width).toBe(272);
	});

	test('賽事頁的吸頂月份列貼在視窗頂端', async ({ page }) => {
		await page.goto(`${group}/matches/`);
		const nav = page.getByRole('navigation', { name: '月份' });
		await expect(nav).toBeVisible();
		await page.evaluate(() => window.scrollTo(0, 600));
		await expect.poll(async () => (await nav.boundingBox())!.y).toBe(0);
	});
});
