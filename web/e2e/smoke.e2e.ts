import {
	BASE,
	currentSeason as current,
	findSeason,
	groupLink,
	OLD_CUP,
	OLD_SEASON,
	openSeasonSheet,
	readJson,
	seasonSheet,
	seasonsData,
	switchGroup,
	expect,
	test
} from './helpers.js';

// 開季初當季球隊還沒登錄齊，退回最近一個有球隊的賽季
const teamSeason = findSeason('teams');
const teamCount: number = teamSeason
	? readJson('teams').teams.filter(
			(t: { group: string; cup_ids: number[] }) =>
				t.group === 'men' && t.cup_ids.includes(teamSeason.cupId)
		).length
	: 0;

test('球隊頁列出帶隊色的圓形隊色圓標', async ({ page }) => {
	test.skip(!teamSeason, '所有賽季的男子組都沒有球隊');
	await page.goto(`${BASE}/${teamSeason!.season}/men/teams/`);
	const rows = page.getByRole('list').getByRole('listitem');
	await expect(rows.first()).toBeVisible();
	expect(await rows.count()).toBe(teamCount);
	const badge = rows.first().locator('.badge');
	await expect(badge).toHaveCSS('border-radius', '50%');
	expect(await badge.evaluate((el) => getComputedStyle(el).backgroundColor)).not.toBe(
		'rgba(0, 0, 0, 0)'
	);
});

test('組別切換連結正確且停留在同一頁面類型', async ({ page }) => {
	await page.goto(`${BASE}/${current}/men/teams/`);
	await expect(groupLink(page, '女子')).toHaveAttribute('href', `${BASE}/${current}/women/teams/`);
	await switchGroup(page, '女子', `${BASE}/${current}/women/teams/`);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(`${current} 女子組 球隊`);
	await expect(groupLink(page, '女子')).toHaveAttribute('aria-current', 'page');
});

test('賽季連結指向同組別的另一季', async ({ page }) => {
	await page.goto(`${BASE}/${current}/women/teams/`);
	const other = seasonsData.seasons.find((s: { name: string }) => s.name !== current);
	await openSeasonSheet(page);
	await expect(seasonSheet(page).getByRole('link', { name: other.name })).toHaveAttribute(
		'href',
		`${BASE}/${other.name}/women/teams/`
	);
});

test('每一頁的頁尾都有資料來源與非官方聲明', async ({ page }) => {
	for (const path of [`${BASE}/`, `${BASE}/${current}/men/teams/`]) {
		await page.goto(path);
		const footer = page.getByRole('contentinfo');
		await expect(footer).toContainText('資料來源');
		await expect(footer).toContainText('非官方');
		await expect(footer).toContainText('商標');
	}
});

test('404 頁顯示「找不到這頁」並可回首頁', async ({ page }) => {
	await page.goto(`${BASE}/404`);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText('找不到這頁');
	await expect(page.getByRole('link', { name: '回首頁' })).toHaveAttribute('href', `${BASE}/`);
});

// 每種預渲染頁面各取一頁；詳情頁取資料齊全的舊賽季
const oldGroup = `${BASE}/${OLD_SEASON}/men`;
const oldGame: { game_id: number; home_team_id: number } = readJson('matches').matches.find(
	(m: { cup_id: number; group: string }) => m.cup_id === OLD_CUP && m.group === 'men'
).matches[0];
const oldPlayer: { player_id: number } = readJson('players').players.find(
	(p: { group: string; seasons: { cup_id: number }[] }) =>
		p.group === 'men' && p.seasons.some((s) => s.cup_id === OLD_CUP)
);
const everyPageKind: [string, string][] = [
	['首頁', `${BASE}/`],
	['賽季組別首頁', `${oldGroup}/`],
	['戰績', `${oldGroup}/standings/`],
	['賽事列表', `${oldGroup}/matches/`],
	['單場頁', `${oldGroup}/matches/${oldGame.game_id}/`],
	['球隊列表', `${oldGroup}/teams/`],
	['球隊頁', `${oldGroup}/teams/${oldGame.home_team_id}/`],
	['球員列表', `${oldGroup}/players/`],
	['球員頁', `${oldGroup}/players/${oldPlayer.player_id}/`],
	['排行榜', `${oldGroup}/leaders/`],
	['歷史紀錄', `${BASE}/records/men/`],
	['styleguide', `${BASE}/styleguide/`],
	['404 頁', `${BASE}/404`],
	['不存在的網址', `${oldGroup}/teams/999/`]
];

// 讀一次就斷言、不重試：page.goto 返回時就該已完成 hydration
test.describe('進頁返回時已完成 hydration', () => {
	for (const [name, path] of everyPageKind) {
		test(name, async ({ page }) => {
			await page.goto(path);
			expect(await page.locator('html').getAttribute('data-hydrated')).toBe('');
		});
	}
});

test('分頁小圖示是可取得的 favicon.png', async ({ page }) => {
	await page.goto(`${BASE}/`);
	const href = await page.locator('link[rel="icon"]').getAttribute('href');
	expect(href).toMatch(/favicon\.png$/);
	const res = await page.request.get(new URL(href!, page.url()).toString());
	expect(res.status()).toBe(200);
	expect(res.headers()['content-type']).toContain('image/png');
});
