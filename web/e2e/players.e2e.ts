import { expect, test, type Page } from '@playwright/test';
import { BASE, currentSeason, findSeason, OLD_CUP, OLD_SEASON, readJson } from './helpers.js';

type Season = {
	cup_id: number;
	team_id: number;
	jersey_number: number | null;
	position: string | null;
	games: number;
};
type Player = { player_id: number; group: string; name: string | null; seasons: Season[] };
const players: Player[] = readJson('players').players;
const teamIds: number[] = readJson('teams')
	.teams.filter((t: { group: string }) => t.group === 'men')
	.map((t: { team_id: number }) => t.team_id);
// 開季初當季還沒有球員，退回最近一個有球員的賽季
const picked = findSeason('players');
const current = picked?.season ?? currentSeason;
const currentCup = picked?.cupId ?? 0;
const noPlayers = '所有賽季的男子組都沒有球員';

const registeredIn = (cup: number, group: string) =>
	players.filter((p) => p.group === group && p.seasons.some((s) => s.cup_id === cup));
const seasonOf = (p: Player, cup: number) => p.seasons.find((s) => s.cup_id === cup)!;
const noOverflow = (page: Page) =>
	page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
const nav = (page: Page) => page.getByRole('navigation', { name: '頁面' });
const playerLink = (page: Page) => nav(page).getByRole('link', { name: '球員', exact: true });

test('導覽「球員」到球員列表', async ({ page }) => {
	await page.goto(`${BASE}/${current}/men/`);
	await playerLink(page).click();
	await expect(page).toHaveURL(`${BASE}/${current}/men/players/`);
	await expect(playerLink(page)).toHaveAttribute('aria-current', 'page');
	await expect(page.getByRole('heading', { level: 1 })).toHaveText('球員');
	expect(await noOverflow(page)).toBeLessThanOrEqual(0);
});

test('列表第一張球員卡進球員頁：基本資料與本季數據', async ({ page }) => {
	test.skip(!picked, noPlayers);
	await page.goto(`${BASE}/${current}/men/players/`);
	const card = page.locator('ul.players li a').first();
	await expect(card).toBeVisible();
	await card.click();
	await expect(page).toHaveURL(/\/men\/players\/\d+\/$/);
	const id = Number(page.url().match(/\/players\/(\d+)\/$/)![1]);
	const player = players.find((p) => p.group === 'men' && p.player_id === id)!;
	expect(player.seasons.find((s) => s.cup_id === currentCup)).toBeDefined();
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(player.name ?? `球員 #${id}`);

	await expect(page.getByRole('heading', { level: 2, name: '基本資料' })).toBeVisible();
	for (const label of ['背號', '位置', '身高', '體重']) {
		await expect(page.locator('dl.profile dt', { hasText: label })).toBeVisible();
	}
	await expect(page.getByRole('heading', { level: 2, name: '本季數據' })).toBeVisible();
	if (seasonOf(player, currentCup).games > 0) {
		await expect(page.locator('.stats .value')).toHaveCount(3);
	}
	expect(await noOverflow(page)).toBeLessThanOrEqual(0);
	await expect(playerLink(page)).toHaveAttribute('aria-current', 'page');
});

test('舉球員的主卡顯示舉球到位率等匯出值，位置顯示中文簡稱', async ({ page }) => {
	const player = registeredIn(currentCup, 'men').find((p) => {
		const s = seasonOf(p, currentCup) as Season & Record<string, number>;
		return s.games > 0 && s.position === 'S' && s.set_total > 0;
	});
	test.skip(!player, '用於比對的賽季男子組沒有有舉球紀錄的舉球員');
	await page.goto(`${BASE}/${current}/men/players/${player!.player_id}/`);
	const s = seasonOf(player!, currentCup) as Season & Record<string, number>;
	const position = page.locator('dl.profile div', { hasText: '位置' }).locator('dd');
	await expect(position).toHaveText('舉球（S）');
	const setRate = `${((s.set_excellent / s.set_total) * 100).toFixed(1)}%`;
	const stats = page.locator('.stats');
	for (const [label, value, detail] of [
		['舉球到位率', setRate, `到位 ${s.set_excellent}／${s.set_total}`],
		['舉球次數', String(s.set_total), null]
	] as const) {
		const cell = stats.locator('div', { has: page.getByText(label, { exact: true }) });
		await expect(cell.locator('.value')).toHaveText(value);
		if (detail) await expect(cell.locator('.detail')).toHaveText(detail);
	}
	const totals = page.locator('dl.totals');
	for (const [label, value] of [
		['出賽場次', s.games],
		['出賽局數', s.sets_played],
		['總得分', s.total_points]
	] as const) {
		await expect(totals.locator('div', { hasText: label }).locator('dd')).toHaveText(String(value));
	}
});

test('沒有單場數據的球員顯示空狀態', async ({ page }) => {
	const player = registeredIn(currentCup, 'men').find((p) => seasonOf(p, currentCup).games === 0);
	test.skip(!player, '用於比對的賽季男子組沒有 games = 0 的球員');
	await page.goto(`${BASE}/${current}/men/players/${player!.player_id}/`);
	await expect(page.getByText('這一季還沒有單場數據')).toBeVisible();
	await expect(page.locator('.stats')).toHaveCount(0);
});

test('列表依球隊分組：每隊一個標題，第一張卡是第一隊的第一位', async ({ page }) => {
	test.skip(!picked, noPlayers);
	const registered = registeredIn(currentCup, 'men');
	const firstTeam = teamIds.find((id) => registered.some((p) => seasonOf(p, currentCup).team_id === id))!;
	const teamCount = teamIds.filter((id) =>
		registered.some((p) => seasonOf(p, currentCup).team_id === id)
	).length;
	const first = registered
		.filter((p) => seasonOf(p, currentCup).team_id === firstTeam)
		.sort(
			(a, b) =>
				(seasonOf(a, currentCup).jersey_number ?? Infinity) -
					(seasonOf(b, currentCup).jersey_number ?? Infinity) ||
				(a.name ?? '').localeCompare(b.name ?? '', 'zh-Hant')
		)[0];
	await page.goto(`${BASE}/${current}/men/players/`);
	await expect(page.getByRole('heading', { level: 2 })).toHaveCount(teamCount);
	const card = page.locator('ul.players li').first();
	await expect(card).toContainText(first.name ?? `球員 #${first.player_id}`);
	const number = seasonOf(first, currentCup).jersey_number;
	await expect(card.locator('.number')).toHaveText(number === null ? '—' : String(number));
});

test('不存在的球員沒有球員頁', async ({ page }) => {
	const res = await page.goto(`${BASE}/${current}/men/players/999999/`);
	expect(res?.status()).toBe(404);
});

test('2024-25 球員列表依資料顯示', async ({ page }) => {
	await page.goto(`${BASE}/${OLD_SEASON}/men/players/`);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText('球員');
	if (registeredIn(OLD_CUP, 'men').length === 0) {
		await expect(page.getByText('這一季沒有週登錄資料')).toBeVisible();
	} else {
		await expect(page.locator('ul.players li').first()).toBeVisible();
	}
	expect(await noOverflow(page)).toBeLessThanOrEqual(0);
});

test('球隊頁的球員卡連到球員頁', async ({ page }) => {
	// 球隊頁的球員卡來自週登錄，退回最近一個有週登錄的賽季
	const rosterSeason = findSeason('rosters');
	test.skip(!rosterSeason, '所有賽季的男子組都沒有週登錄');
	await page.goto(`${BASE}/${rosterSeason!.season}/men/teams/`);
	await page.locator('li a').first().click();
	const card = page.locator('ul.players li a').first();
	await expect(card).toHaveAttribute('href', /\/players\/\d+\/$/);
	await card.click();
	await expect(page).toHaveURL(/\/men\/players\/\d+\/$/);
	await expect(page.getByRole('heading', { level: 2, name: '基本資料' })).toBeVisible();
});

test('球員頁切換組別退回列表', async ({ page }) => {
	test.skip(!picked, noPlayers);
	const player = registeredIn(currentCup, 'men')[0];
	await page.goto(`${BASE}/${current}/men/players/${player.player_id}/`);
	const href = await page
		.getByRole('navigation', { name: '組別' })
		.getByRole('link', { name: '女子組' })
		.getAttribute('href');
	expect(href!.endsWith(`/${current}/women/players/`)).toBe(true);
});

test('歷史賽季球員列表與球員頁顯示當季隊名', async ({ page }) => {
	await page.goto(`${BASE}/2025-26/men/players/`);
	const main = page.getByRole('main');
	await expect(main).toContainText('臺北國北獅');
	await expect(main).not.toContainText('臺北鳳凰');
	await page
		.locator('section')
		.filter({ has: page.getByRole('heading', { level: 2, name: '臺北國北獅' }) })
		.locator('ul.players li a')
		.first()
		.click();
	await expect(page).toHaveURL(/\/2025-26\/men\/players\/\d+\/$/);
	await expect(page.locator('header .team')).toHaveText('臺北國北獅');
});
