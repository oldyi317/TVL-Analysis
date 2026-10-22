import { expect, test, type Locator } from '@playwright/test';
import {
	BASE,
	currentSeason as current,
	findSeason,
	OLD_CUP,
	OLD_SEASON,
	readJson,
	seasonsData
} from './helpers.js';

type Team = {
	team_id: number;
	group: string;
	name: string;
	short_name: string;
	is_active: boolean;
	cup_ids: number[];
};
const teams: Team[] = readJson('teams').teams;
const matchGroups = readJson('matches').matches;
const rosters = readJson('rosters').rosters;
const currentCup: number = seasonsData.current_cup_id;
// 開季初當季沒有週登錄，退回最近一個有週登錄的賽季
const rosterSeason = findSeason('rosters');

const standings = readJson('standings').standings;

type Week = { week_label: string; week_start_date: string | null; players: RosterPlayer[] };
type RosterPlayer = { player_id: number; jersey_number: number | null; position: string | null };
type Standing = { team_id: number; rank: number; wins: number; losses: number; points: number };

const menIn = (cup: number) => teams.filter((t) => t.group === 'men' && t.cup_ids.includes(cup));
const latestWeekOf = (cup: number, group: string, teamId: number): Week | undefined => {
	const roster = rosters.find(
		(r: { cup_id: number; group: string; team_id: number }) =>
			r.cup_id === cup && r.group === group && r.team_id === teamId
	);
	if (!roster || roster.weeks.length === 0) return undefined;
	return roster.weeks.reduce((best: Week, w: Week) =>
		(w.week_start_date ?? '') > (best.week_start_date ?? '') ? w : best
	);
};
const standingOf = (cup: number, group: string, teamId: number): Standing | undefined =>
	standings
		.find((s: { cup_id: number; group: string }) => s.cup_id === cup && s.group === group)
		?.rows.find((r: Standing) => r.team_id === teamId);
const noOverflow = (page: import('@playwright/test').Page) =>
	page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);

test('當季男子組列表：現役隊整列可點', async ({ page }) => {
	await page.goto(`${BASE}/${current}/men/teams/`);
	const active = menIn(currentCup).filter((t) => t.is_active);
	await expect(page.locator('li a')).toHaveCount(active.length);
	expect(await noOverflow(page)).toBeLessThanOrEqual(0);
});

test('2024-25 男子組列表：非現役隊也整列可點', async ({ page }) => {
	await page.goto(`${BASE}/${OLD_SEASON}/men/teams/`);
	const all = menIn(OLD_CUP);
	expect(all.some((t) => !t.is_active)).toBe(true);
	await expect(page.locator('li')).toHaveCount(all.length);
	await expect(page.locator('li a')).toHaveCount(all.length);
	expect(await noOverflow(page)).toBeLessThanOrEqual(0);
});

test('2024-25 男子組球隊頁：點進非現役隊，看到名單', async ({ page }) => {
	const team = menIn(OLD_CUP).find(
		(t) => !t.is_active && latestWeekOf(OLD_CUP, 'men', t.team_id) !== undefined
	)!;
	expect(team).toBeDefined();
	await page.goto(`${BASE}/${OLD_SEASON}/men/teams/`);
	await page.locator('li a', { hasText: team.name }).click();
	await expect(page).toHaveURL(new RegExp(`/${OLD_SEASON}/men/teams/${team.team_id}/$`));
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(team.name);
	const latest = latestWeekOf(OLD_CUP, 'men', team.team_id)!;
	await expect(page.locator('ul.players li')).toHaveCount(latest.players.length);
	expect(await noOverflow(page)).toBeLessThanOrEqual(0);
});

test('球隊頁：戰績、賽事、週登錄', async ({ page }) => {
	test.skip(!rosterSeason, '所有賽季的男子組都沒有週登錄');
	const { cupId, season } = rosterSeason!;
	const team = menIn(cupId).find((t) => t.is_active && latestWeekOf(cupId, 'men', t.team_id));
	test.skip(!team, '用於比對的賽季沒有現役且有週登錄的男子組球隊');
	const teamId = team!.team_id;
	await page.goto(`${BASE}/${season}/men/teams/${teamId}/`);
	await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
	for (const name of ['戰績', '賽事', '週登錄']) {
		await expect(page.getByRole('heading', { level: 2, name })).toBeVisible();
	}

	const games = matchGroups.find(
		(g: { cup_id: number; group: string }) => g.cup_id === cupId && g.group === 'men'
	).matches;
	const mine = games.filter(
		(m: { home_team_id: number; away_team_id: number }) =>
			m.home_team_id === teamId || m.away_team_id === teamId
	);
	await expect(page.locator('.card-link')).toHaveCount(mine.length);

	const standing = standingOf(cupId, 'men', teamId);
	expect(standing).toBeDefined();
	const stats = page.locator('.stats');
	for (const [label, value] of [
		['名次', standing!.rank],
		['勝', standing!.wins],
		['敗', standing!.losses],
		['積分', standing!.points]
	] as const) {
		await expect(stats.getByText(label, { exact: true })).toBeVisible();
		await expect(stats.getByText(String(value), { exact: true }).first()).toBeVisible();
	}

	const latest = latestWeekOf(cupId, 'men', teamId);
	expect(latest).toBeDefined();
	await expect(page.locator('.week')).toContainText(latest!.week_label);
	await expect(page.locator('ul.players li')).toHaveCount(latest!.players.length);
	expect(await noOverflow(page)).toBeLessThanOrEqual(0);

	await page.locator('.card-link').first().click();
	await expect(page).toHaveURL(new RegExp(`/${season}/men/matches/\\d+/$`));
});

test('現役球隊頁頁首有隊徽圖且載入成功', async ({ page }) => {
	const active = menIn(currentCup).filter((t) => t.is_active);
	expect(active.length).toBeGreaterThan(0);
	for (const team of active) {
		await page.goto(`${BASE}/${current}/men/teams/${team.team_id}/`);
		const logo = page.locator('header .badge img');
		await expect(logo).toHaveAttribute('alt', `${team.name}隊徽`);
		expect(await logo.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0);
	}
});

test('補登球員背號為空仍有姓名卡片', async ({ page }) => {
	test.skip(!rosterSeason, '所有賽季的男子組都沒有週登錄');
	const { cupId, season } = rosterSeason!;
	const found = menIn(cupId)
		.filter((t) => t.is_active)
		.map((t) => ({ team: t, week: latestWeekOf(cupId, 'men', t.team_id) }))
		.find(({ week }) => week?.players.some((p) => p.jersey_number === null));
	test.skip(!found, '用於比對的賽季男子組最新週次沒有背號為空的球員');
	await page.goto(`${BASE}/${season}/men/teams/${found!.team.team_id}/`);
	const blank = found!.week!.players.filter((p) => p.jersey_number === null);
	const cards = page.locator('ul.players li').filter({ hasText: '—' });
	await expect(cards).toHaveCount(blank.length);
	await expect(cards.first().locator('.name')).not.toBeEmpty();
});

const seasonOnly = rosters
	.filter((r: { weeks: Week[]; season_roster: unknown[] }) => r.weeks.length === 0 && r.season_roster.length > 0)
	.map((r: { cup_id: number; group: string; team_id: number }) => ({
		season: seasonsData.seasons.find((s: { cup_id: number }) => s.cup_id === r.cup_id)?.name,
		group: r.group,
		team: teams.find((t) => t.team_id === r.team_id && t.cup_ids.includes(r.cup_id) && t.is_active)
	}))
	.find((x: { season?: string; team?: Team }) => x.season && x.team);

test('只有賽季報名名單、沒有週登錄的現役隊顯示賽季報名名單', async ({ page }) => {
	test.skip(!seasonOnly, '沒有「有賽季報名名單但沒有週登錄」的現役隊');
	await page.goto(`${BASE}/${seasonOnly!.season}/${seasonOnly!.group}/teams/${seasonOnly!.team!.team_id}/`);
	for (const name of ['戰績', '賽事', '賽季報名名單']) {
		await expect(page.getByRole('heading', { level: 2, name })).toBeVisible();
	}
	await expect(page.getByText('尚無週登錄資料')).toBeVisible();
	expect(await page.locator('ul.players li').count()).toBeGreaterThan(0);
	expect(await noOverflow(page)).toBeLessThanOrEqual(0);
});

test('不存在的球隊沒有球隊頁', async ({ page }) => {
	const res = await page.goto(`${BASE}/${OLD_SEASON}/men/teams/999/`);
	expect(res?.status()).toBe(404);
});

test('球隊頁的組別與賽季切換', async ({ page }) => {
	const team = menIn(currentCup).find((t) => t.is_active)!;
	await page.goto(`${BASE}/${current}/men/teams/${team.team_id}/`);
	await expect(
		page.getByRole('navigation', { name: '組別' }).getByRole('link', { name: '女子組' })
	).toHaveAttribute('href', new RegExp(`/${current}/women/teams/$`));
	const expected = team.cup_ids.includes(OLD_CUP)
		? `/${OLD_SEASON}/men/teams/${team.team_id}/`
		: `/${OLD_SEASON}/men/teams/`;
	const href = await page
		.getByRole('navigation', { name: '賽季' })
		.getByRole('link', { name: OLD_SEASON })
		.getAttribute('href');
	expect(href!.endsWith(expected)).toBe(true);
	await page.getByRole('navigation', { name: '組別' }).getByRole('link', { name: '女子組' }).click();
	await expect(page).toHaveURL(new RegExp(`/${current}/women/teams/$`));
	await expect(page.getByRole('heading', { level: 1 })).toContainText('女子組 球隊');
});

test('改過名的球隊頁標題用現行隊名、下方列曾用名；未改名的沒有這一行', async ({ page }) => {
	await page.goto(`${BASE}/${current}/men/teams/4/`);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText('臺北鳳凰');
	await expect(page.locator('header .former')).toHaveText('曾用名：臺北國北獅（2025-26）');
	await page.goto(`${BASE}/${current}/men/teams/5/`);
	await expect(page.locator('header .former')).toHaveText('曾用名：桃園臺灣產險（2024-25、2025-26）');
	await page.goto(`${BASE}/${current}/men/teams/1/`);
	await expect(page.locator('header .former')).toHaveCount(0);
});

test('歷史賽季球隊頁的季內賽事顯示當季隊名', async ({ page }) => {
	await page.goto(`${BASE}/2025-26/men/teams/4/`);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText('臺北鳳凰');
	const cards = page.locator('.card-link');
	await expect(cards.first()).toContainText('臺北國北獅');
	await expect(cards.filter({ hasText: '臺北鳳凰' })).toHaveCount(0);
});

test('隊徽載入失敗時落回縮寫文字，隊頁隊徽仍帶名稱', async ({ page }) => {
	await page.route('**/logos/*.png', (route) => route.fulfill({ status: 404 }));
	const team = menIn(currentCup).find((t) => t.is_active);
	expect(team).toBeDefined();
	// 縮寫可能拆成兩個 span，去掉空白再比
	const abbreviation = async (badge: Locator) => (await badge.textContent())?.replace(/\s/g, '');

	await page.goto(`${BASE}/${current}/men/teams/${team!.team_id}/`);
	const badge = page.locator('header .badge');
	await expect(badge).toHaveAttribute('role', 'img');
	await expect(badge).toHaveAttribute('aria-label', `${team!.name}隊徽`);
	expect(await abbreviation(badge)).toBe(team!.short_name);
	await expect(page.locator('.badge img')).toHaveCount(0);

	await page.goto(`${BASE}/${current}/men/standings/`);
	// 先等所有 img 都觸發 onerror 被移除，再讀文字（textContent 不會自動等待）
	await expect(page.locator('.badge img')).toHaveCount(0);
	const badges = page.locator('.badge');
	expect(await badges.count()).toBeGreaterThan(0);
	for (const b of await badges.all()) expect(await abbreviation(b)).not.toBe('');
});
