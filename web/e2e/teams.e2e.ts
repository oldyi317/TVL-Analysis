import { expect, test } from '@playwright/test';
import {
	BASE,
	currentSeason as current,
	OLD_CUP,
	OLD_SEASON,
	readJson,
	seasonsData
} from './helpers.js';

type Team = { team_id: number; group: string; is_active: boolean; cup_ids: number[] };
const teams: Team[] = readJson('teams').teams;
const matchGroups = readJson('matches').matches;
const rosters = readJson('rosters').rosters;
const currentCup: number = seasonsData.current_cup_id;

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

test('2024-25 男子組列表：非現役隊不連結', async ({ page }) => {
	await page.goto(`${BASE}/${OLD_SEASON}/men/teams/`);
	const all = menIn(OLD_CUP);
	const inactive = all.filter((t) => !t.is_active);
	expect(inactive.length).toBeGreaterThan(0);
	await expect(page.locator('li')).toHaveCount(all.length);
	await expect(page.locator('li a')).toHaveCount(all.length - inactive.length);
	expect(await noOverflow(page)).toBeLessThanOrEqual(0);
});

test('球隊頁：戰績、賽事、週登錄', async ({ page }) => {
	await page.goto(`${BASE}/${current}/men/teams/`);
	await page.locator('li a').first().click();
	await expect(page).toHaveURL(/\/teams\/\d+\/$/);
	const teamId = Number(page.url().match(/\/teams\/(\d+)\/$/)![1]);
	await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
	for (const name of ['戰績', '賽事', '週登錄']) {
		await expect(page.getByRole('heading', { level: 2, name })).toBeVisible();
	}

	const games = matchGroups.find(
		(g: { cup_id: number; group: string }) => g.cup_id === currentCup && g.group === 'men'
	).matches;
	const mine = games.filter(
		(m: { home_team_id: number; away_team_id: number }) =>
			m.home_team_id === teamId || m.away_team_id === teamId
	);
	await expect(page.locator('.card-link')).toHaveCount(mine.length);

	const standing = standingOf(currentCup, 'men', teamId);
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

	const latest = latestWeekOf(currentCup, 'men', teamId);
	expect(latest).toBeDefined();
	await expect(page.locator('.week')).toContainText(latest!.week_label);
	await expect(page.locator('ul.players li')).toHaveCount(latest!.players.length);
	expect(await noOverflow(page)).toBeLessThanOrEqual(0);

	await page.locator('.card-link').first().click();
	await expect(page).toHaveURL(new RegExp(`/${current}/men/matches/\\d+/$`));
});

test('補登球員背號為空仍有姓名卡片', async ({ page }) => {
	const found = menIn(currentCup)
		.filter((t) => t.is_active)
		.map((t) => ({ team: t, week: latestWeekOf(currentCup, 'men', t.team_id) }))
		.find(({ week }) => week?.players.some((p) => p.jersey_number === null));
	test.skip(!found, '當季男子組最新週次沒有背號為空的球員');
	await page.goto(`${BASE}/${current}/men/teams/${found!.team.team_id}/`);
	const blank = found!.week!.players.filter((p) => p.jersey_number === null);
	const cards = page.locator('ul.players li').filter({ hasText: '—' });
	await expect(cards).toHaveCount(blank.length);
	await expect(cards.first().locator('.name')).not.toBeEmpty();
});

test('2024-25 現役隊沒有週登錄', async ({ page }) => {
	const team = menIn(OLD_CUP).find((t) => t.is_active)!;
	await page.goto(`${BASE}/${OLD_SEASON}/men/teams/${team.team_id}/`);
	for (const name of ['戰績', '賽事', '週登錄']) {
		await expect(page.getByRole('heading', { level: 2, name })).toBeVisible();
	}
	await expect(page.getByText('這一季沒有週登錄資料')).toBeVisible();
	expect(await noOverflow(page)).toBeLessThanOrEqual(0);
});

test('非現役隊與不存在的球隊沒有球隊頁', async ({ page }) => {
	const inactive = menIn(OLD_CUP).find((t) => !t.is_active)!;
	for (const id of [inactive.team_id, 999]) {
		const res = await page.goto(`${BASE}/${OLD_SEASON}/men/teams/${id}/`);
		expect(res?.status()).toBe(404);
	}
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
