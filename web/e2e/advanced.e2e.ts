import { expect, test, type Page } from '@playwright/test';
import { BASE, currentSeason, findSeason, readJson } from './helpers.js';

type Adv = { player_id: number; cup_id: number; group: string; position: string | null; insufficient_sets: boolean };
type Player = { player_id: number; group: string; seasons: { cup_id: number; games: number }[] };
// 開季初當季還沒有球員數據，退回最近一個有球員的賽季
const picked = findSeason('players');
const cup: number = picked?.cupId ?? 0;
const current = picked?.season ?? currentSeason;
const advanced: Adv[] = readJson('player_advanced').players;
const players: Player[] = readJson('players').players;
type Box = { cup_id: number; group: string; games: { home: { player_id: number }[]; away: { player_id: number }[] }[] };
const boxes: Box[] = readJson('box_scores').box_scores;

const hasGameLog = (a: Adv) =>
	boxes.some(
		(b) =>
			b.cup_id === cup &&
			b.group === a.group &&
			b.games.some((g) => [...g.home, ...g.away].some((r) => r.player_id === a.player_id))
	);

const played = (a: Adv) =>
	players.some(
		(p) =>
			p.group === a.group &&
			p.player_id === a.player_id &&
			p.seasons.some((s) => s.cup_id === cup && s.games > 0)
	);
const pick = (insufficient: boolean) =>
	advanced.find(
		(a) =>
			a.cup_id === cup &&
			a.insufficient_sets === insufficient &&
			(insufficient || a.position !== null) &&
			played(a) &&
			hasGameLog(a)
	);

const open = async (page: Page, a: Adv) => {
	await page.goto(`${BASE}/${current}/${a.group}/players/${a.player_id}/`);
	const toggle = page.getByRole('button', { name: '進階數據' });
	await expect(toggle).toHaveAttribute('aria-expanded', 'false');
	await expect(async () => {
		if ((await toggle.getAttribute('aria-expanded')) !== 'true') await toggle.click();
		await expect(toggle).toHaveAttribute('aria-expanded', 'true', { timeout: 1000 });
	}).toPass({ timeout: 15000 });
};

test('進階數據預設收合，展開後有 PR 值與兩個圖表', async ({ page }) => {
	const a = pick(false);
	test.skip(!a, '用於比對的賽季沒有局數足夠且有位置的出賽球員');
	await open(page, a!);
	const table = page.getByRole('table', { name: 'PR 值' });
	await expect(table.getByRole('row', { name: /攻擊成功率/ })).toBeVisible();
	await expect(page.locator('[role="img"]')).toHaveCount(2);
	expect(
		await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
	).toBeLessThanOrEqual(0);
});

test('同位置分佈切換指標後圖表會重畫', async ({ page }) => {
	const a = pick(false);
	test.skip(!a, '用於比對的賽季沒有局數足夠且有位置的出賽球員');
	await open(page, a!);
	const canvas = page.locator('[role="img"]').first().locator('canvas').first();
	await expect(canvas).toBeVisible();
	const snapshot = () => canvas.evaluate((c: HTMLCanvasElement) => c.toDataURL());
	// 等入場動畫結束：連續兩次快照相同
	let before = await snapshot();
	await expect(async () => {
		await page.waitForTimeout(300);
		const prev = before;
		before = await snapshot();
		expect(before).toBe(prev);
	}).toPass({ timeout: 10000 });
	await page.getByRole('button', { name: '防守負荷', exact: true }).click();
	await expect(page.locator('[role="img"]').first()).toHaveAttribute('aria-label', /防守負荷/);
	await expect(async () => {
		expect(await canvas.evaluate((c: HTMLCanvasElement) => c.toDataURL())).not.toBe(before);
	}).toPass({ timeout: 5000 });
});

test('沒有 PR 值的球員（局數不足或無位置）展開後看到說明', async ({ page }) => {
	const a = advanced.find(
		(x) => x.cup_id === cup && (x.insufficient_sets || x.position === null) && played(x)
	);
	test.skip(!a, '用於比對的賽季沒有局數不足或無位置的出賽球員');
	await open(page, a!);
	const reason = a!.insufficient_sets ? '出賽局數不足 5 局' : '本季無登錄位置';
	await expect(page.getByRole('note').filter({ hasText: reason })).toBeVisible();
	await expect(page.getByRole('table', { name: 'PR 值' })).toHaveCount(0);
});
