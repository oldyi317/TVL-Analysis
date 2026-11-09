import { expect, test, type Page } from '@playwright/test';
import { BASE, currentSeason, findSeason, readJson } from './helpers.js';

type Adv = {
	player_id: number;
	cup_id: number;
	group: string;
	position: string | null;
	below_threshold: boolean;
	pr: Record<string, number | null> | null;
	ratios: { blocks_per_set: number | null };
};
type Player = { player_id: number; group: string; seasons: { cup_id: number; games: number }[] };
// 開季初當季還沒有進階數據，退回最近一個有進階數據的賽季
const picked = findSeason('advanced');
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
const pick = (below: boolean) =>
	advanced.find(
		(a) =>
			a.cup_id === cup &&
			a.below_threshold === below &&
			(below || (a.position !== null && a.pr?.attack_success_rate != null)) &&
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
	test.skip(!a, '用於比對的賽季沒有達浮動門檻且有位置的出賽球員');
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
	test.skip(!a, '用於比對的賽季沒有達浮動門檻且有位置的出賽球員');
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

test('逐場趨勢預設依主要位置選指標，可切換其他指標', async ({ page }) => {
	const a = pick(false);
	test.skip(!a, '用於比對的賽季沒有達浮動門檻且有位置的出賽球員');
	await open(page, a!);
	const defaults: Record<string, string> = {
		S: '舉球到位率',
		L: '接發球到位率',
		OH: '攻擊成功率',
		MB: a!.ratios.blocks_per_set === null ? '攔網得分' : '每局攔網',
		OP: '攻擊成功率'
	};
	const group = page.getByRole('group', { name: '選擇逐場趨勢指標' });
	const chart = page.locator('[role="img"]').last();
	const initial = defaults[a!.position!] ?? '得分';
	await expect(group.getByRole('button', { name: initial, exact: true })).toHaveAttribute('aria-pressed', 'true');
	await expect(chart).toHaveAttribute('aria-label', new RegExp(`${initial}逐場趨勢折線圖：共 `));
	const next = initial === '得分' ? '攻擊成功率' : '得分';
	await group.getByRole('button', { name: next, exact: true }).click();
	await expect(group.getByRole('button', { name: next, exact: true })).toHaveAttribute('aria-pressed', 'true');
	await expect(chart).toHaveAttribute('aria-label', new RegExp(`${next}逐場趨勢折線圖：共 `));
});

const BLOCK_LABELS = ['每局攔網', '場均攔網', '攔網得分'];
const pickAt = (match: (position: string) => boolean) =>
	advanced.find(
		(x) => x.cup_id === cup && !x.below_threshold && x.position !== null && match(x.position) && played(x) && hasGameLog(x)
	);

test('非自由（L）球員的 PR 表、分佈與趨勢可選場均攔網／攔網得分', async ({ page }) => {
	const a = pickAt((pos) => pos !== 'L');
	test.skip(!a, '用於比對的賽季沒有達浮動門檻的非自由（L）球員');
	await open(page, a!);
	const chip = (group: string, name: string) =>
		page.getByRole('group', { name: group }).getByRole('button', { name, exact: true });
	await expect(page.getByRole('table', { name: 'PR 值' }).getByRole('row', { name: /場均攔網/ })).toBeVisible();
	await expect(chip('選擇分佈指標', '場均攔網')).toBeVisible();
	await expect(chip('選擇逐場趨勢指標', '攔網得分')).toBeVisible();
});

test('自由（L）球員進階數據區不出現任何攔網指標', async ({ page }) => {
	const a = pickAt((pos) => pos === 'L');
	test.skip(!a, '用於比對的賽季沒有達浮動門檻的自由（L）球員');
	await open(page, a!);
	await expect(page.getByRole('table', { name: 'PR 值' })).toBeVisible();
	const panel = page.locator('[id$="-panel"]');
	for (const label of BLOCK_LABELS) await expect(panel.getByText(label, { exact: true })).toHaveCount(0);
});

// 文案刻意寫死、不 import noPrReason：測試要獨立於實作，才能抓到頁面文案改了測試沒跟的情況（#84）
const expectedReason = (a: Adv) => {
	if (!a.below_threshold) return '本季無登錄位置';
	return '技術次數未達浮動門檻';
};

test('沒有 PR 值的球員（未達浮動門檻或無位置）展開後看到說明', async ({ page }) => {
	const a = advanced.find(
		(x) => x.cup_id === cup && (x.below_threshold || x.position === null) && played(x)
	);
	test.skip(!a, '用於比對的賽季沒有未達浮動門檻或無位置的出賽球員');
	await open(page, a!);
	await expect(page.getByRole('note').filter({ hasText: expectedReason(a!) })).toBeVisible();
	await expect(page.getByRole('table', { name: 'PR 值' })).toHaveCount(0);
	await expect(page.getByRole('heading', { name: '同位置分佈' })).toHaveCount(0);
});

test('分佈圖提供數字摘要，鍵盤展開表格後切換指標同步更新', async ({ page }) => {
	const a = pick(false);
	test.skip(!a, '用於比對的賽季沒有可比較球員');
	await open(page, a!);
	const section = page.getByRole('region', { name: '同位置分佈', exact: true });
	const chart = section.getByRole('img');
	await expect(chart).toHaveAttribute('aria-label', /攻擊成功率：本人 .+；同位置 \d+ 人，.+；中位數/);
	const toggle = section.getByRole('button', { name: '以表格檢視' });
	await expect(toggle).toHaveAttribute('aria-expanded', 'false');
	await toggle.focus();
	await page.keyboard.press('Enter');
	await expect(toggle).toHaveAttribute('aria-expanded', 'true');
	const table = section.getByRole('table');
	await expect(table).toHaveAccessibleName('同位置分佈：攻擊成功率');
	await expect(table.getByRole('columnheader', { name: '攻擊成功率' })).toBeVisible();
	await section.getByRole('button', { name: '防守負荷', exact: true }).click();
	await expect(chart).toHaveAttribute('aria-label', /防守負荷：本人/);
	await expect(table).toHaveAccessibleName('同位置分佈：防守負荷');
	await expect(table.getByRole('columnheader', { name: '防守負荷' })).toBeVisible();
	await expect(table.getByRole('columnheader', { name: '攻擊成功率' })).toHaveCount(0);
	await toggle.click();
	await expect(toggle).toHaveAttribute('aria-expanded', 'false');
	await expect(table).toHaveCount(0);
});

test('趨勢表格包含最新一場數值，切換指標後摘要與表格同步', async ({ page }) => {
	const a = pick(false);
	test.skip(!a, '用於比對的賽季沒有可比較球員');
	await open(page, a!);
	const section = page.getByRole('region', { name: '逐場趨勢', exact: true });
	await section.getByRole('button', { name: '得分', exact: true }).click();
	const toggle = section.getByRole('button', { name: '以表格檢視' });
	await expect(toggle).toHaveAttribute('aria-expanded', 'false');
	await toggle.click();
	await expect(toggle).toHaveAttribute('aria-expanded', 'true');
	const table = section.getByRole('table');
	await expect(table).toHaveAccessibleName('逐場趨勢：得分');
	const rows = table.locator('tbody tr');
	const count = await rows.count();
	const date = await rows.last().getByRole('rowheader').innerText();
	const value = await rows.last().getByRole('cell').first().innerText();
	await expect(section.getByRole('img')).toHaveAttribute('aria-label', `得分逐場趨勢折線圖：共 ${count} 場，最新一場 ${date}：${value}。`);
	await section.getByRole('button', { name: '攻擊成功率', exact: true }).click();
	await expect(table).toHaveAccessibleName('逐場趨勢：攻擊成功率');
	await expect(table.getByRole('columnheader', { name: '攻擊成功率' })).toBeVisible();
	await expect(section.getByRole('img')).toHaveAttribute('aria-label', /攻擊成功率逐場趨勢折線圖：共 \d+ 場，最新一場/);
	expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(0);
});
