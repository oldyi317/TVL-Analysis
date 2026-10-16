import { readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import { BASE, findSeason, seasonsData } from './helpers.js';

type Game = {
	game_id: number;
	golden_set: unknown;
	home: { player_id: number; jersey_number: number | null; position: string | null }[];
};
// 開季初當季還沒有單場數據，退回最近一個有數據的賽季
const boxData = JSON.parse(readFileSync('src/lib/data/box_scores.json', 'utf8'));
const picked = findSeason('boxscores');
const noBox = '所有賽季的男子組都沒有單場數據';
const games: Game[] = picked
	? boxData.box_scores.find(
			(g: { cup_id: number; group: string }) => g.cup_id === picked.cupId && g.group === 'men'
		).games
	: [];
const season = picked?.season ?? '';
const sample: Game | undefined = games[0];
const golden = games.find((g) => g.golden_set !== null);

const playersData = JSON.parse(readFileSync('src/lib/data/players.json', 'utf8'));
const firstHome = sample?.home[0];
const firstName: string = firstHome
	? (playersData.players.find(
			(p: { group: string; player_id: number }) =>
				p.group === 'men' && p.player_id === firstHome.player_id
		)?.name ?? `球員 #${firstHome.player_id}`)
	: '';

const matchesData = JSON.parse(readFileSync('src/lib/data/matches.json', 'utf8'));
const OLD_CUP = 20;
const oldSeason: string = seasonsData.seasons.find((s: { cup_id: number }) => s.cup_id === OLD_CUP).name;
const oldGame: number = matchesData.matches.find(
	(g: { cup_id: number; group: string }) => g.cup_id === OLD_CUP && g.group === 'men'
).matches[0].game_id;

const urlOf = (gameId: number) => `${BASE}/${season}/men/matches/${gameId}/`;

test.describe('手機', () => {
	test('雙方單場數據表、精簡欄位、可展開、無橫向捲動', async ({ page }) => {
		test.skip(!sample, noBox);
		await page.goto(urlOf(sample!.game_id));
		await expect(page.getByRole('table', { name: /單場數據$/ })).toHaveCount(2);

		const table = page.getByRole('table', { name: /單場數據$/ }).first();
		await expect(table.getByRole('columnheader').filter({ visible: true })).toHaveText([
			'背號',
			'球員',
			'得分',
			'攻擊',
			'攔網',
			'發球',
			'詳細'
		]);
		await expect(table.getByRole('columnheader', { name: '接發' })).toBeHidden();

		const button = page.getByRole('button', { name: /詳細數據/ }).first();
		const detailId = await button.getAttribute('aria-controls');
		await expect(async () => {
			await button.click();
			await expect(button).toHaveAttribute('aria-expanded', 'true', { timeout: 500 });
		}).toPass({ timeout: 10_000 });
		const detail = page.locator(`#${detailId}`);
		await expect(detail).toBeVisible();
		for (const text of ['接發', '防守', '舉球', '出賽局數']) {
			await expect(detail).toContainText(text);
		}
		await expect(table.locator('tbody tr.main').first().locator('.pos')).toBeVisible();

		const overflow = await page.evaluate(
			() => document.documentElement.scrollWidth - document.documentElement.clientWidth
		);
		expect(overflow).toBeLessThanOrEqual(0);
	});

	test('黃金決勝局的數據另成一段', async ({ page }) => {
		test.skip(!picked, noBox);
		test.skip(!golden, '用於比對的賽季男子組沒有黃金決勝局數據');
		await page.goto(urlOf(golden!.game_id));
		await expect(page.getByRole('heading', { name: '黃金決勝局數據' })).toBeVisible();
		await expect(page.getByRole('table', { name: /黃金決勝局數據$/ })).toHaveCount(2);
	});

	test('沒有單場數據的賽事顯示提示', async ({ page }) => {
		await page.goto(`${BASE}/${oldSeason}/men/matches/${oldGame}/`);
		await expect(page.getByText('這場沒有單場數據')).toBeVisible();
		await expect(page.getByRole('table', { name: /單場數據$/ })).toHaveCount(0);
	});
});

test.describe('桌機', () => {
	test.use({ viewport: { width: 1280, height: 800 } });

	test('完整欄位且沒有展開按鈕', async ({ page }) => {
		test.skip(!sample, noBox);
		await page.goto(urlOf(sample!.game_id));
		const tables = page.getByRole('table', { name: /單場數據$/ });
		await expect(tables).toHaveCount(2);
		for (const i of [0, 1]) {
			for (const name of ['接發', '防守', '舉球', '局數']) {
				await expect(tables.nth(i).getByRole('columnheader', { name, exact: true })).toBeVisible();
			}
		}

		const firstRow = tables.first().locator('tbody tr.main').first();
		await expect(firstRow.locator('td.num').first()).toHaveText(String(firstHome!.jersey_number ?? '—'));
		await expect(firstRow.locator('.name')).toHaveText(firstName);
		const pos: Record<string, string> = { OH: '主攻', MB: '攔中', OP: '副攻', S: '舉球', L: '自由' };
		await expect(firstRow.locator('.pos')).toHaveText(pos[firstHome!.position ?? ''] ?? '—');

		await expect(page.getByRole('button', { name: /詳細數據/ }).first()).toBeHidden();
		await expect(page.locator('tr.detail').first()).toBeHidden();
	});
});
