import { readFileSync } from 'node:fs';
import { BASE, findSeason, readJson, seasonsData, expect, test } from './helpers.js';
import { isPlayed } from '../src/lib/match.js';

type Game = {
	game_id: number;
	golden_set: unknown;
	home: { player_id: number; jersey_number: number | null; position: string | null }[];
	away: { player_id: number; jersey_number: number | null; position: string | null }[];
};
type MatchRecord = {
	game_id: number;
	date: string;
	home_team_id: number;
	away_team_id: number;
	sets: [number, number][];
	golden_set: [number, number] | null;
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
const matchMap = new Map<string, MatchRecord>();
for (const block of matchesData.matches as { cup_id: number; group: string; matches: MatchRecord[] }[]) {
	for (const m of block.matches) matchMap.set(`${block.cup_id}-${block.group}-${m.game_id}`, m);
}
const seasonName = (cupId: number) =>
	seasonsData.seasons.find((s: { cup_id: number }) => s.cup_id === cupId)?.name as string;
const bothSidesGames = (boxData.box_scores as { cup_id: number; group: string; games: Game[] }[]).flatMap((b) =>
	b.games
		.filter((g) => g.home.length > 0 && g.away.length > 0)
		.map((g) => {
			const match = matchMap.get(`${b.cup_id}-${b.group}-${g.game_id}`);
			return match ? { cupId: b.cup_id, season: seasonName(b.cup_id), group: b.group, game: g, match } : null;
		})
		.filter((x): x is NonNullable<typeof x> => x !== null && isPlayed(x.match))
);
const goldenGame = bothSidesGames.findLast((g) => g.game.golden_set !== null);
const teamAdvancedData = readJson('team_advanced').team_advanced as {
	cup_id: number;
	group: string;
	teams: { team_id: number; trend: { game_id: number }[] }[];
}[];
const boxIds = new Set<number>(boxData.box_scores.flatMap((b: { games: Game[] }) => b.games.map((g) => g.game_id)));
const noBoxGame = matchesData.matches
	.filter((m: { group: string }) => m.group === 'men')
	.flatMap((m: { cup_id: number; matches: { game_id: number; sets: unknown[] }[] }) =>
		m.matches
			.filter((g) => isPlayed(g) && !boxIds.has(g.game_id))
			.map((g) => ({ gameId: g.game_id, season: seasonName(m.cup_id) }))
	)[0];

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
		await button.click();
		await expect(button).toHaveAttribute('aria-expanded', 'true');
		const detail = page.locator(`#${detailId}`);
		await expect(detail).toBeVisible();
		for (const text of ['接發', '防守', '舉球']) {
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
		test.skip(!noBoxGame, '沒有「已開打但沒有單場數據」的男子組賽事');
		await page.goto(`${BASE}/${noBoxGame.season}/men/matches/${noBoxGame.gameId}/`);
		await expect(page.getByText('這場沒有單場數據')).toBeVisible();
		await expect(page.getByRole('table', { name: /單場數據$/ })).toHaveCount(0);
		await expect(page.getByRole('heading', { name: '賽後對戰比較' })).toHaveCount(0);
	});

	test('賽後對戰比較（男子、女子、五局）：位置、表頭、數字與單場數據及逐局比分一致，375px 無橫向捲動', async ({ page }) => {
		const menGame = bothSidesGames.findLast((g) => g.group === 'men');
		const womenGame = bothSidesGames.findLast((g) => g.group === 'women');
		const fiveSetGame = bothSidesGames.findLast((g) => g.match.sets.length === 5);
		test.skip(!menGame || !womenGame || !fiveSetGame, '缺少男子、女子或打滿五局的單場數據樣本');
		await page.setViewportSize({ width: 375, height: 812 });

		const sumColumn = async (table: import('@playwright/test').Locator, colIndex: number) => {
			const cells = await table.locator(`tbody tr.main td:nth-child(${colIndex})`).allTextContents();
			return cells.reduce((acc, text) => acc + Number(text.trim()), 0);
		};

		for (const item of [menGame!, womenGame!, fiveSetGame!]) {
			await page.goto(`${BASE}/${item.season}/${item.group}/matches/${item.game.game_id}/`);

			const headings = await page.locator('main h2').allTextContents();
			expect(headings.slice(0, 3)).toEqual(['逐局比分', '賽後對戰比較', '單場數據']);

			const sources = page.getByRole('table', { name: '得分來源' });
			const metrics = page.getByRole('table', { name: '球隊指標' });
			await expect(sources.getByRole('rowheader')).toHaveText([
				'攻擊得分',
				'攔網得分',
				'發球得分',
				'對手失誤得分',
				'總得分'
			]);
			await expect(metrics.getByRole('rowheader')).toHaveText([
				'攻擊成功率',
				'發球得分率',
				'接發球到位率',
				'防守到位率',
				'舉球到位率',
				'每局攔網'
			]);
			await expect(sources.getByRole('columnheader')).toHaveCount(3);
			await expect(metrics.getByRole('columnheader')).toHaveCount(3);

			const boxTables = page.getByRole('table', { name: /單場數據$/ });
			const homeBox = boxTables.nth(0);
			const awayBox = boxTables.nth(1);
			const setsTotalCells = page.locator('.sets-table tr.total td');
			const expectedRows: [string, number, number][] = [
				['攻擊得分', await sumColumn(homeBox, 4), await sumColumn(awayBox, 4)],
				['攔網得分', await sumColumn(homeBox, 5), await sumColumn(awayBox, 5)],
				['發球得分', await sumColumn(homeBox, 6), await sumColumn(awayBox, 6)],
				[
					'總得分',
					Number((await setsTotalCells.nth(0).textContent())!.trim()),
					Number((await setsTotalCells.nth(1).textContent())!.trim())
				]
			];
			for (const [label, homeSum, awaySum] of expectedRows) {
				const row = sources.locator('tbody tr', { has: page.getByRole('rowheader', { name: label, exact: true }) });
				await expect(row.locator('td').nth(0)).toHaveText(String(homeSum));
				await expect(row.locator('td').nth(1)).toHaveText(String(awaySum));
			}

			const overflow = await page.evaluate(
				() => document.documentElement.scrollWidth - document.documentElement.clientWidth
			);
			expect(overflow).toBeLessThanOrEqual(0);
		}
	});

	test('賽後對戰比較的六項球隊指標與球隊頁逐場趨勢一致', async ({ page }) => {
		const target = bothSidesGames.findLast((g) => g.group === 'men');
		test.skip(!target, noBox);
		const { season: targetSeason, cupId, group, game, match } = target!;
		const scope = teamAdvancedData.find((s) => s.cup_id === cupId && s.group === group);
		const teamTrend = scope?.teams.find((t) => t.team_id === match.home_team_id)?.trend ?? [];
		const trendIndex = teamTrend.findIndex((g) => g.game_id === game.game_id);
		expect(trendIndex).toBeGreaterThanOrEqual(0);

		await page.goto(`${BASE}/${targetSeason}/${group}/matches/${game.game_id}/`);
		const matchMetrics = await page
			.getByRole('table', { name: '球隊指標' })
			.locator('tbody tr td:nth-child(2)')
			.allTextContents();

		await page.goto(`${BASE}/${targetSeason}/${group}/teams/${match.home_team_id}/`);
		const trendFrame = page.getByRole('region', { name: '球隊逐場趨勢' });
		await trendFrame.getByRole('button', { name: '以表格檢視' }).click();
		const trendRow = trendFrame.getByRole('table', { name: '球隊逐場趨勢' }).locator('tbody tr').nth(trendIndex);
		const trendCells = (await trendRow.locator('td').allTextContents()).slice(1);
		expect(matchMetrics.map((t) => t.trim())).toEqual(trendCells.map((t) => t.trim()));
	});

	test('有黃金決勝局的賽事，賽後對戰比較不含黃金決勝局', async ({ page }) => {
		test.skip(!goldenGame, '沒有含黃金決勝局的單場數據樣本');
		const { season: goldenSeason, group, game } = goldenGame!;
		await page.goto(`${BASE}/${goldenSeason}/${group}/matches/${game.game_id}/`);
		const setsTotal = page.locator('.sets-table tr.total td');
		const homeRegularTotal = (await setsTotal.nth(0).textContent())!.trim();
		const awayRegularTotal = (await setsTotal.nth(1).textContent())!.trim();
		const sources = page.getByRole('table', { name: '得分來源' });
		const totalRow = sources.locator('tbody tr', { has: page.getByRole('rowheader', { name: '總得分', exact: true }) });
		await expect(totalRow.locator('td').nth(0)).toHaveText(homeRegularTotal);
		await expect(totalRow.locator('td').nth(1)).toHaveText(awayRegularTotal);
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
			for (const name of ['接發', '防守', '舉球']) {
				await expect(tables.nth(i).getByRole('columnheader', { name, exact: true })).toBeVisible();
			}
			await expect(tables.nth(i).getByRole('columnheader', { name: '局數' })).toHaveCount(0);
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
