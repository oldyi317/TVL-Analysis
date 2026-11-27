import { readFileSync } from 'node:fs';
import type { Locator } from '@playwright/test';
import { BASE, findSeason, readJson, seasonsData, expect, test } from './helpers.js';
import { isPlayed } from '../src/lib/match.js';

type StatCell = {
	player_id: number;
	jersey_number: number | null;
	position: string | null;
	total_points: number;
	attack_points: number;
	block_points: number;
	serve_points: number;
};
type Game = {
	game_id: number;
	golden_set: unknown;
	home: StatCell[];
	away: StatCell[];
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
		await expect(page.getByRole('heading', { name: '單場前三名' })).toHaveCount(0);
	});

	test('賽後對戰比較（男子、女子、五局）：位置、表頭、數字與單場數據及逐局比分一致，375px 無橫向捲動', async ({ page }) => {
		const menGame = bothSidesGames.findLast((g) => g.group === 'men');
		const womenGame = bothSidesGames.findLast((g) => g.group === 'women');
		const fiveSetGame = bothSidesGames.findLast((g) => g.match.sets.length === 5);
		test.skip(!menGame || !womenGame || !fiveSetGame, '缺少男子、女子或打滿五局的單場數據樣本');
		await page.setViewportSize({ width: 375, height: 812 });

		const readComparisonTable = (table: Locator) =>
			table.evaluate((el) => ({
				colCount: el.querySelectorAll('thead th[scope="col"]').length,
				rows: Array.from(el.querySelectorAll('tbody tr'), (tr) => ({
					header: tr.querySelector('th[scope="row"]')!.textContent!.trim(),
					cells: Array.from(tr.querySelectorAll('td'), (td) => td.textContent!.trim())
				}))
			}));

		for (const item of [menGame!, womenGame!, fiveSetGame!]) {
			await page.goto(`${BASE}/${item.season}/${item.group}/matches/${item.game.game_id}/`);

			const [headings, sourcesData, metricsData, boxSums, setsTexts, overflow] = await Promise.all([
				page.locator('main h2').allTextContents(),
				readComparisonTable(page.getByRole('table', { name: '得分來源' })),
				readComparisonTable(page.getByRole('table', { name: '球隊指標' })),
				page.getByRole('table', { name: /單場數據$/ }).evaluateAll((tables) =>
					tables.map((table) => {
						const sums = [0, 0, 0];
						for (const tr of table.querySelectorAll('tbody tr.main')) {
							const cells = tr.querySelectorAll(':scope > td');
							sums[0] += Number(cells[3].textContent!.trim());
							sums[1] += Number(cells[4].textContent!.trim());
							sums[2] += Number(cells[5].textContent!.trim());
						}
						return sums;
					})
				),
				page.locator('.sets-table tr.total td').allTextContents(),
				page.evaluate(
					() => document.documentElement.scrollWidth - document.documentElement.clientWidth
				)
			]);

			expect(headings.slice(0, 4)).toEqual(['逐局比分', '賽後對戰比較', '單場前三名', '單場數據']);
			expect(sourcesData.rows.map((r) => r.header)).toEqual([
				'攻擊得分',
				'攔網得分',
				'發球得分',
				'對手失誤得分',
				'總得分'
			]);
			expect(metricsData.rows.map((r) => r.header)).toEqual([
				'攻擊成功率',
				'發球得分率',
				'接發球到位率',
				'防守到位率',
				'舉球到位率',
				'每局攔網'
			]);
			expect(sourcesData.colCount).toBe(3);
			expect(metricsData.colCount).toBe(3);
			expect(boxSums).toHaveLength(2);

			const [homeSums, awaySums] = boxSums;
			const setsTotals = setsTexts.map((t) => Number(t.trim()));
			const sourceByHeader = new Map(sourcesData.rows.map((r) => [r.header, r.cells]));
			const expectedRows: [string, number, number][] = [
				['攻擊得分', homeSums[0], awaySums[0]],
				['攔網得分', homeSums[1], awaySums[1]],
				['發球得分', homeSums[2], awaySums[2]],
				['總得分', setsTotals[0], setsTotals[1]]
			];
			for (const [label, homeSum, awaySum] of expectedRows) {
				expect(sourceByHeader.get(label)).toEqual([String(homeSum), String(awaySum)]);
			}

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

		// 單場前三名也只算正賽單場數據，不含黃金決勝局
		const boxTables = page.getByRole('table', { name: /單場數據$/ });
		const regularTopTotal = Math.max(
			...(await boxTables.nth(0).locator('tbody tr.main td:nth-child(3)').allTextContents()).map((t) => Number(t.trim())),
			...(await boxTables.nth(1).locator('tbody tr.main td:nth-child(3)').allTextContents()).map((t) => Number(t.trim()))
		);
		const topScoreFirst = page
			.getByRole('table', { name: '得分單場前三名', exact: true })
			.locator('tbody tr')
			.first()
			.locator('td.num');
		await expect(topScoreFirst).toHaveText(String(regularTopTotal));
	});

	test('單場前三名（男子、女子、舊賽季）：四項前三名與單場數據表一致、可連到球員頁、375px 無橫向捲動', async ({ page }) => {
		const menGame = bothSidesGames.findLast((g) => g.group === 'men');
		const womenGame = bothSidesGames.findLast((g) => g.group === 'women');
		const oldSeasonGame = bothSidesGames.find((g) => g.cupId < seasonsData.current_cup_id);
		test.skip(!menGame || !womenGame || !oldSeasonGame, '缺少男子、女子或舊賽季的單場數據樣本');
		await page.setViewportSize({ width: 375, height: 812 });

		const CATEGORIES = ['得分', '攻擊得分', '攔網得分', '發球得分'] as const;

		// 男子組放最後一場，比對完直接點球員連結，省一次重複載入
		for (const item of [womenGame!, oldSeasonGame!, menGame!]) {
			await page.goto(`${BASE}/${item.season}/${item.group}/matches/${item.game.game_id}/`);

			const [boxSides, topEntries, overflow] = await Promise.all([
				page.getByRole('table', { name: /單場數據$/ }).evaluateAll((tables) =>
					tables.map((table) => {
						const team = table.getAttribute('aria-label')!.replace(/ 單場數據$/, '');
						return Array.from(table.querySelectorAll('tbody tr.main'), (tr) => {
							const cells = tr.querySelectorAll(':scope > td');
							return {
								name: tr.querySelector('.name')!.textContent!.trim(),
								team,
								values: [2, 3, 4, 5].map((i) => Number(cells[i].textContent!.trim())) as [
									number,
									number,
									number,
									number
								]
							};
						});
					})
				),
				page.getByRole('table', { name: /單場前三名$/ }).evaluateAll((tables) =>
					tables.map(
						(table) =>
							[
								table.getAttribute('aria-label')!,
								{
									headers: Array.from(table.querySelectorAll('thead th[scope="col"]'), (th) =>
										th.textContent!.trim()
									),
									rows: Array.from(table.querySelectorAll('tbody tr'), (tr) => ({
										rank: Number(tr.querySelector('td.rank')!.textContent!.trim()),
										name: tr.querySelector('.name')!.textContent!.trim(),
										team: tr.querySelector('.team')!.textContent!.trim(),
										value: Number(tr.querySelector('td.num')!.textContent!.trim())
									}))
								}
							] as const
					)
				),
				page.evaluate(
					() => document.documentElement.scrollWidth - document.documentElement.clientWidth
				)
			]);

			expect(boxSides).toHaveLength(2);
			const allPlayers = boxSides.flat();
			const topTables = new Map(topEntries);

			for (let idx = 0; idx < CATEGORIES.length; idx++) {
				const cat = CATEGORIES[idx];
				const positive = allPlayers
					.map((p) => ({ name: p.name, team: p.team, value: p.values[idx] }))
					.filter((p) => p.value > 0)
					.sort((a, b) => b.value - a.value);
				const board = topTables.get(`${cat}單場前三名`);
				if (positive.length === 0) {
					expect(board).toBeUndefined();
					continue;
				}
				const cutoffValue = positive[Math.min(2, positive.length - 1)].value;
				const expected = positive.filter((p) => p.value >= cutoffValue);

				expect(board).toBeDefined();
				expect(board!.headers).toEqual(['名次', '球員', cat]);
				const actual = board!.rows;
				expect(actual).toHaveLength(expected.length);

				// 名次依同分並列跳號，且每組數值對應的球員集合與單場數據表一致
				expect(actual.map((a) => a.value)).toEqual(expected.map((e) => e.value));
				for (let r = 0; r < actual.length; r++) {
					const expectedRank = r > 0 && actual[r].value === actual[r - 1].value ? actual[r - 1].rank : r + 1;
					expect(actual[r].rank).toBe(expectedRank);
				}
				const keyOf = (x: { name: string; team: string; value: number }) => `${x.name}|${x.team}|${x.value}`;
				expect(actual.map(keyOf).sort()).toEqual(expected.map(keyOf).sort());
			}

			expect(overflow).toBeLessThanOrEqual(0);
		}

		// 點第一位球員名稱連到正確的球員頁
		const firstLink = page
			.getByRole('table', { name: '得分單場前三名', exact: true })
			.locator('tbody tr')
			.first()
			.getByRole('link');
		const [firstPlayerName, href] = await Promise.all([
			firstLink.locator('.name').textContent(),
			firstLink.getAttribute('href')
		]);
		expect(href).toMatch(new RegExp(`/${menGame!.season}/${menGame!.group}/players/\\d+/$`));
		await firstLink.click();
		await expect(page).toHaveURL(new RegExp(`/${menGame!.season}/${menGame!.group}/players/\\d+/$`));
		await expect(page.getByRole('heading', { level: 1 })).toHaveText(firstPlayerName!.trim());
	});

	test('單場前三名：某項目全場為 0 時該項目不顯示', async ({ page }) => {
		const zeroServeGame = bothSidesGames.find((g) => {
			const rows = [...g.game.home, ...g.game.away];
			return rows.some((r) => r.total_points > 0) && rows.every((r) => r.serve_points === 0);
		});
		test.skip(!zeroServeGame, '沒有全場發球得分皆為 0 的單場數據樣本');
		await page.goto(`${BASE}/${zeroServeGame!.season}/${zeroServeGame!.group}/matches/${zeroServeGame!.game.game_id}/`);
		await expect(page.getByRole('table', { name: '得分單場前三名', exact: true })).toBeVisible();
		await expect(page.getByRole('table', { name: '發球得分單場前三名', exact: true })).toHaveCount(0);
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
