import { render } from 'svelte/server';
import BoxScoreTable from './components/BoxScoreTable.svelte';
import { describe, expect, it } from 'vitest';
import type { BoxScoresData, MatchTopBoard, PlayerEntry, StatRow } from '#lib/data/types.js';
import { seasons, seasonTeam, teams } from '#lib/site.js';
import { detailCols, findBoxScore, goodOverTotal, toBoxScoreView, toMatchTopView } from './boxScore.js';

const stat = (player_id: number, over: Partial<StatRow> = {}): StatRow => ({
	player_id,
	team_id: 1,
	jersey_number: player_id,
	position: 'OH',
	sets_played: 3,
	attack_total: 20,
	attack_points: 10,
	block_points: 2,
	serve_total: 8,
	serve_points: 1,
	receive_total: 12,
	receive_excellent: 6,
	dig_total: 9,
	dig_excellent: 4,
	set_total: 0,
	set_excellent: 0,
	total_points: 13,
	...over
});

const player = (player_id: number, group: 'men' | 'women', name: string | null): PlayerEntry => ({
	player_id,
	group,
	name,
	height_cm: null,
	weight_kg: null,
	seasons: []
});

const data: BoxScoresData = {
	box_scores: [
		{
			cup_id: 21,
			group: 'men',
			games: [{ game_id: 5, home: [stat(1), stat(2)], away: [stat(3)], golden_set: null, top_performers: [] }]
		},
		{ cup_id: 20, group: 'men', games: [] }
	]
};

describe('findBoxScore', () => {
	it('找得到對應場次', () => {
		expect(findBoxScore(data, 21, 'men', 5)?.game_id).toBe(5);
	});

	it('場次、組別、賽季不符都回 undefined', () => {
		expect(findBoxScore(data, 21, 'men', 6)).toBeUndefined();
		expect(findBoxScore(data, 21, 'women', 5)).toBeUndefined();
		expect(findBoxScore(data, 19, 'men', 5)).toBeUndefined();
	});

	it('該季 games 為空', () => {
		expect(findBoxScore(data, 20, 'men', 5)).toBeUndefined();
	});
});

describe('toBoxScoreView', () => {
	const players = [player(1, 'men', '王小明'), player(1, 'women', '女同號'), player(2, 'men', null)];

	it('姓名以 group + player_id 對回，找不到或姓名為空用 fallback', () => {
		const game = findBoxScore(data, 21, 'men', 5)!;
		const view = toBoxScoreView(game, 'men', players);
		expect(view.home.map((r) => r.name)).toEqual(['王小明', '球員 #2']);
		expect(view.away[0].name).toBe('球員 #3');
		expect(toBoxScoreView(game, 'women', players).home[0].name).toBe('女同號');
	});

	it('背號與位置為 null 原樣保留，列順序與數值不變', () => {
		const game = {
			game_id: 9,
			home: [stat(2, { jersey_number: null, position: null, total_points: 0 }), stat(1)],
			away: [],
			golden_set: null,
			top_performers: []
		};
		const [first, second] = toBoxScoreView(game, 'men', players).home;
		expect(first).toMatchObject({ player_id: 2, number: null, position: null, total_points: 0 });
		expect(second).toMatchObject({ player_id: 1, number: 1, position: 'OH', attack_points: 10, dig_total: 9 });
	});

	it('golden_set 為 null 時 view 也是 null', () => {
		const game = findBoxScore(data, 21, 'men', 5)!;
		expect(toBoxScoreView(game, 'men', players).golden_set).toBeNull();
	});

	it('golden_set 有值時雙方各自轉換', () => {
		const game = {
			game_id: 9,
			home: [stat(1)],
			away: [stat(3)],
			golden_set: { home: [stat(1, { total_points: 4 })], away: [stat(2, { total_points: 2 })] },
			top_performers: []
		};
		const view = toBoxScoreView(game, 'men', players);
		expect(view.golden_set?.home[0]).toMatchObject({ name: '王小明', total_points: 4 });
		expect(view.golden_set?.away[0]).toMatchObject({ name: '球員 #2', total_points: 2 });
	});
});


it('單場數據表不顯示出賽局數欄與相關提示', () => {
	const view = toBoxScoreView(
		{ game_id: 9, home: [stat(1, { sets_played: null }), stat(2, { sets_played: 3 })], away: [], golden_set: null, top_performers: [] },
		'men',
		[]
	);
	expect(view.home[0]).not.toHaveProperty('sets_played');
	const html = render(BoxScoreTable, { props: { rows: view.home, label: '測試' } }).body;
	expect(html).not.toContain('局數');
	expect(html).not.toContain('官網單場數據不提供出賽局數');
	expect(html).not.toContain('>null<');
});

it('單場數據表其餘五欄的欄名用「接發」，總數為 0 顯示「0／0」', () => {
	const view = toBoxScoreView(
		{ game_id: 9, home: [stat(1)], away: [], golden_set: null, top_performers: [] },
		'men',
		[]
	);
	const html = render(BoxScoreTable, { props: { rows: view.home, label: '測試' } }).body.replace(/<!--[\s\S]*?-->/g, '');
	const headers = Array.from(html.matchAll(/<th scope="col" class="num wide[^"]*">([^<]*)<\/th>/g), (m) => m[1]);
	expect(headers).toEqual(['攻擊次數', '發球次數', '接發', '防守', '舉球']);
	expect(html).toContain('6／12');
	expect(html).toContain('4／9');
	expect(html).toContain('0／0');
});

describe('goodOverTotal 與 detailCols', () => {
	const s = {
		attack_total: 20, serve_total: 8,
		receive_excellent: 6, receive_total: 12,
		dig_excellent: 4, dig_total: 9,
		set_excellent: 0, set_total: 0
	};

	it('五欄順序與預設欄名，接發球欄名用 CONTEXT 的用詞', () => {
		const cols = detailCols();
		expect(cols.map((c) => c.label)).toEqual(['攻擊次數', '發球次數', '接發球', '防守', '舉球']);
		expect(cols.map((c) => c.value(s))).toEqual(['20', '8', '6／12', '4／9', '0／0']);
	});

	it('receiveLabel 只改接發球那一欄', () => {
		expect(detailCols({ receiveLabel: '接發' }).map((c) => c.label)).toEqual(['攻擊次數', '發球次數', '接發', '防守', '舉球']);
	});

	it('zeroTotal 只在總數為 0 時取代顯示，不給就照樣組成「0／0」', () => {
		expect(detailCols({ zeroTotal: '—' }).map((c) => c.value(s))).toEqual(['20', '8', '6／12', '4／9', '—']);
		expect(goodOverTotal(0, 0)).toBe('0／0');
		expect(goodOverTotal(0, 0, '—')).toBe('—');
		expect(goodOverTotal(3, 5, '—')).toBe('3／5');
	});
});

describe('toMatchTopView', () => {
	const menTeams = teams.filter((t) => t.group === 'men');
	const [homeTeam, awayTeam] = menTeams;
	const cupId = seasons[0].cup_id;
	const players = [
		player(1, 'men', '王小明'),
		player(2, 'men', null),
		player(3, 'men', '  '),
		player(1, 'women', '女同號')
	];
	const boards: MatchTopBoard[] = [
		{
			key: 'total_points',
			label: '得分',
			rows: [
				{ rank: 1, player_id: 1, team_id: homeTeam.team_id, value: 24 },
				{ rank: 2, player_id: 2, team_id: awayTeam.team_id, value: 18 },
				{ rank: 3, player_id: 3, team_id: homeTeam.team_id, value: 12 },
				{ rank: 3, player_id: 99, team_id: awayTeam.team_id, value: 12 }
			]
		}
	];

	it('對回當季球隊與姓名，姓名為 null、空白或查無時沿用備援名稱且不連到球員頁', () => {
		const view = toMatchTopView(boards, 'men', cupId, players);
		expect(view).toHaveLength(1);
		expect(view[0].key).toBe('total_points');
		expect(view[0].label).toBe('得分');
		expect(view[0].rows.map((r) => [r.rank, r.name, r.linked, r.team.short_name, r.value])).toEqual([
			[1, '王小明', true, seasonTeam(homeTeam, cupId).short_name, 24],
			[2, '球員 #2', false, seasonTeam(awayTeam, cupId).short_name, 18],
			[3, '球員 #3', false, seasonTeam(homeTeam, cupId).short_name, 12],
			[3, '球員 #99', false, seasonTeam(awayTeam, cupId).short_name, 12]
		]);
	});

	it('球隊不在 teams.json 時丟錯', () => {
		const broken: MatchTopBoard[] = [
			{ key: 'total_points', label: '得分', rows: [{ rank: 1, player_id: 1, team_id: -1, value: 10 }] }
		];
		expect(() => toMatchTopView(broken, 'men', cupId, players)).toThrow('team_id -1');
	});
});
