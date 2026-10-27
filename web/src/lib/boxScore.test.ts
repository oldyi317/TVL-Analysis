import { render } from 'svelte/server';
import BoxScoreTable from './components/BoxScoreTable.svelte';
import { describe, expect, it } from 'vitest';
import type { BoxScoresData, PlayerEntry, StatRow } from '#lib/data/types.js';
import { findBoxScore, toBoxScoreView } from './boxScore.js';

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
			games: [{ game_id: 5, home: [stat(1), stat(2)], away: [stat(3)], golden_set: null }]
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
			home: [stat(2, { jersey_number: null, position: null, sets_played: 0, total_points: 0 }), stat(1)],
			away: [],
			golden_set: null
		};
		const [first, second] = toBoxScoreView(game, 'men', players).home;
		expect(first).toMatchObject({ player_id: 2, number: null, position: null, sets_played: 0, total_points: 0 });
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
			golden_set: { home: [stat(1, { total_points: 4 })], away: [stat(2, { total_points: 2 })] }
		};
		const view = toBoxScoreView(game, 'men', players);
		expect(view.golden_set?.home[0]).toMatchObject({ name: '王小明', total_points: 4 });
		expect(view.golden_set?.away[0]).toMatchObject({ name: '球員 #2', total_points: 2 });
	});
});


it('box score 保留未知局數並顯示破折號', () => {
	const view = toBoxScoreView({ game_id: 9, home: [stat(1, { sets_played: null })], away: [], golden_set: null }, 'men', []);
	expect(view.home[0].sets_played).toBeNull();
	const html = render(BoxScoreTable, { props: { rows: view.home, label: '測試' } }).body;
	expect(html).toMatch(/class="num wide[^"]*"[^>]*>—<\/td>/);
	expect(html).not.toContain('>null<');
	expect(html).toContain('官網單場數據不提供出賽局數');
});
