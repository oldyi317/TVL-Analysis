import { describe, expect, it } from 'vitest';
import type { BoxScoresData, GameBoxScore, StatRow } from '#lib/data/types.js';
import type { MatchView } from '#lib/site.js';
import { distributionOption, noPrReason, playerGameLog, trendOption, type GameLogEntry } from './advanced.js';

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

const match = (game_id: number, date: string, home: number, away: number): MatchView =>
	({
		game_id,
		date,
		home: { team_id: home },
		away: { team_id: away }
	}) as MatchView;

const game = (game_id: number, over: Partial<GameBoxScore> = {}): GameBoxScore => ({
	game_id,
	home: [],
	away: [],
	golden_set: null,
	...over
});

const boxes = (games: GameBoxScore[]): BoxScoresData => ({
	box_scores: [{ cup_id: 21, group: 'men', games }]
});

describe('playerGameLog', () => {
	const matches = [
		match(1, '2026-01-10', 10, 20),
		match(2, '2026-01-03', 30, 10),
		match(3, '2026-01-17', 10, 40)
	];

	it('依日期升冪，各場取該球員正規局的得分與攻擊成功率', () => {
		const data = boxes([
			game(1, { home: [stat(7, { total_points: 15, attack_points: 9, attack_total: 20 })] }),
			game(2, { away: [stat(7, { total_points: 8, attack_points: 3, attack_total: 12 })] }),
			game(3, { home: [stat(7, { total_points: 4, attack_points: 1, attack_total: 4 })] })
		]);
		expect(playerGameLog(data, matches, 21, 'men', 7)).toEqual([
			{ date: '2026-01-03', points: 8, attack_rate: 25, golden: false },
			{ date: '2026-01-10', points: 15, attack_rate: 45, golden: false },
			{ date: '2026-01-17', points: 4, attack_rate: 25, golden: false }
		]);
	});

	it('attack_total 為 0 時 attack_rate 為 null', () => {
		const data = boxes([game(1, { home: [stat(7, { attack_total: 0, attack_points: 0 })] })]);
		expect(playerGameLog(data, matches, 21, 'men', 7)[0].attack_rate).toBeNull();
	});

	it('別人的列、別的賽季與組別不算', () => {
		const data = boxes([game(1, { home: [stat(8)] })]);
		expect(playerGameLog(data, matches, 21, 'men', 7)).toEqual([]);
		expect(playerGameLog(data, matches, 99, 'men', 8)).toEqual([]);
		expect(playerGameLog(data, matches, 21, 'women', 8)).toEqual([]);
	});

	it('黃金局名單含這位球員才標 golden，且黃金局數據不併入', () => {
		const data = boxes([
			game(1, {
				home: [stat(7, { total_points: 15 })],
				golden_set: { home: [stat(7, { total_points: 3 })], away: [] }
			}),
			game(2, {
				away: [stat(7, { total_points: 8 })],
				golden_set: { home: [stat(99)], away: [stat(98)] }
			})
		]);
		const log = playerGameLog(data, matches, 21, 'men', 7);
		expect(log.map((e) => [e.date, e.golden, e.points])).toEqual([
			['2026-01-03', false, 8],
			['2026-01-10', true, 15]
		]);
	});

	it('球員只出現在黃金局、正規局沒有他的列 → 跳過這場', () => {
		const data = boxes([
			game(1, { golden_set: { home: [stat(7)], away: [] } })
		]);
		expect(playerGameLog(data, matches, 21, 'men', 7)).toEqual([]);
	});

	it('對不到 matches 的 game_id → 跳過', () => {
		const data = boxes([game(500, { home: [stat(7)] }), game(1, { home: [stat(7)] })]);
		expect(playerGameLog(data, matches, 21, 'men', 7).map((e) => e.date)).toEqual(['2026-01-10']);
	});
});

describe('distributionOption', () => {
	const dist = { median: 30, values: [10, 20, 30, 40, 50] };
	type Item = { value: number; isPlayer: boolean };
	const seriesOf = (o: object) => (o as { series: { data: Item[]; markLine: unknown }[] }).series[0];

	it('x 軸為 1..N 名次，長條值為升冪原始值', () => {
		const o = distributionOption(dist, 40, '攻擊成功率') as { xAxis: { data: string[] } };
		expect(o.xAxis.data).toEqual(['1', '2', '3', '4', '5']);
		expect(seriesOf(o).data.map((d) => d.value)).toEqual([10, 20, 30, 40, 50]);
	});

	it('只標記與球員數值相等的那一根', () => {
		const flags = seriesOf(distributionOption(dist, 40, '攻擊成功率')).data.map((d) => d.isPlayer);
		expect(flags).toEqual([false, false, false, true, false]);
	});

	it('多人同值時所有相等的長條都標記；playerValue 為 null 或不在池內則都不標', () => {
		const tied = { median: 20, values: [10, 20, 20, 30] };
		expect(seriesOf(distributionOption(tied, 20, 'x')).data.map((d) => d.isPlayer)).toEqual([
			false,
			true,
			true,
			false
		]);
		expect(seriesOf(distributionOption(dist, null, 'x')).data.some((d) => d.isPlayer)).toBe(false);
		expect(seriesOf(distributionOption(dist, 35, 'x')).data.some((d) => d.isPlayer)).toBe(false);
	});

	it('markLine 畫中位數橫線', () => {
		const ml = seriesOf(distributionOption(dist, 40, 'x')).markLine as {
			data: { yAxis: number }[];
			lineStyle: { type: string };
		};
		expect(ml.data).toEqual([{ yAxis: 30 }]);
		expect(ml.lineStyle.type).toBe('dashed');
	});
});

describe('trendOption', () => {
	const entry = (over: Partial<GameLogEntry>): GameLogEntry => ({
		date: '2026-01-03',
		points: 8,
		attack_rate: 25,
		golden: false,
		...over
	});
	const log = [
		entry({}),
		entry({ date: '2026-01-10', points: 15, attack_rate: null, golden: true }),
		entry({ date: '2026-01-17', points: 4, attack_rate: 50 })
	];
	type S = { name: string; yAxisIndex: number; connectNulls: boolean; data: unknown[] };
	const build = () => trendOption(log) as { xAxis: { data: string[] }; yAxis: unknown[]; series: S[] };

	it('x 軸為 MM/DD，兩條折線各用一個 y 軸', () => {
		const o = build();
		expect(o.xAxis.data).toEqual(['01/03', '01/10', '01/17']);
		expect(o.yAxis).toHaveLength(2);
		expect(o.series.map((s) => [s.name, s.yAxisIndex])).toEqual([
			['得分', 0],
			['攻擊成功率 %', 1]
		]);
	});

	it('攻擊成功率 null 的點保留成空點，不連線', () => {
		const s = build().series[1];
		expect(s.data).toEqual([25, null, 50]);
		expect(s.connectNulls).toBe(false);
	});

	it('黃金局場次的點是放大的 diamond，其他點維持原值', () => {
		const data = build().series[0].data;
		expect(data[0]).toBe(8);
		expect(data[2]).toBe(4);
		expect(data[1]).toMatchObject({ value: 15, symbol: 'diamond', golden: true });
		expect((data[1] as { symbolSize: number }).symbolSize).toBeGreaterThan(8);
	});
});

describe('noPrReason', () => {
	it('局數不足、無位置各有說明，正常為 null，沒資料另有說明', () => {
		expect(noPrReason({ insufficient_sets: true, position: 'OH' })).toContain('局數不足');
		expect(noPrReason({ insufficient_sets: false, position: null })).toContain('無登錄位置');
		expect(noPrReason({ insufficient_sets: false, position: 'OH' })).toBeNull();
		expect(noPrReason(null)).toContain('沒有進階數據');
	});
});

describe('邊界案例', () => {
	type Fmt = (p: unknown) => string;
	it('同位置只有自己一人時 tooltip 標「這位球員」', () => {
		const o = distributionOption({ median: 5, values: [5] }, 5, 'x') as { tooltip: { formatter: Fmt } };
		const text = o.tooltip.formatter({ name: '1', value: 5, data: { isPlayer: true } });
		expect(text).toContain('這位球員');
		expect(text).not.toContain('同值');
	});

	it('空分佈不丟例外，長條與 x 軸皆空', () => {
		const o = distributionOption({ median: 0, values: [] }, null, 'x') as {
			xAxis: { data: string[] };
			series: { data: unknown[] }[];
		};
		expect(o.xAxis.data).toEqual([]);
		expect(o.series[0].data).toEqual([]);
	});

	it('trend tooltip 只有黃金局場次附註', () => {
		const log: GameLogEntry[] = [
			{ date: '2026-01-03', points: 8, attack_rate: 25, golden: false },
			{ date: '2026-01-10', points: 15, attack_rate: 40, golden: true }
		];
		const f = (trendOption(log) as { tooltip: { formatter: Fmt } }).tooltip.formatter;
		const params = (dataIndex: number) => [{ dataIndex, seriesName: '得分', value: 1 }];
		expect(f(params(0))).not.toContain('此場另有黃金決勝局');
		expect(f(params(1))).toContain('此場另有黃金決勝局');
	});
});
