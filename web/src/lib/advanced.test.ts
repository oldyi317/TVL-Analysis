import { describe, expect, it } from 'vitest';
import type {
	BoxScoresData,
	GameBoxScore,
	PlayerAdvancedEntry,
	PlayerSeason,
	PositionDistribution,
	StatRow
} from '#lib/data/types.js';
import type { MatchView } from '#lib/site.js';
import {
	availableMetrics,
	METRIC_KEYS,
	METRIC_LABELS,
	PR_METRICS,
	TREND_METRICS,
	defaultTrendMetric,
	distributionOption,
	distributionMetricKeys,
	distributionNote,
	distributionSummary,
	escapeHtml,
	headlineStats,
	inPrPool,
	noPrReason,
	playerGameLog,
	toGameStats,
	trendOption,
	trendPoints,
	trendSummary,
	type GameLogEntry,
	type MetricKey,
	type TrendMetric
} from './advanced.js';

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

const match = (game_id: number, date: string, home: number, away: number, sets: [number, number][] = []): MatchView =>
	({
		game_id,
		date,
		home: { team_id: home },
		away: { team_id: away },
		sets
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

	it('依日期升冪，各場保留該球員正規局的原始數字', () => {
		const data = boxes([
			game(1, { home: [stat(7, { total_points: 15 })] }),
			game(2, { away: [stat(7, { total_points: 8, attack_points: 3, attack_total: 12 })] }),
			game(3, { home: [stat(7, { total_points: 4 })] })
		]);
		const log = playerGameLog(data, matches, 21, 'men', 7);
		expect(log.map((e) => [e.date, e.golden, e.stats.total_points])).toEqual([
			['2026-01-03', false, 8],
			['2026-01-10', false, 15],
			['2026-01-17', false, 4]
		]);
		expect(log[0].stats).toEqual({
			sets_played: 3,
			attack_total: 12,
			attack_points: 3,
			block_points: 2,
			serve_total: 8,
			serve_points: 1,
			receive_total: 12,
			receive_excellent: 6,
			dig_total: 9,
			dig_excellent: 4,
			set_total: 0,
			set_excellent: 0,
			total_points: 8
		});
	});

	it('每場帶該場比分的局數，沒有比分記為 null', () => {
		const withSets = [
			match(1, '2026-01-10', 10, 20, [[25, 20], [20, 25], [15, 10]]),
			match(2, '2026-01-03', 30, 10, [])
		];
		const data = boxes([game(1, { home: [stat(7)] }), game(2, { away: [stat(7)] })]);
		expect(playerGameLog(data, withSets, 21, 'men', 7).map((e) => e.sets)).toEqual([null, 3]);
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
		expect(log.map((e) => [e.date, e.golden, e.stats.total_points])).toEqual([
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

	it('x 軸為 1..N 名次但不顯示刻度，軸名說明人數與排序；長條值為升冪原始值', () => {
		const o = distributionOption(dist, 40, '攻擊成功率') as {
			xAxis: { data: string[]; axisLabel: { show: boolean }; name: string };
		};
		expect(o.xAxis.data).toEqual(['1', '2', '3', '4', '5']);
		expect(o.xAxis.axisLabel.show).toBe(false);
		expect(o.xAxis.name).toContain('5 人');
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
	const entry = (date: string, over: Partial<StatRow> = {}, golden = false, sets: number | null = 3): GameLogEntry => ({
		date,
		golden,
		sets,
		stats: toGameStats(stat(7, over))
	});
	const log = [
		entry('2026-01-03', { total_points: 8, attack_points: 3, attack_total: 12 }),
		entry('2026-01-10', { total_points: 15, attack_points: 0, attack_total: 0 }, true),
		entry('2026-01-17', { total_points: 4, attack_points: 1, attack_total: 1 })
	];
	type Fmt = (p: { dataIndex: number }[]) => string;
	type O = {
		xAxis: { data: string[] };
		yAxis: { min?: number; max?: number; name: string };
		series: { name: string; connectNulls: boolean; data: unknown[] }[];
		tooltip: { formatter: Fmt };
	};
	const build = (metric: TrendMetric) => trendOption(log, metric) as O;

	it('x 軸為 MM/DD，只畫選定指標一條線、單一 y 軸', () => {
		const o = build('points');
		expect(o.xAxis.data).toEqual(['01/03', '01/10', '01/17']);
		expect(Array.isArray(o.yAxis)).toBe(false);
		expect(o.series.map((s) => s.name)).toEqual(['得分']);
	});

	it('比率指標取一位小數、y 軸固定 0–100；分母為 0 的場次是空點、不連線', () => {
		const o = build('attack_success_rate');
		expect(o.yAxis).toMatchObject({ min: 0, max: 100, name: '%' });
		expect(o.series[0].connectNulls).toBe(false);
		expect(o.series[0].data).toEqual([25, null, 100]);
	});

	it('計數指標 y 軸從 0 起、不設上限', () => {
		const o = build('points');
		expect(o.yAxis.min).toBe(0);
		expect(o.yAxis.max).toBeUndefined();
	});

	it('攔網得分是每場原始計數，y 軸取整數', () => {
		const blocks = [entry('2026-01-03', { block_points: 2, sets_played: null }), entry('2026-01-10', { block_points: 0 })];
		const o = trendOption(blocks, 'block_points') as O & { yAxis: { minInterval?: number } };
		expect(o.series[0].data).toEqual([2, 0]);
		expect(o.series[0].name).toBe('攔網得分');
		expect(o.yAxis.minInterval).toBe(1);
	});

	it('每局攔網以該場比分局數為分母，取兩位小數；沒有比分的場次沒有值', () => {
		const blocks = [
			entry('2026-01-03', { block_points: 2 }, false, 3),
			entry('2026-01-10', { block_points: 6 }, false, 4),
			entry('2026-01-17', { block_points: 1 }, false, null)
		];
		const o = trendOption(blocks, 'blocks_per_set') as O;
		expect(o.series[0].data).toEqual([0.67, 1.5, null]);
		expect(o.tooltip.formatter([{ dataIndex: 1 }])).toBe('2026-01-10<br/>每局攔網：1.5（攔網 6／4 局）');
	});

	it('其餘到位率與發球得分率各取對應的分子分母', () => {
		const one = [
			entry('2026-01-03', {
				serve_points: 1,
				serve_total: 4,
				receive_excellent: 3,
				receive_total: 4,
				dig_excellent: 1,
				dig_total: 8,
				set_excellent: 7,
				set_total: 10
			})
		];
		const value = (m: TrendMetric) => (trendOption(one, m) as O).series[0].data[0];
		expect(value('serve_point_rate')).toBe(25);
		expect(value('receive_excellent_rate')).toBe(75);
		expect(value('dig_excellent_rate')).toBe(12.5);
		expect(value('set_excellent_rate')).toBe(70);
	});

	it('黃金局場次的點是放大的 diamond，其他點維持原值', () => {
		const data = build('points').series[0].data;
		expect(data[0]).toBe(8);
		expect(data[2]).toBe(4);
		expect(data[1]).toMatchObject({ value: 15, symbol: 'diamond', golden: true });
		expect((data[1] as { symbolSize: number }).symbolSize).toBeGreaterThan(8);
	});

	it('tooltip 附分子／分母；分母為 0 顯示「—」；只有黃金局場次附註', () => {
		const f = build('attack_success_rate').tooltip.formatter;
		expect(f([{ dataIndex: 0 }])).toBe('2026-01-03<br/>攻擊成功率：25%（得分 3／12）');
		expect(f([{ dataIndex: 2 }])).toBe('2026-01-17<br/>攻擊成功率：100%（得分 1／1）');
		expect(f([{ dataIndex: 1 }])).toBe('2026-01-10<br/>攻擊成功率：—<br/>此場另有黃金決勝局，未計入');
		expect(build('points').tooltip.formatter([{ dataIndex: 0 }])).toBe('2026-01-03<br/>得分：8');
	});
});

describe('escapeHtml', () => {
	it('轉義 HTML 特殊字元，一般姓名與隊名原樣保留', () => {
		expect(escapeHtml(`<img src=x onerror="alert('&')">`)).toBe('&lt;img src=x onerror=&quot;alert(&#39;&amp;&#39;)&quot;&gt;');
		expect(escapeHtml('臺北鳳凰（2）Lin 12')).toBe('臺北鳳凰（2）Lin 12');
	});
});

describe('defaultTrendMetric', () => {
	it('依主要位置對應主卡第一個指標，沒有位置看得分', () => {
		expect(defaultTrendMetric('S', 0)).toBe('set_excellent_rate');
		expect(defaultTrendMetric('L', 0)).toBe('receive_excellent_rate');
		expect(defaultTrendMetric('OH', 0)).toBe('attack_success_rate');
		expect(defaultTrendMetric('MB', 0)).toBe('blocks_per_set');
		expect(defaultTrendMetric('OP', 0)).toBe('attack_success_rate');
		expect(defaultTrendMetric(null, 0)).toBe('points');
		expect(defaultTrendMetric('X', 0)).toBe('points');
	});
});

describe('noPrReason', () => {
	it('未達浮動門檻、無位置各有說明，正常為 null，沒資料另有說明', () => {
		expect(noPrReason({ below_threshold: true, position: 'OH' })).toBe(
			'技術次數未達浮動門檻，不計算 PR 值與分佈'
		);
		expect(noPrReason({ below_threshold: false, position: null })).toContain('無登錄位置');
		expect(noPrReason({ below_threshold: false, position: 'OH' })).toBeNull();
		expect(noPrReason(null)).toContain('沒有進階數據');
	});
});

describe('inPrPool', () => {
	const cases = {
		沒有資料: null,
		全部未達門檻: { below_threshold: true, position: 'OH' },
		無位置: { below_threshold: false, position: null },
		正常: { below_threshold: false, position: 'OH' }
	};
	it('只有至少一項達門檻且有位置才進池', () => {
		expect(inPrPool(cases.沒有資料)).toBe(false);
		expect(inPrPool(cases.全部未達門檻)).toBe(false);
		expect(inPrPool(cases.無位置)).toBe(false);
		expect(inPrPool(cases.正常)).toBe(true);
	});

	it('逐項達標：一項有 PR 值、另一項為 null 的球員仍進池', () => {
		const entry = { below_threshold: false, position: 'OH', pr: { attack_success_rate: 80, dig_excellent_rate: null } };
		expect(inPrPool(entry)).toBe(true);
		expect(noPrReason(entry)).toBeNull();
	});

	it('與 noPrReason 一致：noPrReason 為 null 若且唯若進池', () => {
		for (const x of Object.values(cases)) expect(noPrReason(x) === null).toBe(inPrPool(x));
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
});

const season = (over: Partial<PlayerSeason> = {}): PlayerSeason => ({
	cup_id: 21,
	team_id: 1,
	jersey_number: 7,
	position: 'S',
	season_roster_position: null,
	games: 10,
	sets_played: 36,
	attack_total: 40,
	attack_points: 18,
	block_points: 6,
	serve_total: 90,
	serve_points: 5,
	receive_total: 20,
	receive_excellent: 9,
	dig_total: 80,
	dig_excellent: 30,
	set_total: 480,
	set_excellent: 312,
	total_points: 29,
	...over
});

const advancedFor = (over: { below_threshold?: boolean; team_sets?: number; blocks_per_set?: number | null } = {}) => ({
	below_threshold: over.below_threshold ?? false,
	team_sets: over.team_sets ?? 36,
	ratios: { blocks_per_set: over.blocks_per_set === undefined ? 0.17 : over.blocks_per_set }
});

describe('headlineStats', () => {
	it('舉球：舉球到位率、防守到位率、攔網得分，率類附分子／分母', () => {
		expect(headlineStats(season()).items).toEqual([
			{ label: '舉球到位率', value: '65.0%', detail: '到位 312／480' },
			{ label: '防守到位率', value: '37.5%', detail: '到位 30／80' },
			{ label: '攔網得分', value: '6', detail: null }
		]);
	});

	it('分母為 0 的率顯示「—」，不顯示 0%', () => {
		const [, dig] = headlineStats(season({ dig_total: 0, dig_excellent: 0 })).items;
		expect(dig).toEqual({ label: '防守到位率', value: '—', detail: null });
	});

	const attackRate = {
		label: '攻擊成功率',
		value: '45.0%',
		detail: '得分 18／40'
	};
	const receiveRate = {
		label: '接發球到位率',
		value: '45.0%',
		detail: '到位 9／20'
	};

	it.each([
		[
			'L',
			[
				receiveRate,
				{ label: '防守到位率', value: '37.5%', detail: '到位 30／80' },
				{ label: '防守總次數', value: '100', detail: null }
			]
		],
		['OH', [attackRate, receiveRate, { label: '攔網得分', value: '6', detail: null }]],
		[
			'MB',
			[attackRate, { label: '攔網得分', value: '6', detail: null }]
		],
		[
			'OP',
			[
				attackRate,
				{ label: '攻擊得分', value: '18', detail: null },
				{ label: '攔網得分', value: '6', detail: null }
			]
		]
	])('%s 的主卡指標', (position, expected) => {
		expect(headlineStats(season({ position })).items).toEqual(expected);
	});

	it.each(['S', 'OH', 'OP'])('%s 的攔網卡不受球隊已打局數影響，照樣顯示攔網得分', (position) => {
		const items = headlineStats(season({ position }), advancedFor({ team_sets: 0 })).items;
		expect(items).toHaveLength(3);
		expect(items[2]).toEqual({ label: '攔網得分', value: '6', detail: null });
	});

	it('沒有主要位置時依賽季報名名單位置選主卡', () => {
		expect(headlineStats(season({ position: null, season_roster_position: 'L' })).items[0]).toEqual(
			receiveRate
		);
	});

	it('主要位置與報名名單位置都沒有時退回攻擊／攔網／發球得分', () => {
		expect(headlineStats(season({ position: null })).items).toEqual([
			{ label: '攻擊得分', value: '18', detail: null },
			{ label: '攔網得分', value: '6', detail: null },
			{ label: '發球得分', value: '5', detail: null }
		]);
	});

	it('攔網得分除以球隊已打局數，詳情寫攔網與球隊 N 局', () => {
		const items = headlineStats(season({ position: 'MB' }), advancedFor({ team_sets: 36, blocks_per_set: 0.17 })).items;
		expect(items[0]).toEqual({ label: '每局攔網', value: '0.17', detail: '攔網 6／球隊 36 局' });
	});

	it('球隊已打局數為 0、沒有進階數據時不放每局攔網', () => {
		const labels = (a: ReturnType<typeof advancedFor> | undefined) =>
			headlineStats(season({ position: 'MB' }), a).items.map((i) => i.label);
		expect(labels(advancedFor({ team_sets: 0, blocks_per_set: null }))).not.toContain('每局攔網');
		expect(labels(undefined)).not.toContain('每局攔網');
	});

	it('樣本少由 below_threshold 決定，沒有進階數據時不標', () => {
		expect(headlineStats(season({ sets_played: 36 }), advancedFor({ below_threshold: true })).fewSets).toBe(true);
		expect(headlineStats(season({ games: 1, sets_played: null }), advancedFor({ below_threshold: false })).fewSets).toBe(false);
		expect(headlineStats(season({ games: 1 })).fewSets).toBe(false);
	});
});

it('每局攔網在球隊已打局數無值時，分佈、PR 與趨勢選項移除它，MB 預設改看攔網得分', () => {
	for (const keys of [METRIC_KEYS, PR_METRICS, TREND_METRICS]) {
		expect(availableMetrics(keys, null)).not.toContain('blocks_per_set');
		expect(availableMetrics(keys, 0.17)).toContain('blocks_per_set');
	}
	expect(defaultTrendMetric('MB', null)).toBe('block_points');
	expect(defaultTrendMetric('MB', 0.17)).toBe('blocks_per_set');
});

it('場均攔網與攔網得分不需球隊已打局數，沒有每局攔網時仍可選', () => {
	expect(availableMetrics(METRIC_KEYS, null)).toContain('blocks_per_game');
	expect(availableMetrics(PR_METRICS, null)).toContain('blocks_per_game');
	expect(availableMetrics(TREND_METRICS, null)).toContain('block_points');
});

it('自由（L）的進階數據不出現任何攔網指標', () => {
	const blockKeys = ['blocks_per_set', 'blocks_per_game', 'block_points'];
	for (const keys of [METRIC_KEYS, PR_METRICS, TREND_METRICS] as readonly string[][]) {
		for (const sets of [0.17, null]) {
			expect(availableMetrics(keys, sets, 'L').filter((k) => blockKeys.includes(k))).toEqual([]);
			expect(availableMetrics(keys, sets, 'MB').some((k) => blockKeys.includes(k))).toBe(true);
		}
	}
});

it('分佈與 PR 指標的標籤與晶片順序維持原樣', () => {
	expect(METRIC_KEYS.map((k) => METRIC_LABELS[k])).toEqual([
		'攻擊成功率',
		'接發球到位率',
		'發球得分率',
		'防守到位率',
		'場均得分',
		'舉球到位率',
		'每局攔網',
		'場均攔網',
		'防守負荷',
		'防守效率'
	]);
});


describe('分佈圖數字摘要', () => {
	it('說明本人值、由低到高的同值位置範圍與中位數', () => {
		expect(distributionSummary({ values: [20, 40, 40, 80], median: 40 }, 40, 'attack_success_rate')).toBe(
			'攻擊成功率：本人 40%；同位置 4 人，由低到高第 2–3 位（同值 2 人）；中位數 40%。'
		);
	});
});


it('分佈摘要區分缺值、零值與空分佈', () => {
	expect(distributionSummary({ values: [0, 10], median: 5 }, 0, 'points_per_game')).toBe(
		'場均得分：本人 0；同位置 2 人，由低到高第 1 位；中位數 5。'
	);
	expect(distributionSummary({ values: [], median: null }, null, 'attack_success_rate')).toBe(
		'攻擊成功率：本人 無數據；同位置 0 人，本人無組內位置；中位數 無數據。'
	);
});


it('趨勢摘要包含場數與最新一場，不以舊有效值代替最新缺值', () => {
	const log: GameLogEntry[] = [
		{ date: '2026-01-01', golden: false, sets: 3, stats: toGameStats(stat(1)) },
		{ date: '2026-01-02', golden: true, sets: 3, stats: toGameStats(stat(1, { attack_total: 0, attack_points: 0 })) }
	];
	const points = trendPoints(log, 'attack_success_rate');
	expect(points.map((p) => p.value)).toEqual([50, null]);
	expect(trendSummary(points, 'attack_success_rate')).toBe(
		'攻擊成功率逐場趨勢折線圖：共 2 場，最新一場 2026-01-02：無數據。'
	);
	expect(trendSummary(trendPoints(log, 'points'), 'points')).toBe(
		'得分逐場趨勢折線圖：共 2 場，最新一場 2026-01-02：13。'
	);
	expect(trendSummary([], 'points')).toBe('得分逐場趨勢折線圖：共 0 場，本季尚無單場數據。');
});

describe('分佈指標與比較人數', () => {
	const dist = (counts: Partial<Record<MetricKey, number[]>>) => ({
		cup_id: 21,
		group: 'men' as const,
		position: 'OH',
		metrics: Object.fromEntries(
			METRIC_KEYS.map((k) => [k, { median: 1, values: counts[k] ?? [] }])
		) as PositionDistribution['metrics']
	});
	const adv = (pr: Record<string, number | null>, ratios: Record<string, number | null>) =>
		({ position: 'OH', pr, ratios }) as unknown as PlayerAdvancedEntry;

	it('只列本人有值、且該項分佈有人納入比較的指標', () => {
		const advanced = adv(
			{ attack_success_rate: 80, dig_excellent_rate: null },
			{ attack_success_rate: 45, dig_excellent_rate: 30, points_per_game: 5, defense_load: 9, blocks_per_set: null }
		);
		const d = dist({
			attack_success_rate: [30, 45],
			dig_excellent_rate: [20, 30],
			points_per_game: [5],
			defense_load: []
		});
		expect(distributionMetricKeys(advanced, d, 'OH')).toEqual(['attack_success_rate', 'points_per_game']);
	});

	it('場均得分與防守負荷以 blocks_per_game 的 PR 是否達標為準', () => {
		const ratios = { attack_success_rate: 45, points_per_game: 5, defense_load: 9, blocks_per_set: null };
		const d = dist({ points_per_game: [5], defense_load: [9] });
		const unmet = adv({ blocks_per_game: null }, ratios);
		expect(distributionMetricKeys(unmet, d, 'OH')).toEqual([]);
		const met = adv({ blocks_per_game: 50 }, ratios);
		expect(distributionMetricKeys(met, d, 'OH')).toEqual(['points_per_game', 'defense_load']);
	});

	it('比較人數取自所選指標的 values 長度', () => {
		expect(distributionNote('攻擊手', 12)).toBe('與本季同組別 攻擊手 12 位球員比較');
	});
});
