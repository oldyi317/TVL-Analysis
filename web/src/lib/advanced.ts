import type {
	BoxScoresData,
	MetricDistribution,
	PlayerAdvancedData,
	PlayerAdvancedEntry,
	PlayerSeason,
	PositionDistribution,
	StatRow
} from '#lib/data/types.js';
import type { Group, MatchView } from '#lib/site.js';

export type MetricKey = keyof PositionDistribution['metrics'];
export type PrKey = keyof NonNullable<PlayerAdvancedEntry['pr']>;

export const METRIC_LABELS: Record<MetricKey, string> = {
	attack_success_rate: '攻擊成功率',
	receive_excellent_rate: '接發球到位率',
	serve_point_rate: '發球得分率',
	dig_excellent_rate: '防守到位率',
	points_per_game: '場均得分',
	set_excellent_rate: '舉球到位率',
	blocks_per_set: '每局攔網',
	defense_load: '防守負荷',
	defense_efficiency: '防守效率'
};

export const PR_METRICS: readonly PrKey[] = [
	'attack_success_rate',
	'receive_excellent_rate',
	'serve_point_rate',
	'dig_excellent_rate',
	'set_excellent_rate',
	'blocks_per_set',
	'defense_efficiency'
];

export const METRIC_KEYS = Object.keys(METRIC_LABELS) as MetricKey[];

const RATE_METRICS: ReadonlySet<MetricKey> = new Set([
	'attack_success_rate',
	'receive_excellent_rate',
	'serve_point_rate',
	'dig_excellent_rate',
	'set_excellent_rate',
	'defense_efficiency'
]);

export const isRateMetric = (key: MetricKey) => RATE_METRICS.has(key);

export type HeadlineStat = {
	label: string;
	value: string;
	detail: string | null;
};

const rate = (label: string, good: number, total: number, goodWord: string): HeadlineStat =>
	total === 0
		? { label, value: '—', detail: null }
		: {
				label,
				value: `${((good / total) * 100).toFixed(1)}%`,
				detail: `${goodWord} ${good}／${total}`
			};

const excellentRate = (label: string, good: number, total: number) =>
	rate(label, good, total, '到位');

const count = (label: string, value: number): HeadlineStat => ({
	label,
	value: String(value),
	detail: null
});

const attackRate = (s: PlayerSeason) => rate('攻擊成功率', s.attack_points, s.attack_total, '得分');

const blocksPerSet = (s: PlayerSeason): HeadlineStat =>
	s.sets_played === 0
		? { label: '每局攔網', value: '—', detail: null }
		: {
				label: '每局攔網',
				value: (s.block_points / s.sets_played).toFixed(2),
				detail: `攔網 ${s.block_points}／${s.sets_played} 局`
			};

const HEADLINES: Record<string, (s: PlayerSeason) => HeadlineStat[]> = {
	S: (s) => [
		excellentRate('舉球到位率', s.set_excellent, s.set_total),
		count('舉球次數', s.set_total),
		excellentRate('防守到位率', s.dig_excellent, s.dig_total)
	],
	L: (s) => [
		excellentRate('接發球到位率', s.receive_excellent, s.receive_total),
		excellentRate('防守到位率', s.dig_excellent, s.dig_total),
		count('防守總次數', s.receive_total + s.dig_total)
	],
	OH: (s) => [
		attackRate(s),
		excellentRate('接發球到位率', s.receive_excellent, s.receive_total),
		count('攻擊得分', s.attack_points)
	],
	MB: (s) => [blocksPerSet(s), attackRate(s), count('攔網得分', s.block_points)],
	OP: (s) => [attackRate(s), count('攻擊得分', s.attack_points), count('發球得分', s.serve_points)]
};

const scoringHeadlines = (s: PlayerSeason) => [
	count('攻擊得分', s.attack_points),
	count('攔網得分', s.block_points),
	count('發球得分', s.serve_points)
];

const MIN_SETS = 5;

export function headlineStats(season: PlayerSeason): {
	items: HeadlineStat[];
	fewSets: boolean;
} {
	const headlines = HEADLINES[season.position ?? ''] ?? scoringHeadlines;
	return { items: headlines(season), fewSets: season.sets_played < MIN_SETS };
}

export function findAdvanced(
	data: PlayerAdvancedData,
	cupId: number,
	group: Group,
	playerId: number
): PlayerAdvancedEntry | undefined {
	return data.players.find((p) => p.cup_id === cupId && p.group === group && p.player_id === playerId);
}

export function findDistribution(
	data: PlayerAdvancedData,
	cupId: number,
	group: Group,
	position: string
): PositionDistribution | undefined {
	return data.distributions.find(
		(d) => d.cup_id === cupId && d.group === group && d.position === position
	);
}

export function distributionOption(
	dist: MetricDistribution,
	playerValue: number | null,
	metricLabel: string
): object {
	const tied = playerValue === null ? 0 : dist.values.filter((v) => v === playerValue).length;
	const mark = tied === 1 ? '這位球員' : `與這位球員同值（${tied} 人）`;
	return {
		legend: { show: false },
		grid: { bottom: 28 },
		tooltip: {
			trigger: 'item',
			formatter: (p: { name: string; value: number; data: { isPlayer: boolean } }) =>
				`由低到高第 ${p.name} 位<br/>${metricLabel}：${p.value}${p.data.isPlayer ? `<br/><b>${mark}</b>` : ''}`
		},
		xAxis: {
			type: 'category',
			data: dist.values.map((_, i) => String(i + 1)),
			axisLabel: { show: false },
			axisTick: { show: false },
			name: `同位置球員 ${dist.values.length} 人，由低到高排列`,
			nameLocation: 'middle',
			nameGap: 12
		},
		yAxis: { type: 'value' },
		series: [
			{
				type: 'bar',
				name: metricLabel,
				barMinHeight: 2,
				data: dist.values.map((value) => {
					const isPlayer = value === playerValue;
					return { value, isPlayer, itemStyle: { opacity: isPlayer ? 1 : 0.35 } };
				}),
				markLine: {
					silent: true,
					symbol: 'none',
					lineStyle: { type: 'dashed' },
					label: { formatter: `中位數 ${dist.median}`, fontSize: 14, position: 'insideStartTop' },
					data: [{ yAxis: dist.median }]
				}
			}
		]
	};
}

type PoolFields = Pick<PlayerAdvancedEntry, 'insufficient_sets' | 'position'>;

// 與後端 player_advanced.py 的 _in_pool 同一規則：局數足夠且有登錄位置才有 PR 值與分佈
export function inPrPool(
	advanced: PoolFields | null | undefined
): advanced is PoolFields & { position: string } {
	return !!advanced && !advanced.insufficient_sets && advanced.position !== null;
}

export function noPrReason(advanced: PoolFields | null): string | null {
	if (inPrPool(advanced)) return null;
	if (!advanced) return '本季沒有進階數據';
	if (advanced.insufficient_sets) return '出賽局數不足 5 局，不計算 PR 值與分佈';
	return '本季無登錄位置，不計算 PR 值與分佈';
}

export type GameStats = Omit<StatRow, 'player_id' | 'team_id' | 'jersey_number' | 'position'>;

export type GameLogEntry = {
	date: string;
	golden: boolean;
	stats: GameStats;
};

export const toGameStats = ({ player_id: _p, team_id: _t, jersey_number: _j, position: _pos, ...stats }: StatRow): GameStats =>
	stats;

type TrendValue = { value: number | null; detail: string | null };

type TrendDef = {
	label: string;
	isRate: boolean;
	integer?: boolean;
	compute: (s: GameStats) => TrendValue;
};

const trendRate =
	(good: (s: GameStats) => number, total: (s: GameStats) => number, goodWord: string) =>
	(s: GameStats): TrendValue =>
		total(s) === 0
			? { value: null, detail: null }
			: {
					value: Math.round((good(s) / total(s)) * 1000) / 10,
					detail: `${goodWord} ${good(s)}／${total(s)}`
				};

const TREND_DEFS = {
	points: { label: '得分', isRate: false, integer: true, compute: (s) => ({ value: s.total_points, detail: null }) },
	attack_success_rate: {
		label: METRIC_LABELS.attack_success_rate,
		isRate: true,
		compute: trendRate((s) => s.attack_points, (s) => s.attack_total, '得分')
	},
	serve_point_rate: {
		label: METRIC_LABELS.serve_point_rate,
		isRate: true,
		compute: trendRate((s) => s.serve_points, (s) => s.serve_total, '得分')
	},
	blocks_per_set: {
		label: METRIC_LABELS.blocks_per_set,
		isRate: false,
		compute: (s) =>
			s.sets_played === 0
				? { value: null, detail: null }
				: {
						value: Math.round((s.block_points / s.sets_played) * 100) / 100,
						detail: `攔網 ${s.block_points}／${s.sets_played} 局`
					}
	},
	receive_excellent_rate: {
		label: METRIC_LABELS.receive_excellent_rate,
		isRate: true,
		compute: trendRate((s) => s.receive_excellent, (s) => s.receive_total, '到位')
	},
	dig_excellent_rate: {
		label: METRIC_LABELS.dig_excellent_rate,
		isRate: true,
		compute: trendRate((s) => s.dig_excellent, (s) => s.dig_total, '到位')
	},
	set_excellent_rate: {
		label: METRIC_LABELS.set_excellent_rate,
		isRate: true,
		compute: trendRate((s) => s.set_excellent, (s) => s.set_total, '到位')
	}
} satisfies Record<string, TrendDef>;

export type TrendMetric = keyof typeof TREND_DEFS;

export const TREND_METRICS = Object.keys(TREND_DEFS) as TrendMetric[];

export const trendLabel = (metric: TrendMetric) => TREND_DEFS[metric].label;

const DEFAULT_TREND: Record<string, TrendMetric> = {
	S: 'set_excellent_rate',
	L: 'receive_excellent_rate',
	OH: 'attack_success_rate',
	MB: 'blocks_per_set',
	OP: 'attack_success_rate'
};

export const defaultTrendMetric = (position: string | null): TrendMetric =>
	DEFAULT_TREND[position ?? ''] ?? 'points';

const goldenPoint = (value: number) => ({ value, symbol: 'diamond', symbolSize: 14, golden: true });

export function trendOption(log: GameLogEntry[], metric: TrendMetric): object {
	const def: TrendDef = TREND_DEFS[metric];
	const values = log.map((e) => def.compute(e.stats));
	const unit = def.isRate ? '%' : '';
	return {
		legend: { show: false },
		tooltip: {
			trigger: 'axis',
			formatter: (params: { dataIndex: number }[]) => {
				const i = params[0].dataIndex;
				const { value, detail } = values[i];
				const shown = value === null ? '—' : `${value}${unit}${detail ? `（${detail}）` : ''}`;
				const lines = [log[i].date, `${def.label}：${shown}`];
				if (log[i].golden) lines.push('此場另有黃金決勝局，未計入');
				return lines.join('<br/>');
			}
		},
		xAxis: {
			type: 'category',
			data: log.map((e) => `${e.date.slice(5, 7)}/${e.date.slice(8, 10)}`),
			axisLabel: { interval: 'auto' }
		},
		yAxis: def.isRate
			? { type: 'value', name: '%', min: 0, max: 100 }
			: { type: 'value', name: def.label, min: 0, minInterval: def.integer ? 1 : undefined },
		series: [
			{
				type: 'line',
				name: def.label,
				connectNulls: false,
				data: values.map(({ value }, i) =>
					log[i].golden && value !== null ? goldenPoint(value) : value
				)
			}
		]
	};
}

const inRows = (rows: StatRow[] | undefined, playerId: number) =>
	rows?.some((r) => r.player_id === playerId) ?? false;

export function playerGameLog(
	boxScores: BoxScoresData,
	matches: MatchView[],
	cupId: number,
	group: Group,
	playerId: number
): GameLogEntry[] {
	const games = boxScores.box_scores.find((b) => b.cup_id === cupId && b.group === group)?.games ?? [];
	const log: GameLogEntry[] = [];
	for (const game of games) {
		const match = matches.find((m) => m.game_id === game.game_id);
		if (!match) continue;
		const row = [...game.home, ...game.away].find((r) => r.player_id === playerId);
		if (!row) continue;
		const golden = game.golden_set
			? inRows(game.golden_set.home, playerId) || inRows(game.golden_set.away, playerId)
			: false;
		log.push({ date: match.date, golden, stats: toGameStats(row) });
	}
	return log.sort((a, b) => a.date.localeCompare(b.date));
}
