import type {
	BoxScoresData,
	MetricDistribution,
	PlayerAdvancedData,
	PlayerAdvancedEntry,
	PlayerSeason,
	PositionDistribution,
	StatRow
} from '#lib/data/types.js';
import { displayPosition, type Group, type MatchView } from '#lib/site.js';
import { STAT_LABELS } from '#lib/statLabels.js';

export type MetricKey = keyof PositionDistribution['metrics'];
export type PrKey = keyof NonNullable<PlayerAdvancedEntry['pr']>;

export const METRIC_LABELS: Record<MetricKey, string> = {
	attack_success_rate: STAT_LABELS.attack_success_rate,
	receive_excellent_rate: STAT_LABELS.receive_excellent_rate,
	serve_point_rate: STAT_LABELS.serve_point_rate,
	dig_excellent_rate: STAT_LABELS.dig_excellent_rate,
	points_per_game: STAT_LABELS.points_per_game,
	set_excellent_rate: STAT_LABELS.set_excellent_rate,
	blocks_per_set: STAT_LABELS.blocks_per_set,
	blocks_per_game: '場均攔網',
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
	'blocks_per_game',
	'defense_efficiency'
];

export const METRIC_KEYS = Object.keys(METRIC_LABELS) as MetricKey[];

const BLOCK_METRICS: ReadonlySet<string> = new Set(['blocks_per_set', 'blocks_per_game', 'block_points']);

export const availableMetrics = <T extends string>(
	keys: readonly T[],
	blocksPerSet: number | null,
	position: string | null = null
): T[] =>
	keys.filter(
		(key) => (blocksPerSet !== null || key !== 'blocks_per_set') && (position !== 'L' || !BLOCK_METRICS.has(key))
	);

const RATE_METRICS: ReadonlySet<string> = new Set([
	'attack_success_rate',
	'receive_excellent_rate',
	'serve_point_rate',
	'dig_excellent_rate',
	'set_excellent_rate',
	'defense_efficiency'
]);

export const isRateMetric = (key: string) => RATE_METRICS.has(key);

export const formatMetricValue = (key: string, value: number | null): string =>
	value === null ? '—' : (isRateMetric(key) ? `${value}%` : String(value));

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

const blockPoints = (s: PlayerSeason) => count(STAT_LABELS.block_points, s.block_points);

const attackRate = (s: PlayerSeason) => rate(STAT_LABELS.attack_success_rate, s.attack_points, s.attack_total, '得分');

type HeadlineAdvanced = Pick<PlayerAdvancedEntry, 'below_threshold' | 'team_sets'> & {
	ratios: Pick<PlayerAdvancedEntry['ratios'], 'blocks_per_set'>;
};

const blocksPerSet = (s: PlayerSeason, a: HeadlineAdvanced | undefined): HeadlineStat[] =>
	a?.ratios.blocks_per_set == null
		? []
		: [
				{
					label: STAT_LABELS.blocks_per_set,
					value: a.ratios.blocks_per_set.toFixed(2),
					detail: `攔網 ${s.block_points}／球隊 ${a.team_sets} 局`
				}
			];

const HEADLINES: Record<string, (s: PlayerSeason, a?: HeadlineAdvanced) => HeadlineStat[]> = {
	S: (s) => [
		excellentRate(STAT_LABELS.set_excellent_rate, s.set_excellent, s.set_total),
		excellentRate(STAT_LABELS.dig_excellent_rate, s.dig_excellent, s.dig_total),
		blockPoints(s)
	],
	L: (s) => [
		excellentRate(STAT_LABELS.receive_excellent_rate, s.receive_excellent, s.receive_total),
		excellentRate(STAT_LABELS.dig_excellent_rate, s.dig_excellent, s.dig_total),
		count('防守總次數', s.receive_total + s.dig_total)
	],
	OH: (s) => [
		attackRate(s),
		excellentRate(STAT_LABELS.receive_excellent_rate, s.receive_excellent, s.receive_total),
		blockPoints(s)
	],
	MB: (s, a) => [
		...blocksPerSet(s, a),
		attackRate(s),
		blockPoints(s)
	],
	OP: (s) => [attackRate(s), count(STAT_LABELS.attack_points, s.attack_points), blockPoints(s)]
};

const scoringHeadlines = (s: PlayerSeason) => [
	count(STAT_LABELS.attack_points, s.attack_points),
	count(STAT_LABELS.block_points, s.block_points),
	count(STAT_LABELS.serve_points, s.serve_points)
];

export function headlineStats(season: PlayerSeason, advanced?: HeadlineAdvanced | null): {
	items: HeadlineStat[];
	fewSets: boolean;
} {
	const headlines = HEADLINES[displayPosition(season) ?? ''] ?? scoringHeadlines;
	return {
		items: headlines(season, advanced ?? undefined),
		fewSets: advanced?.below_threshold ?? false
	};
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

const SHARED_GATE: Partial<Record<MetricKey, string>> = {
	points_per_game: 'blocks_per_game',
	defense_load: 'blocks_per_game'
};

export function distributionMetricKeys(
	advanced: PlayerAdvancedEntry,
	distribution: PositionDistribution,
	position: string | null
): MetricKey[] {
	const pr: Partial<Record<string, number | null>> = { ...advanced.pr };
	return availableMetrics(METRIC_KEYS, advanced.ratios.blocks_per_set, position).filter(
		(key) =>
			advanced.ratios[key] !== null &&
			pr[SHARED_GATE[key] ?? key] !== null &&
			distribution.metrics[key].values.length > 0
	);
}

export const distributionNote = (positionText: string, count: number) =>
	`與本季同組別 ${positionText} ${count} 位球員比較`;

export function distributionSummary(dist: MetricDistribution, playerValue: number | null, metric: MetricKey): string {
	const first = playerValue === null ? -1 : dist.values.indexOf(playerValue);
	const last = playerValue === null ? -1 : dist.values.lastIndexOf(playerValue);
	const place = first < 0
		? '本人無組內位置'
		: first === last
			? `由低到高第 ${first + 1} 位`
			: `由低到高第 ${first + 1}–${last + 1} 位（同值 ${last - first + 1} 人）`;
	return `${METRIC_LABELS[metric]}：本人 ${playerValue === null ? '無數據' : formatMetricValue(metric, playerValue)}；同位置 ${dist.values.length} 人，${place}；中位數 ${dist.median === null ? '無數據' : formatMetricValue(metric, dist.median)}。`;
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

type PoolFields = Pick<PlayerAdvancedEntry, 'below_threshold' | 'position'>;

export function inPrPool(
	advanced: PoolFields | null | undefined
): advanced is PoolFields & { position: string } {
	return !!advanced && !advanced.below_threshold && advanced.position !== null;
}

export function noPrReason(advanced: PoolFields | null): string | null {
	if (inPrPool(advanced)) return null;
	if (!advanced) return '本季沒有進階數據';
	if (advanced.below_threshold) return '技術次數未達浮動門檻，不計算 PR 值與分佈';
	return '本季無登錄位置，不計算 PR 值與分佈';
}

export type GameStats = Omit<StatRow, 'player_id' | 'team_id' | 'jersey_number' | 'position'>;

export type GameLogEntry = {
	date: string;
	golden: boolean;
	sets: number | null;
	stats: GameStats;
};

export const toGameStats = ({ player_id: _p, team_id: _t, jersey_number: _j, position: _pos, ...stats }: StatRow): GameStats =>
	stats;

type TrendValue = { value: number | null; detail: string | null };

type TrendDef = {
	label: string;
	isRate: boolean;
	integer?: boolean;
	compute: (s: GameStats, sets: number | null) => TrendValue;
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
		compute: (s, sets) =>
			!sets
				? { value: null, detail: null }
				: {
						value: Math.round((s.block_points / sets) * 100) / 100,
						detail: `攔網 ${s.block_points}／${sets} 局`
					}
	},
	block_points: {
		label: STAT_LABELS.block_points,
		isRate: false,
		integer: true,
		compute: (s) => ({ value: s.block_points, detail: null })
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

export const defaultTrendMetric = (position: string | null, blocksPerSet: number | null): TrendMetric =>
	blocksPerSet === null && position === 'MB' ? 'block_points' : DEFAULT_TREND[position ?? ''] ?? 'points';

const goldenPoint = (value: number) => ({ value, symbol: 'diamond', symbolSize: 14, golden: true });

export type TrendPoint = TrendValue & { date: string; heading: string; golden: boolean };

export function trendPoints(log: GameLogEntry[], metric: TrendMetric): TrendPoint[] {
	const compute = TREND_DEFS[metric].compute as TrendDef['compute'];
	return log.map((e) => ({ ...compute(e.stats, e.sets), date: e.date, heading: e.date, golden: e.golden }));
}

export function trendSummary(points: TrendPoint[], metric: TrendMetric): string {
	const title = `${trendLabel(metric)}逐場趨勢折線圖：共 ${points.length} 場`;
	const latest = points.at(-1);
	return latest
		? `${title}，最新一場 ${latest.date}：${latest.value === null ? '無數據' : formatMetricValue(metric, latest.value)}。`
		: `${title}，本季尚無單場數據。`;
}

export function trendOption(log: GameLogEntry[], metric: TrendMetric): object {
	return trendLineOption(trendPoints(log, metric), metric);
}

const HTML_ESCAPES: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

// ECharts 以 innerHTML 顯示 tooltip，球員、隊名等資料字串要先轉義；不用 echarts 的 format.encodeHTML，靜態匯入會讓 Chart.svelte 的延遲載入失效
export const escapeHtml = (text: string): string => text.replace(/[&<>"']/g, (c) => HTML_ESCAPES[c]);

// 球員與球隊逐場趨勢共用的折線圖
export function trendLineOption(points: TrendPoint[], metric: TrendMetric): object {
	const def: TrendDef = TREND_DEFS[metric];
	const unit = def.isRate ? '%' : '';
	return {
		legend: { show: false },
		tooltip: {
			trigger: 'axis',
			formatter: (params: { dataIndex: number }[]) => {
				const { value, detail, heading, golden } = points[params[0].dataIndex];
				const shown = value === null ? '—' : `${value}${unit}${detail ? `（${detail}）` : ''}`;
				const lines = [escapeHtml(heading), `${def.label}：${shown}`];
				if (golden) lines.push('此場另有黃金決勝局，未計入');
				return lines.join('<br/>');
			}
		},
		xAxis: {
			type: 'category',
			data: points.map((p) => `${p.date.slice(5, 7)}/${p.date.slice(8, 10)}`),
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
				data: points.map(({ value, golden }) => (golden && value !== null ? goldenPoint(value) : value))
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
		log.push({ date: match.date, golden, sets: match.sets.length || null, stats: toGameStats(row) });
	}
	return log.sort((a, b) => a.date.localeCompare(b.date));
}
