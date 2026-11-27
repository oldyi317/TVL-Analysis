import { chartOption, type ChartOption } from '#lib/chart/spec.js';
import type {
	BoxScoresData,
	MetricDistribution,
	PlayerAdvancedData,
	PlayerAdvancedEntry,
	PlayerSeason,
	PositionDistribution,
	RegularStatRow,
	StatRow
} from '#lib/data/types.js';
import { setsWon, winnerSide, type Side } from '#lib/match.js';
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
	teamSets: number,
	position: string | null = null
): T[] =>
	keys.filter(
		(key) => (teamSets > 0 || key !== 'blocks_per_set') && (position !== 'L' || !BLOCK_METRICS.has(key))
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

type HeadlineAdvanced = Pick<PlayerAdvancedEntry, 'below_threshold' | 'team_sets' | 'ratios'>;

const SCORING_RATES: ReadonlySet<string> = new Set(['attack_success_rate', 'serve_point_rate']);

const rateDetail = (key: string, numerator: number | null, denominator: number | null) =>
	`${SCORING_RATES.has(key) ? '得分' : '到位'} ${numerator}／${denominator}`;

const metricHeadline = (key: MetricKey, advanced?: HeadlineAdvanced): HeadlineStat => {
	const metric = advanced?.ratios[key];
	const value = metric?.value ?? null;
	let detail = null;
	if (metric && value !== null) {
		detail = key === 'blocks_per_set'
			? `攔網 ${metric.numerator}／球隊 ${metric.denominator} 局`
			: rateDetail(key, metric.numerator, metric.denominator);
	}
	return { label: METRIC_LABELS[key], value: formatMetricValue(key, value), detail };
};

const count = (label: string, value: number): HeadlineStat => ({
	label,
	value: String(value),
	detail: null
});

const blockPoints = (s: PlayerSeason) => count(STAT_LABELS.block_points, s.block_points);

const HEADLINES: Record<string, (s: PlayerSeason, a?: HeadlineAdvanced) => HeadlineStat[]> = {
	S: (s, a) => [
		metricHeadline('set_excellent_rate', a),
		metricHeadline('dig_excellent_rate', a),
		blockPoints(s)
	],
	L: (_, a) => [
		metricHeadline('receive_excellent_rate', a),
		metricHeadline('dig_excellent_rate', a),
		count('防守總次數', a?.ratios.defense_load.value ?? 0)
	],
	OH: (s, a) => [
		metricHeadline('attack_success_rate', a),
		metricHeadline('receive_excellent_rate', a),
		blockPoints(s)
	],
	MB: (s, a) => [
		...(a && a.team_sets > 0 ? [metricHeadline('blocks_per_set', a)] : []),
		metricHeadline('attack_success_rate', a),
		blockPoints(s)
	],
	OP: (s, a) => [metricHeadline('attack_success_rate', a), count(STAT_LABELS.attack_points, s.attack_points), blockPoints(s)]
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

export function distributionMetricKeys(
	advanced: PlayerAdvancedEntry,
	distribution: PositionDistribution,
	position: string | null
): MetricKey[] {
	return availableMetrics(METRIC_KEYS, advanced.team_sets, position).filter(
		(key) => advanced.eligible.includes(key) && advanced.ratios[key].value !== null && distribution.metrics[key].values.length > 0
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
): ChartOption {
	const tied = playerValue === null ? 0 : dist.values.filter((v) => v === playerValue).length;
	const mark = tied === 1 ? '這位球員' : `與這位球員同值（${tied} 人）`;
	return chartOption<{ isPlayer: boolean }>({
		tooltip: {
			trigger: 'item',
			lines: ({ name, value, data }) => [
				`由低到高第 ${name} 位`, `${metricLabel}：${value}`, ...(data.isPlayer ? [{ bold: mark }] : [])
			]
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
	});
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

export type GameStats = Omit<RegularStatRow, 'player_id' | 'team_id' | 'jersey_number' | 'position' | 'metrics'>;

export type GameLogEntry = {
	date: string;
	golden: boolean;
	metrics: RegularStatRow['metrics'];
	stats: GameStats;
};

export type PlayerGame = GameLogEntry & {
	gameId: number;
	opponent: string;
	side: Side;
	won: boolean | null;
	score: [number, number] | null;
	stage: string | null;
	playedGoldenSet: boolean;
};

export const toGameStats = ({
	player_id: _p, team_id: _t, jersey_number: _j, position: _pos, metrics: _metrics, ...stats
}: StatRow & Partial<Pick<RegularStatRow, 'metrics'>>): GameStats =>
	stats;

type TrendValue = { value: number | null; detail: string | null };

type TrendDef = {
	label: string;
	isRate: boolean;
	integer?: boolean;
};

const TREND_DEFS = {
	points: { label: '得分', isRate: false, integer: true },
	attack_success_rate: { label: METRIC_LABELS.attack_success_rate, isRate: true },
	serve_point_rate: { label: METRIC_LABELS.serve_point_rate, isRate: true },
	blocks_per_set: { label: METRIC_LABELS.blocks_per_set, isRate: false },
	block_points: { label: STAT_LABELS.block_points, isRate: false, integer: true },
	receive_excellent_rate: { label: METRIC_LABELS.receive_excellent_rate, isRate: true },
	dig_excellent_rate: { label: METRIC_LABELS.dig_excellent_rate, isRate: true },
	set_excellent_rate: { label: METRIC_LABELS.set_excellent_rate, isRate: true }
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

export const defaultTrendMetric = (position: string | null, teamSets: number): TrendMetric =>
	teamSets === 0 && position === 'MB' ? 'block_points' : DEFAULT_TREND[position ?? ''] ?? 'points';

const goldenPoint = (value: number) => ({ value, symbol: 'diamond', symbolSize: 14, golden: true });

export type TrendPoint = TrendValue & { date: string; heading: string; golden: boolean };

function trendValue(entry: GameLogEntry, key: TrendMetric): TrendValue {
	if (key === 'points' || key === 'block_points') {
		return { value: entry.stats[key === 'points' ? 'total_points' : key], detail: null };
	}
	const [numerator, denominator, value] = entry.metrics[key];
	if (value === null) return { value, detail: null };
	const detail = key === 'blocks_per_set'
		? `攔網 ${numerator}／${denominator} 局`
		: rateDetail(key, numerator, denominator);
	return { value, detail };
}

export function trendPoints(log: GameLogEntry[], metric: TrendMetric): TrendPoint[] {
	return log.map((e) => ({ ...trendValue(e, metric), date: e.date, heading: e.date, golden: e.golden }));
}

export function trendSummary(points: TrendPoint[], metric: TrendMetric): string {
	const title = `${trendLabel(metric)}逐場趨勢折線圖：共 ${points.length} 場`;
	const latest = points.at(-1);
	return latest
		? `${title}，最新一場 ${latest.date}：${latest.value === null ? '無數據' : formatMetricValue(metric, latest.value)}。`
		: `${title}，本季尚無單場數據。`;
}

// 球員與球隊逐場趨勢共用的折線圖
export function trendLineOption(points: TrendPoint[], metric: TrendMetric): ChartOption {
	const def: TrendDef = TREND_DEFS[metric];
	const unit = def.isRate ? '%' : '';
	return chartOption({
		tooltip: {
			trigger: 'axis',
			lines: (dataIndex) => {
				const { value, detail, heading, golden } = points[dataIndex];
				const shown = value === null ? '—' : `${value}${unit}${detail ? `（${detail}）` : ''}`;
				const lines = [heading, `${def.label}：${shown}`];
				if (golden) lines.push('此場另有黃金決勝局，未計入');
				return lines;
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
	});
}

function hasTechnicalStats({ sets_played: _sets, ...skills }: GameStats): boolean {
	return Object.values(skills).some((value) => value !== 0);
}

export function playerGameLog(
	boxScores: BoxScoresData,
	matches: MatchView[],
	cupId: number,
	group: Group,
	playerId: number
): PlayerGame[] {
	const games = boxScores.box_scores.find((b) => b.cup_id === cupId && b.group === group)?.games ?? [];
	const log: PlayerGame[] = [];
	for (const game of games) {
		const match = matches.find((m) => m.game_id === game.game_id);
		if (!match) continue;
		const homeRow = game.home.find((r) => r.player_id === playerId);
		const row = homeRow ?? game.away.find((r) => r.player_id === playerId);
		if (!row) continue;
		const side = homeRow ? 'home' : 'away';
		const winner = winnerSide(match.sets);
		const [homeSets, awaySets] = setsWon(match.sets);
		const goldenRow = game.golden_set?.home.find((r) => r.player_id === playerId)
			?? game.golden_set?.away.find((r) => r.player_id === playerId);
		log.push({
			gameId: game.game_id,
			date: match.date,
			stage: match.stage,
			side,
			opponent: (homeRow ? match.away : match.home).name,
			won: winner === null ? null : winner === side,
			score: match.sets.length === 0 ? null : homeRow ? [homeSets, awaySets] : [awaySets, homeSets],
			// 趨勢圖沿用名單標記，紀錄表只標有技術統計的黃金決勝局。
			golden: goldenRow !== undefined,
			playedGoldenSet: goldenRow !== undefined && hasTechnicalStats(toGameStats(goldenRow)),
			metrics: row.metrics,
			stats: toGameStats(row)
		});
	}
	return log.sort((a, b) => a.date.localeCompare(b.date));
}

export function playerGameLogRows(log: readonly PlayerGame[]): PlayerGame[] {
	return log
		.filter((game) => hasTechnicalStats(game.stats))
		.sort((a, b) => b.date.localeCompare(a.date) || b.gameId - a.gameId);
}
