import type {
	BoxScoresData,
	MetricDistribution,
	PlayerAdvancedData,
	PlayerAdvancedEntry,
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

export const POSITION_LABELS: Record<string, string> = {
	L: '自由球員（L）',
	MB: '中間手（MB）',
	OH: '主攻手（OH）',
	OP: '副攻手（OP）',
	S: '舉球員（S）'
};

const RATE_METRICS: ReadonlySet<MetricKey> = new Set([
	'attack_success_rate',
	'receive_excellent_rate',
	'serve_point_rate',
	'dig_excellent_rate',
	'set_excellent_rate',
	'defense_efficiency'
]);

export const isRateMetric = (key: MetricKey) => RATE_METRICS.has(key);

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

export function noPrReason(
	advanced: Pick<PlayerAdvancedEntry, 'insufficient_sets' | 'position'> | null
): string | null {
	if (!advanced) return '本季沒有進階數據';
	if (advanced.insufficient_sets) return '出賽局數不足 5 局，不計算 PR 值與分佈';
	if (advanced.position === null) return '本季無登錄位置，不計算 PR 值與分佈';
	return null;
}

export type GameLogEntry = {
	date: string;
	points: number;
	attack_rate: number | null;
	golden: boolean;
};

const goldenPoint = (value: number) => ({ value, symbol: 'diamond', symbolSize: 14, golden: true });

export function trendOption(log: GameLogEntry[]): object {
	const dates = log.map((e) => `${e.date.slice(5, 7)}/${e.date.slice(8, 10)}`);
	return {
		legend: { data: ['得分', '攻擊成功率 %'] },
		tooltip: {
			trigger: 'axis',
			formatter: (params: { dataIndex: number; seriesName: string; value: number | null }[]) => {
				const i = params[0].dataIndex;
				const lines = params.map((p) => `${p.seriesName}：${p.value ?? '—'}`);
				if (log[i].golden) lines.push('此場另有黃金決勝局，未計入');
				return [log[i].date, ...lines].join('<br/>');
			}
		},
		xAxis: { type: 'category', data: dates, axisLabel: { interval: 'auto' } },
		yAxis: [
			{ type: 'value', name: '得分', minInterval: 1 },
			{ type: 'value', name: '%', min: 0, max: 100, splitLine: { show: false } }
		],
		series: [
			{
				type: 'line',
				name: '得分',
				yAxisIndex: 0,
				connectNulls: false,
				data: log.map((e) => (e.golden ? goldenPoint(e.points) : e.points))
			},
			{
				type: 'line',
				name: '攻擊成功率 %',
				yAxisIndex: 1,
				connectNulls: false,
				lineStyle: { type: 'dashed' },
				data: log.map((e) =>
					e.golden && e.attack_rate !== null ? goldenPoint(e.attack_rate) : e.attack_rate
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
		log.push({
			date: match.date,
			points: row.total_points,
			attack_rate:
				row.attack_total === 0 ? null : Math.round((row.attack_points / row.attack_total) * 1000) / 10,
			golden
		});
	}
	return log.sort((a, b) => a.date.localeCompare(b.date));
}
