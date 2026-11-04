import type { TeamAdvanced, TeamAdvancedData, TeamMetrics } from '#lib/data/types.js';
import { formatMetricValue } from '#lib/advanced.js';
import type { Group } from '#lib/site.js';
import { STAT_LABELS } from '#lib/statLabels.js';

export type TeamMetricKey = keyof TeamMetrics;

export const TEAM_METRIC_KEYS: readonly TeamMetricKey[] = [
	'attack_success_rate',
	'serve_point_rate',
	'receive_excellent_rate',
	'dig_excellent_rate',
	'set_excellent_rate',
	'blocks_per_set'
];

export const MIN_CONCLUSION_GAMES = 3;

export type TeamComparison = { team: TeamAdvanced; teams: readonly TeamAdvanced[] };

export function teamComparison(
	data: TeamAdvancedData,
	cupId: number,
	group: Group,
	teamId: number
): TeamComparison | undefined {
	const scope = data.team_advanced.find((s) => s.cup_id === cupId && s.group === group);
	const team = scope?.teams.find((t) => t.team_id === teamId);
	return scope && team ? { team, teams: scope.teams } : undefined;
}

export const metricLabel = (key: TeamMetricKey): string => STAT_LABELS[key];

export function rankText(teams: readonly TeamAdvanced[], team: TeamAdvanced, key: TeamMetricKey): string {
	const rank = team.metrics[key].rank;
	if (rank === null) return '—';
	const ranks = teams.map((t) => t.metrics[key].rank).filter((r) => r !== null);
	const tied = ranks.filter((r) => r === rank).length > 1;
	return `${tied ? '並列' : ''}第 ${rank} / ${ranks.length}`;
}

export function metricsConclusion(c: TeamComparison): string | null {
	if (c.team.games < MIN_CONCLUSION_GAMES) return null;
	const ranked = TEAM_METRIC_KEYS.filter((key) => c.team.metrics[key].rank !== null);
	if (ranked.length === 0) return null;
	const rank = (key: TeamMetricKey) => c.team.metrics[key].rank as number;
	const best = ranked.reduce((a, b) => (rank(b) < rank(a) ? b : a));
	const worst = ranked.reduce((a, b) => (rank(b) > rank(a) ? b : a));
	const text = (key: TeamMetricKey) => rankText(c.teams, c.team, key);
	if (rank(best) === rank(worst)) return `各項球隊指標都是${text(best)}`;
	return `${metricLabel(best)} ${text(best)}，${metricLabel(worst)} ${text(worst)}`;
}

type DotPoint = {
	value: [number, number];
	name: string;
	detail: string;
	isTeam: boolean;
	symbolSize: number;
	itemStyle: { opacity: number };
};

function rowPoints(c: TeamComparison, key: TeamMetricKey, row: number, teamName: (id: number) => string): DotPoint[] {
	const valued = c.teams.filter((t) => t.metrics[key].value !== null);
	const values = valued.map((t) => t.metrics[key].value as number);
	const min = Math.min(...values);
	const span = Math.max(...values) - min;
	return valued.map((t) => {
		const value = t.metrics[key].value as number;
		const isTeam = t.team_id === c.team.team_id;
		return {
			value: [span === 0 ? 50 : Math.round(((value - min) / span) * 100), row],
			name: teamName(t.team_id),
			detail: `${metricLabel(key)}：${formatMetricValue(key, value)}（${rankText(c.teams, t, key)}）`,
			isTeam,
			symbolSize: isTeam ? 16 : 10,
			itemStyle: { opacity: isTeam ? 1 : 0.35 }
		};
	});
}

export function metricsDotOption(c: TeamComparison, teamName: (id: number) => string): object {
	const points = TEAM_METRIC_KEYS.flatMap((key, row) => rowPoints(c, key, row, teamName));
	return {
		legend: { show: false },
		grid: { top: 8, bottom: 8 },
		tooltip: { trigger: 'item', formatter: (p: { data: DotPoint }) => `${p.data.name}<br/>${p.data.detail}` },
		xAxis: {
			type: 'value',
			min: 0,
			max: 100,
			interval: 100,
			axisLabel: { formatter: (v: number) => (v === 0 ? '組內最低' : '組內最高'), alignMinLabel: 'left', alignMaxLabel: 'right' }
		},
		yAxis: { type: 'category', data: TEAM_METRIC_KEYS.map(metricLabel), inverse: true },
		series: [
			{
				type: 'scatter',
				name: '球隊指標',
				data: [...points.filter((p) => !p.isTeam), ...points.filter((p) => p.isTeam)]
			}
		]
	};
}

export type MetricsTableRow = {
	team_id: number;
	name: string;
	isTeam: boolean;
	cells: { key: TeamMetricKey; value: string; rank: string }[];
};

export function metricsTable(c: TeamComparison, teamName: (id: number) => string): MetricsTableRow[] {
	const ordered = [c.team, ...c.teams.filter((t) => t.team_id !== c.team.team_id)];
	return ordered.map((t) => ({
		team_id: t.team_id,
		name: teamName(t.team_id),
		isTeam: t.team_id === c.team.team_id,
		cells: TEAM_METRIC_KEYS.map((key) => ({
			key,
			value: formatMetricValue(key, t.metrics[key].value),
			rank: rankText(c.teams, t, key)
		}))
	}));
}
