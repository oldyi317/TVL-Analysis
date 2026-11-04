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

export type ScoreSourceKey = 'attack' | 'block' | 'serve' | 'opponent_error';

export const SCORE_SOURCES: readonly { key: ScoreSourceKey; label: string }[] = [
	{ key: 'attack', label: STAT_LABELS.attack_points },
	{ key: 'block', label: STAT_LABELS.block_points },
	{ key: 'serve', label: STAT_LABELS.serve_points },
	{ key: 'opponent_error', label: '對手失誤得分' }
];

export const OPPONENT_ERROR_NOTE = '對手失誤得分是推算值：比分總分 − 技術得分';

const hasPoints = (t: TeamAdvanced) => t.points.total > 0;
const share = (t: TeamAdvanced, key: ScoreSourceKey): number => (t.points[key] / t.points.total) * 100;
const percent = (value: number): string => `${Math.round(value)}%`;

export function sourcesConclusion(c: TeamComparison): string | null {
	if (c.team.games < MIN_CONCLUSION_GAMES || !hasPoints(c.team)) return null;
	const others = c.teams.filter((t) => t.team_id !== c.team.team_id && hasPoints(t));
	if (others.length === 0) return null;
	const candidates = SCORE_SOURCES.map(({ key, label }) => {
		const mine = share(c.team, key);
		const highRank = 1 + others.filter((t) => share(t, key) > mine).length;
		const lowRank = 1 + others.filter((t) => share(t, key) < mine).length;
		const tied = others.some((t) => share(t, key) === mine);
		const fromHigh = highRank <= lowRank;
		return { label, mine, tied, fromHigh, rank: fromHigh ? highRank : lowRank };
	});
	const best = candidates.reduce((a, b) =>
		b.rank < a.rank || (b.rank === a.rank && b.fromHigh && !a.fromHigh) ? b : a
	);
	const side = best.fromHigh ? '高' : '低';
	const where = best.rank === 1 ? `最${side}` : `第 ${best.rank} ${side}`;
	return `${best.label}佔 ${percent(best.mine)}，組內${best.tied ? '並列' : ''}${where}`;
}

function sourceTeams(c: TeamComparison): TeamAdvanced[] {
	return [c.team, ...c.teams.filter((t) => t.team_id !== c.team.team_id)].filter(hasPoints);
}

export function sourcesBarOption(c: TeamComparison, teamName: (id: number) => string): object {
	const teams = sourceTeams(c);
	const tooltipText = (t: TeamAdvanced) =>
		[
			teamName(t.team_id),
			...SCORE_SOURCES.map(
				({ key, label }) => `${label}：${t.points[key]} 分（${percent(share(t, key))}）`
			),
			`比分總分：${t.points.total} 分`,
			OPPONENT_ERROR_NOTE
		].join('<br/>');
	return {
		legend: { data: SCORE_SOURCES.map((s) => s.label) },
		grid: { top: 36, bottom: 8, left: 8, right: 16, containLabel: true },
		tooltip: {
			trigger: 'axis',
			axisPointer: { type: 'shadow' },
			formatter: (params: { dataIndex: number }[]) => tooltipText(teams[params[0].dataIndex])
		},
		xAxis: { type: 'value', min: 0, max: 100, axisLabel: { formatter: '{value}%' } },
		yAxis: {
			type: 'category',
			inverse: true,
			data: teams.map((t) => teamName(t.team_id) + (t.team_id === c.team.team_id ? '（本隊）' : ''))
		},
		series: SCORE_SOURCES.map(({ key, label }) => ({
			type: 'bar',
			name: label,
			stack: 'score-sources',
			data: teams.map((t) => ({
				value: Math.round(share(t, key) * 10) / 10,
				itemStyle: { opacity: t.team_id === c.team.team_id ? 1 : 0.35 }
			}))
		}))
	};
}

export type SourcesTableRow = {
	team_id: number;
	name: string;
	isTeam: boolean;
	total: number;
	cells: { key: ScoreSourceKey; points: number; share: string }[];
};

export function sourcesTable(c: TeamComparison, teamName: (id: number) => string): SourcesTableRow[] {
	return sourceTeams(c).map((t) => ({
		team_id: t.team_id,
		name: teamName(t.team_id),
		isTeam: t.team_id === c.team.team_id,
		total: t.points.total,
		cells: SCORE_SOURCES.map(({ key }) => ({ key, points: t.points[key], share: percent(share(t, key)) }))
	}));
}
