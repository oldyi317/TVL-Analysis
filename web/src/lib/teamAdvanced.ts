import type { TeamAdvanced, TeamAdvancedData, TeamGameMetric, TeamMetrics, TeamTrendGame } from '#lib/data/types.js';
import { formatMetricValue, trendLineOption } from '#lib/advanced.js';
import { teamPalette } from '#lib/color.js';
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

export type PregameComparison = { home: TeamAdvanced; away: TeamAdvanced; teams: readonly TeamAdvanced[] };

export function pregameComparison(
	data: TeamAdvancedData,
	cupId: number,
	group: Group,
	homeId: number,
	awayId: number
): PregameComparison | undefined {
	const scope = data.team_advanced.find((s) => s.cup_id === cupId && s.group === group);
	const withGames = (id: number) => scope?.teams.find((t) => t.team_id === id && t.games > 0);
	const home = withGames(homeId);
	const away = withGames(awayId);
	return scope && home && away ? { home, away, teams: scope.teams } : undefined;
}

// 兩隊各在幾項球隊指標領先（六項都是越高越好；數值相同或任一方缺值不算）
function leads(m: PregameComparison): [number, number] {
	let home = 0;
	let away = 0;
	for (const key of TEAM_METRIC_KEYS) {
		const h = m.home.metrics[key].value;
		const a = m.away.metrics[key].value;
		if (h === null || a === null || h === a) continue;
		if (h > a) home++;
		else away++;
	}
	return [home, away];
}

export function pregameConclusion(m: PregameComparison, homeName: string, awayName: string): string | null {
	if (Math.min(m.home.games, m.away.games) < MIN_CONCLUSION_GAMES) return null;
	const [home, away] = leads(m);
	if (home === away) return home === 0 ? null : `兩隊各在 ${home} 項指標領先`;
	const sides = [
		{ name: homeName, count: home },
		{ name: awayName, count: away }
	];
	const [leader, trailer] = home > away ? sides : sides.reverse();
	const first = `${leader.name} 在 ${leader.count} 項指標領先`;
	return trailer.count === 0 ? first : `${first}，${trailer.name} 在 ${trailer.count} 項領先`;
}

type Pair<T> = { home: T; away: T };
type PregameBar = { value: number | null; actual: string };

// 六項尺度不同，每列以組內最高值為滿格 100；主隊畫在左（負值）、客隊在右
function pregameBars(m: PregameComparison, side: 'home' | 'away'): PregameBar[] {
	return TEAM_METRIC_KEYS.map((key) => {
		const value = m[side].metrics[key].value;
		const max = Math.max(0, ...m.teams.map((t) => t.metrics[key].value ?? 0));
		const length = value === null ? null : max === 0 ? 0 : Math.round((value / max) * 100);
		return {
			value: length === null ? null : side === 'home' ? -length : length,
			actual: formatMetricValue(key, value)
		};
	});
}

export function pregameBarOption(m: PregameComparison, names: Pair<string>, colors: Pair<string>): object {
	const sides = (['home', 'away'] as const).map((side) => ({ side, bars: pregameBars(m, side) }));
	const tooltipText = (row: number) =>
		[metricLabel(TEAM_METRIC_KEYS[row]), ...sides.map(({ side, bars }) => `${names[side]}：${bars[row].actual}`)].join(
			'<br/>'
		);
	return {
		legend: { data: [names.home, names.away] },
		grid: { top: 8 },
		tooltip: {
			trigger: 'axis',
			axisPointer: { type: 'shadow' },
			formatter: (params: { dataIndex: number }[]) => tooltipText(params[0].dataIndex)
		},
		// 兩側多留空間給長條末端的數值標籤，手機寬度下不與指標名稱重疊
		xAxis: { type: 'value', min: -170, max: 170, show: false },
		yAxis: { type: 'category', inverse: true, data: TEAM_METRIC_KEYS.map(metricLabel), axisTick: { show: false } },
		series: sides.map(({ side, bars }) => ({
			type: 'bar',
			name: names[side],
			stack: 'pregame',
			itemStyle: { color: teamPalette(colors[side]).chart, borderRadius: side === 'home' ? [6, 0, 0, 6] : [0, 6, 6, 0] },
			label: {
				show: true,
				position: side === 'home' ? 'left' : 'right',
				formatter: (p: { data: PregameBar }) => p.data.actual
			},
			data: bars
		}))
	};
}

type PregameCell = { value: string; rank: string };
export type PregameTableRow = { key: TeamMetricKey; label: string } & Pair<PregameCell>;

export function pregameTable(m: PregameComparison): PregameTableRow[] {
	const cell = (t: TeamAdvanced, key: TeamMetricKey): PregameCell => ({
		value: formatMetricValue(key, t.metrics[key].value),
		rank: rankText(m.teams, t, key)
	});
	return TEAM_METRIC_KEYS.map((key) => ({ key, label: metricLabel(key), home: cell(m.home, key), away: cell(m.away, key) }));
}

export const TREND_RECENT_GAMES = 3;

const trendDetail = (key: TeamMetricKey, { numerator, denominator }: TeamGameMetric): string =>
	key === 'blocks_per_set'
		? `攔網 ${numerator}／${denominator} 局`
		: `${key === 'attack_success_rate' || key === 'serve_point_rate' ? '得分' : '到位'} ${numerator}／${denominator}`;

const opponentName = (game: TeamTrendGame, teamName: (id: number) => string): string | null =>
	game.opponent_id === null ? null : teamName(game.opponent_id);

export function teamTrendOption(team: TeamAdvanced, key: TeamMetricKey, teamName: (id: number) => string): object {
	return trendLineOption(
		team.trend.map((game) => {
			const metric = game.metrics[key];
			const opponent = opponentName(game, teamName);
			return {
				value: metric.value,
				detail: metric.value === null ? null : trendDetail(key, metric),
				date: game.date,
				heading: opponent ? `${game.date} 對 ${opponent}` : game.date,
				golden: false
			};
		}),
		key
	);
}

// 近 3 場與整季都是分子總和除以分母總和，差距用未四捨五入的值計算；剛好 3 場時近 3 場就是整季，不產生
export function trendConclusion(team: TeamAdvanced, key: TeamMetricKey): string | null {
	if (team.games < MIN_CONCLUSION_GAMES || team.trend.length <= TREND_RECENT_GAMES) return null;
	const recent = team.trend.slice(-TREND_RECENT_GAMES).map((g) => g.metrics[key]);
	const recentDen = recent.reduce((n, m) => n + m.denominator, 0);
	const season = team.metrics[key];
	if (recentDen === 0 || season.denominator === 0) return null;
	const recentNum = recent.reduce((n, m) => n + m.numerator, 0);
	const isBlocks = key === 'blocks_per_set';
	const scale = isBlocks ? 1 : 100;
	const diff = (recentNum / recentDen - season.numerator / season.denominator) * scale;
	const size = isBlocks ? Math.round(Math.abs(diff) * 100) / 100 : Math.round(Math.abs(diff));
	const subject = `近 ${TREND_RECENT_GAMES} 場${metricLabel(key)}`;
	if (size === 0) return `${subject}與本季整體持平`;
	return isBlocks
		? `${subject}比本季整體${diff > 0 ? '多' : '少'} ${size} 次`
		: `${subject}比本季整體${diff > 0 ? '高' : '低'} ${size} 個百分點`;
}

export type TrendTableRow = {
	game_id: number;
	date: string;
	opponent: string;
	cells: { key: TeamMetricKey; value: string }[];
};

export function trendTable(team: TeamAdvanced, teamName: (id: number) => string): TrendTableRow[] {
	return team.trend.map((game) => ({
		game_id: game.game_id,
		date: game.date,
		opponent: opponentName(game, teamName) ?? '—',
		cells: TEAM_METRIC_KEYS.map((key) => ({ key, value: formatMetricValue(key, game.metrics[key].value) }))
	}));
}
