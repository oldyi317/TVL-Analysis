import logosJson from '#lib/data/logos.json';
import matchesJson from '#lib/data/matches.json';
import playersJson from '#lib/data/players.json';
import rostersJson from '#lib/data/rosters.json';
import seasonsJson from '#lib/data/seasons.json';
import standingsJson from '#lib/data/standings.json';
import teamsJson from '#lib/data/teams.json';
import type { StyleMatch } from '#lib/match.js';
import type {
	GroupStandings,
	MatchEntry,
	MatchesData,
	PlayerEntry,
	PlayerSeason,
	PlayersData,
	RosterWeek,
	RostersData,
	Season,
	SeasonsData,
	StandingsData,
	StandingsEntry,
	Team,
	TeamRoster,
	TeamsData
} from '#lib/data/types.js';

export const SITE_URL = 'https://oldyi317.github.io/TVL-Analysis';
export const SITE_DESCRIPTION =
	'台灣企業排球聯賽（TVL）球迷自製數據站：賽事比分、球隊與球員數據、賽前勝率';

export const GROUPS = ['men', 'women'] as const;
export type Group = (typeof GROUPS)[number];
export const GROUP_LABEL: Record<Group, string> = { men: '男子組', women: '女子組' };

const seasonsData = seasonsJson as SeasonsData;
export const seasons: readonly Season[] = seasonsData.seasons;
export const teams: readonly Team[] = (teamsJson as TeamsData).teams;
export const players: readonly PlayerEntry[] = (playersJson as PlayersData).players;

const current = seasons.find((s) => s.cup_id === seasonsData.current_cup_id);
if (!current) throw new Error(`seasons.json 的 current_cup_id ${seasonsData.current_cup_id} 不在賽季清單中`);
export const currentSeason: Season = current;

export function isGroup(value: string): value is Group {
	return (GROUPS as readonly string[]).includes(value);
}

export function findSeason(name: string): Season | undefined {
	return seasons.find((s) => s.name === name);
}

export type SeasonTeam = Team & { cup_id: number; logo: string | null };

export type SetRecordKey = '3-0' | '3-1' | '3-2' | '2-3' | '1-3' | '0-3';
export type StandingsRow = {
	rank: number;
	team: SeasonTeam;
	wins: number;
	losses: number;
	win_pct: number | null;
	points: number;
	sets_won: number;
	sets_lost: number;
	points_for: number;
	points_against: number;
	set_ratio: number | null;
	point_ratio: number | null;
	set_record: Record<SetRecordKey, number>;
	last5: ('W' | 'L')[];
};

const logoFiles = new Set<string>(logosJson);

// 隊徽：該季專用圖優先；沒有時，只有當季隊名仍是現行隊名才用現行圖，否則不畫
export function seasonTeam(team: Team, cupId: number): SeasonTeam {
	const names = team.names_by_cup[String(cupId)];
	const name = names?.name ?? team.name;
	const stem = `${team.group}-${team.team_id}`;
	let file: string | null = null;
	if (logoFiles.has(`${stem}-${cupId}.png`)) file = `${stem}-${cupId}.png`;
	else if (name === team.name && logoFiles.has(`${stem}.png`)) file = `${stem}.png`;
	return {
		...team,
		name,
		short_name: names?.short_name ?? team.short_name,
		cup_id: cupId,
		logo: file && `/logos/${file}`
	};
}

export type FormerName = { name: string; seasons: string[] };

// 曾用名：names_by_cup 中與現行隊名不同的季，同名合併、依賽季排序
export function formerNames(team: Team): FormerName[] {
	const result: FormerName[] = [];
	for (const season of seasons) {
		const name = team.names_by_cup[String(season.cup_id)]?.name;
		if (name === undefined || name === team.name) continue;
		const last = result.at(-1);
		if (last && last.name === name) last.seasons.push(season.name);
		else result.push({ name, seasons: [season.name] });
	}
	return result;
}

export function teamsIn(cupId: number, group: Group): SeasonTeam[] {
	return teams
		.filter((t) => t.group === group && t.cup_ids.includes(cupId))
		.map((t) => seasonTeam(t, cupId));
}

export function seasonGroupEntries(): { season: string; group: Group }[] {
	return seasons.flatMap((season) => GROUPS.map((group) => ({ season: season.name, group })));
}

export function standingsFor(cupId: number, group: Group): GroupStandings | undefined {
	return (standingsJson as StandingsData).standings.find(
		(s) => s.cup_id === cupId && s.group === group
	);
}

// 對不到球隊代表資料檔壞了：直接丟錯讓 build 失敗，不默默少一列
export function toStandingsRow(entry: StandingsEntry, group: Group, cupId: number): StandingsRow {
	const { team_id, ...rest } = entry;
	const team = teams.find((t) => t.group === group && t.team_id === team_id);
	if (!team) throw new Error(`戰績的 team_id ${team_id}（${group}）不在 teams.json`);
	return { ...rest, team: seasonTeam(team, cupId) };
}

export type MatchView = StyleMatch & {
	game_id: number;
	stage: string | null;
	week: number | null;
};

function teamOf(teamId: number, group: Group, cupId: number, what: string): SeasonTeam {
	const team = teams.find((t) => t.group === group && t.team_id === teamId);
	if (!team) throw new Error(`${what}的 team_id ${teamId}（${group}）不在 teams.json`);
	return seasonTeam(team, cupId);
}

export function toMatchView(entry: MatchEntry, group: Group, cupId: number): MatchView {
	return {
		game_id: entry.game_id,
		date: entry.date,
		weekday: entry.weekday,
		start_time: entry.start_time,
		venue: entry.venue ?? '場館未定',
		stage: entry.stage,
		week: entry.week,
		home: teamOf(entry.home_team_id, group, cupId, '賽事'),
		away: teamOf(entry.away_team_id, group, cupId, '賽事'),
		sets: entry.sets.map(([h, a]) => [h, a]),
		golden_set: entry.golden_set,
		home_win_prob: entry.home_win_prob
	};
}

export function matchesFor(cupId: number, group: Group): MatchView[] {
	const found = (matchesJson as MatchesData).matches.find(
		(m) => m.cup_id === cupId && m.group === group
	);
	return (found?.matches ?? []).map((entry) => toMatchView(entry, group, cupId));
}

export function findMatch(cupId: number, group: Group, gameId: number): MatchView | undefined {
	return matchesFor(cupId, group).find((m) => m.game_id === gameId);
}

export function matchEntries(): { season: string; group: Group; game: string }[] {
	return seasonGroupEntries().flatMap(({ season, group }) => {
		const cupId = findSeason(season)?.cup_id;
		if (cupId === undefined) return [];
		return matchesFor(cupId, group).map((m) => ({ season, group, game: String(m.game_id) }));
	});
}

export function findTeam(cupId: number, group: Group, teamId: number): Team | undefined {
	return teams.find(
		(t) => t.group === group && t.team_id === teamId && t.cup_ids.includes(cupId)
	);
}

export function teamEntries(): { season: string; group: Group; team: string }[] {
	return seasonGroupEntries().flatMap(({ season, group }) => {
		const cupId = findSeason(season)?.cup_id;
		if (cupId === undefined) return [];
		return teamsIn(cupId, group)
			.filter((t) => findTeam(cupId, group, t.team_id))
			.map((t) => ({ season, group, team: String(t.team_id) }));
	});
}

export function teamStanding(cupId: number, group: Group, teamId: number): StandingsEntry | undefined {
	return standingsFor(cupId, group)?.rows.find((r) => r.team_id === teamId);
}

export function teamMatches(cupId: number, group: Group, teamId: number): MatchView[] {
	return matchesFor(cupId, group)
		.filter((m) => m.home.team_id === teamId || m.away.team_id === teamId)
		.sort(
			(a, b) =>
				a.date.localeCompare(b.date) || (a.start_time ?? '').localeCompare(b.start_time ?? '')
		);
}

export type RosterCardPlayer = {
	player_id: number;
	name: string;
	number: number | null;
	position: string | null;
};

export type RosterView = {
	kind: 'week' | 'season';
	week_label: string | null;
	week_start_date: string | null;
	players: RosterCardPlayer[];
};

export function pickLatestWeek(weeks: readonly RosterWeek[]): RosterWeek | undefined {
	return weeks.reduce<RosterWeek | undefined>((best, w) => {
		if (!best) return w;
		return (w.week_start_date ?? '') > (best.week_start_date ?? '') ? w : best;
	}, undefined);
}

export function playerDisplayName(player: { player_id: number; name: string | null }): string {
	return player.name ?? `球員 #${player.player_id}`;
}

const POSITION_NAMES: Record<string, string> = {
	OH: '主攻',
	MB: '攔中',
	OP: '副攻',
	S: '舉球',
	L: '自由'
};

export function positionLabel(code: string | null, withCode = false): string {
	if (code === null) return '—';
	const name = POSITION_NAMES[code];
	if (!name) return code;
	return withCode ? `${name}（${code}）` : name;
}

export function playerName(
	group: Group,
	playerId: number,
	entries: readonly PlayerEntry[]
): string {
	const entry = entries.find((e) => e.group === group && e.player_id === playerId);
	return playerDisplayName(entry ?? { player_id: playerId, name: null });
}

export function toRosterView(
	week: RosterWeek,
	group: Group,
	entries: readonly PlayerEntry[]
): RosterView {
	return {
		kind: 'week',
		week_label: week.week_label,
		week_start_date: week.week_start_date,
		players: toCardPlayers(week.players, group, entries)
	};
}

function toCardPlayers(
	list: readonly { player_id: number; jersey_number: number | null; position: string | null }[],
	group: Group,
	entries: readonly PlayerEntry[]
): RosterCardPlayer[] {
	return list.map((p) => ({
		player_id: p.player_id,
		name: playerName(group, p.player_id, entries),
		number: p.jersey_number,
		position: p.position
	}));
}

export function pickTeamRoster(
	roster: TeamRoster,
	group: Group,
	entries: readonly PlayerEntry[]
): RosterView | undefined {
	const week = pickLatestWeek(roster.weeks);
	if (week) return toRosterView(week, group, entries);
	if (roster.season_roster.length === 0) return undefined;
	return {
		kind: 'season',
		week_label: null,
		week_start_date: null,
		players: toCardPlayers(roster.season_roster, group, entries)
	};
}

export function displayPosition(season: PlayerSeason): string | null {
	return season.position ?? season.season_roster_position;
}

function registeredIn(cupId: number, group: Group): PlayerEntry[] {
	return players.filter((p) => p.group === group && p.seasons.some((s) => s.cup_id === cupId));
}

export function playerEntries(): { season: string; group: Group; player: string }[] {
	return seasonGroupEntries().flatMap(({ season, group }) => {
		const cupId = findSeason(season)?.cup_id;
		if (cupId === undefined) return [];
		return registeredIn(cupId, group).map((p) => ({ season, group, player: String(p.player_id) }));
	});
}

export function findPlayerSeason(
	cupId: number,
	group: Group,
	playerId: number
): { player: PlayerEntry; season: PlayerSeason; team: SeasonTeam } | undefined {
	const player = players.find((p) => p.group === group && p.player_id === playerId);
	return player && withSeason(player, cupId);
}

function withSeason(
	player: PlayerEntry,
	cupId: number
): { player: PlayerEntry; season: PlayerSeason; team: SeasonTeam } | undefined {
	const season = player.seasons.find((s) => s.cup_id === cupId);
	const team = season && teams.find((t) => t.group === player.group && t.team_id === season.team_id);
	if (!season || !team) return undefined;
	return { player, season, team: seasonTeam(team, cupId) };
}

function byNumberThenName(a: RosterCardPlayer, b: RosterCardPlayer): number {
	return (
		(a.number ?? Infinity) - (b.number ?? Infinity) || a.name.localeCompare(b.name, 'zh-Hant')
	);
}

export function playersByTeam(
	cupId: number,
	group: Group
): { team: SeasonTeam; players: RosterCardPlayer[] }[] {
	const rows = registeredIn(cupId, group).flatMap((p) => {
		const found = withSeason(p, cupId);
		if (!found) return [];
		const card: RosterCardPlayer = {
			player_id: p.player_id,
			name: playerDisplayName(p),
			number: found.season.jersey_number,
			position: displayPosition(found.season)
		};
		return [{ team: found.team, card }];
	});
	return teams
		.filter((t) => t.group === group)
		.map((team) => ({
			team: seasonTeam(team, cupId),
			players: rows
				.filter((r) => r.team.team_id === team.team_id)
				.map((r) => r.card)
				.sort(byNumberThenName)
		}))
		.filter((g) => g.players.length > 0);
}

export function latestRoster(cupId: number, group: Group, teamId: number): RosterView | undefined {
	const roster = (rostersJson as RostersData).rosters.find(
		(r) => r.cup_id === cupId && r.group === group && r.team_id === teamId
	);
	return roster && pickTeamRoster(roster, group, players);
}
