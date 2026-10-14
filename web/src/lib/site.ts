import matchesJson from '#lib/data/matches.json';
import playersJson from '#lib/data/players.json';
import rostersJson from '#lib/data/rosters.json';
import seasonsJson from '#lib/data/seasons.json';
import standingsJson from '#lib/data/standings.json';
import teamsJson from '#lib/data/teams.json';
import type { StandingsRow } from '#lib/components/StandingsTable.svelte';
import type { StyleMatch } from '#lib/match.js';
import type {
	GroupStandings,
	MatchEntry,
	MatchesData,
	PlayerEntry,
	PlayersData,
	RosterWeek,
	RostersData,
	Season,
	SeasonsData,
	StandingsData,
	StandingsEntry,
	Team,
	TeamsData
} from '#lib/data/types.js';

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

export function teamsIn(cupId: number, group: Group): Team[] {
	return teams.filter((t) => t.group === group && t.cup_ids.includes(cupId));
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
export function toStandingsRow(entry: StandingsEntry, group: Group): StandingsRow {
	const { team_id, ...rest } = entry;
	const team = teams.find((t) => t.group === group && t.team_id === team_id);
	if (!team) throw new Error(`戰績的 team_id ${team_id}（${group}）不在 teams.json`);
	return { ...rest, team };
}

export type MatchView = StyleMatch & {
	game_id: number;
	stage: string | null;
	week: number | null;
};

function teamOf(teamId: number, group: Group, what: string): Team {
	const team = teams.find((t) => t.group === group && t.team_id === teamId);
	if (!team) throw new Error(`${what}的 team_id ${teamId}（${group}）不在 teams.json`);
	return team;
}

export function toMatchView(entry: MatchEntry, group: Group): MatchView {
	return {
		game_id: entry.game_id,
		date: entry.date,
		weekday: entry.weekday,
		start_time: entry.start_time,
		venue: entry.venue ?? '場館未定',
		stage: entry.stage,
		week: entry.week,
		home: teamOf(entry.home_team_id, group, '賽事'),
		away: teamOf(entry.away_team_id, group, '賽事'),
		sets: entry.sets.map(([h, a]) => [h, a]),
		golden_set: entry.golden_set
	};
}

export function matchesFor(cupId: number, group: Group): MatchView[] {
	const found = (matchesJson as MatchesData).matches.find(
		(m) => m.cup_id === cupId && m.group === group
	);
	return (found?.matches ?? []).map((entry) => toMatchView(entry, group));
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
		(t) => t.group === group && t.team_id === teamId && t.is_active && t.cup_ids.includes(cupId)
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
	week_label: string;
	week_start_date: string | null;
	players: RosterCardPlayer[];
};

export function pickLatestWeek(weeks: readonly RosterWeek[]): RosterWeek | undefined {
	return weeks.reduce<RosterWeek | undefined>((best, w) => {
		if (!best) return w;
		return (w.week_start_date ?? '') > (best.week_start_date ?? '') ? w : best;
	}, undefined);
}

export function playerName(
	group: Group,
	playerId: number,
	playerEntries: readonly PlayerEntry[]
): string {
	return (
		playerEntries.find((e) => e.group === group && e.player_id === playerId)?.name ??
		`球員 #${playerId}`
	);
}

export function toRosterView(
	week: RosterWeek,
	group: Group,
	playerEntries: readonly PlayerEntry[]
): RosterView {
	return {
		week_label: week.week_label,
		week_start_date: week.week_start_date,
		players: week.players.map((p) => ({
			player_id: p.player_id,
			name: playerName(group, p.player_id, playerEntries),
			number: p.jersey_number,
			position: p.position
		}))
	};
}

export function latestRoster(cupId: number, group: Group, teamId: number): RosterView | undefined {
	const roster = (rostersJson as RostersData).rosters.find(
		(r) => r.cup_id === cupId && r.group === group && r.team_id === teamId
	);
	const week = roster && pickLatestWeek(roster.weeks);
	if (!week) return undefined;
	return toRosterView(week, group, players);
}
