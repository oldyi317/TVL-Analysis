import seasonsJson from '#lib/data/seasons.json';
import standingsJson from '#lib/data/standings.json';
import teamsJson from '#lib/data/teams.json';
import type { StandingsRow } from '#lib/components/StandingsTable.svelte';
import type {
	GroupStandings,
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
export function toStandingsRow(
	entry: StandingsEntry,
	group: Group,
	teamList: readonly Team[] = teams
): StandingsRow {
	const team = teamList.find((t) => t.group === group && t.team_id === entry.team_id);
	if (!team) throw new Error(`戰績的 team_id ${entry.team_id}（${group}）不在 teams.json`);
	return {
		rank: entry.rank,
		team,
		wins: entry.wins,
		losses: entry.losses,
		points: entry.points,
		set_ratio: entry.set_ratio,
		point_ratio: entry.point_ratio,
		set_record: entry.set_record,
		last5: entry.last5
	};
}
