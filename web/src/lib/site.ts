import seasonsJson from '#lib/data/seasons.json';
import teamsJson from '#lib/data/teams.json';
import type { Season, SeasonsData, Team, TeamsData } from '#lib/data/types.js';

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
