import { error } from '@sveltejs/kit';
import type { Season } from '#lib/data/types.js';
import { findSeason, isGroup, type Group } from '#lib/site.js';

export type SeasonGroupParams = { season: string; group: string | undefined };
export type SeasonGroup = { season: Season; group: Group };

export function resolveSeasonGroup(params: SeasonGroupParams): SeasonGroup | undefined {
	const season = findSeason(params.season);
	if (!season || !params.group || !isGroup(params.group)) return undefined;
	return { season, group: params.group };
}

export function parseId(raw: string): number | undefined {
	return /^\d+$/.test(raw) ? Number(raw) : undefined;
}

export function requireSeasonGroup(params: SeasonGroupParams): SeasonGroup {
	const resolved = resolveSeasonGroup(params);
	if (!resolved) error(404, '找不到這個賽季或組別');
	return resolved;
}
