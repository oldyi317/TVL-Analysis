import { findSeason, isGroup, playersByTeam, seasonGroupEntries } from '#lib/site.js';
import { error } from '@sveltejs/kit';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = seasonGroupEntries;

export const load: PageLoad = ({ params }) => {
	const season = findSeason(params.season);
	if (!season || !isGroup(params.group)) error(404, '找不到這個賽季或組別');
	return { season, group: params.group, teams: playersByTeam(season.cup_id, params.group) };
};
