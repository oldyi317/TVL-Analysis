import { error } from '@sveltejs/kit';
import { groupByMonth } from '#lib/match.js';
import { findSeason, isGroup, matchesFor, seasonGroupEntries } from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = seasonGroupEntries;

export const load: PageLoad = ({ params }) => {
	const season = findSeason(params.season);
	if (!season || !isGroup(params.group)) error(404, '找不到這個賽季或組別');
	const group = params.group;
	return { season, group, months: groupByMonth(matchesFor(season.cup_id, group)) };
};
