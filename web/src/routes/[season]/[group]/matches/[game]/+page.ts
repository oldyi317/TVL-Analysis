import { error } from '@sveltejs/kit';
import { findMatch, findSeason, isGroup, matchEntries } from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = matchEntries;

export const load: PageLoad = ({ params }) => {
	const season = findSeason(params.season);
	if (!season || !isGroup(params.group)) error(404, '找不到這個賽季或組別');
	const match = findMatch(season.cup_id, params.group, Number(params.game));
	if (!match) error(404, '找不到這場賽事');
	return { season, group: params.group, match };
};
