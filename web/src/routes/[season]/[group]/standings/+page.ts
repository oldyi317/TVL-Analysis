import { error } from '@sveltejs/kit';
import { findSeason, isGroup, seasonGroupEntries, standingsFor, toStandingsRow } from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = seasonGroupEntries;

export const load: PageLoad = ({ params }) => {
	const season = findSeason(params.season);
	if (!season || !isGroup(params.group)) error(404, '找不到這個賽季或組別');
	const standings = standingsFor(season.cup_id, params.group);
	const group = params.group;
	return {
		season,
		group,
		note: standings?.note ?? null,
		rows: (standings?.rows ?? []).map((entry) => toStandingsRow(entry, group, season.cup_id))
	};
};
