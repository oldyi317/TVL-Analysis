import { error } from '@sveltejs/kit';
import { latestMatchDay, nextMatch } from '#lib/match.js';
import {
	findSeason,
	isGroup,
	matchesFor,
	seasonGroupEntries,
	standingsFor,
	toStandingsRow
} from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = seasonGroupEntries;

export const load: PageLoad = ({ params }) => {
	const season = findSeason(params.season);
	if (!season || !isGroup(params.group)) error(404, '找不到這個賽季或組別');
	const group = params.group;
	const matches = matchesFor(season.cup_id, group);
	const standings = standingsFor(season.cup_id, group);
	return {
		season,
		group,
		latest: latestMatchDay(matches),
		next: nextMatch(matches),
		note: standings?.note ?? null,
		rows: (standings?.rows ?? []).map((entry) => toStandingsRow(entry, group, season.cup_id))
	};
};
