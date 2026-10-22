import { latestMatchDay, nextMatch } from '#lib/match.js';
import { requireSeasonGroup } from '#lib/route.js';
import {
	matchesFor,
	seasonGroupEntries,
	standingsFor,
	toStandingsRow
} from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = seasonGroupEntries;

export const load: PageLoad = ({ params }) => {
	const { season, group } = requireSeasonGroup(params);
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
