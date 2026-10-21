import { requireSeasonGroup } from '#lib/route.js';
import { seasonGroupEntries, standingsFor, toStandingsRow } from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = seasonGroupEntries;

export const load: PageLoad = ({ params }) => {
	const { season, group } = requireSeasonGroup(params);
	const standings = standingsFor(season.cup_id, group);
	return {
		season,
		group,
		note: standings?.note ?? null,
		rows: (standings?.rows ?? []).map((entry) => toStandingsRow(entry, group, season.cup_id))
	};
};
