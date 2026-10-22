import { groupByMonth } from '#lib/match.js';
import { requireSeasonGroup } from '#lib/route.js';
import { matchesFor, seasonGroupEntries } from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = seasonGroupEntries;

export const load: PageLoad = ({ params }) => {
	const { season, group } = requireSeasonGroup(params);
	return { season, group, months: groupByMonth(matchesFor(season.cup_id, group)) };
};
