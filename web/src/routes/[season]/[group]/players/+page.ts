import { requireSeasonGroup } from '#lib/route.js';
import { playersByTeam, seasonGroupEntries } from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = seasonGroupEntries;

export const load: PageLoad = ({ params }) => {
	const { season, group } = requireSeasonGroup(params);
	return { season, group, teams: playersByTeam(season.cup_id, group) };
};
