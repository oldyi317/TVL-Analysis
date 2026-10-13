import { error } from '@sveltejs/kit';
import { GROUPS, findSeason, isGroup, seasons, teamsIn } from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = () =>
	seasons.flatMap((season) => GROUPS.map((group) => ({ season: season.name, group })));

export const load: PageLoad = ({ params }) => {
	const season = findSeason(params.season);
	if (!season || !isGroup(params.group)) error(404, '找不到這個賽季或組別');
	return { season, group: params.group, teams: teamsIn(season.cup_id, params.group) };
};
