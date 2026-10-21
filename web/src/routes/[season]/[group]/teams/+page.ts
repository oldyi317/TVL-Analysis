import { error } from '@sveltejs/kit';
import { findSeason, isGroup, seasonGroupEntries, teamStanding, teamsIn } from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = seasonGroupEntries;

export const load: PageLoad = ({ params }) => {
	const season = findSeason(params.season);
	if (!season || !isGroup(params.group)) error(404, '找不到這個賽季或組別');
	const group = params.group;
	const teams = teamsIn(season.cup_id, group).map((team) => ({
		team,
		standing: teamStanding(season.cup_id, group, team.team_id) ?? null
	}));
	return { season, group, teams };
};
