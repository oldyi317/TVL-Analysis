import { requireSeasonGroup } from '#lib/route.js';
import { seasonGroupEntries, teamStanding, teamsIn } from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = seasonGroupEntries;

export const load: PageLoad = ({ params }) => {
	const { season, group } = requireSeasonGroup(params);
	const teams = teamsIn(season.cup_id, group).map((team) => ({
		team,
		standing: teamStanding(season.cup_id, group, team.team_id) ?? null
	}));
	return { season, group, teams };
};
