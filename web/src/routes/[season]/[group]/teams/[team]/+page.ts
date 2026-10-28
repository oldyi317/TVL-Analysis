import { error } from '@sveltejs/kit';
import { parseId, requireSeasonGroup } from '#lib/route.js';
import {
	findTeam,
	formerNames,
	latestRoster,
	seasonTeam,
	teamEntries,
	teamMatches,
	teamStanding
} from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = teamEntries;

export const load: PageLoad = ({ params }) => {
	const { season, group } = requireSeasonGroup(params);
	const teamId = parseId(params.team);
	const team = teamId === undefined ? undefined : findTeam(season.cup_id, group, teamId);
	if (!team) error(404, '這一季沒有這支球隊');
	return {
		season,
		group,
		team,
		seasonTeam: seasonTeam(team, season.cup_id),
		formerNames: formerNames(team),
		standing: teamStanding(season.cup_id, group, team.team_id),
		matches: teamMatches(season.cup_id, group, team.team_id),
		roster: latestRoster(season.cup_id, group, team.team_id)
	};
};
