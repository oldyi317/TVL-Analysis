import { error } from '@sveltejs/kit';
import {
	currentSeason,
	findSeason,
	findTeam,
	formerNames,
	isGroup,
	latestRoster,
	seasonTeam,
	teamEntries,
	teamMatches,
	teamStanding
} from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = teamEntries;

export const load: PageLoad = ({ params }) => {
	const season = findSeason(params.season);
	if (!season || !isGroup(params.group)) error(404, '找不到這個賽季或組別');
	const teamId = Number(params.team);
	const team = findTeam(season.cup_id, params.group, teamId);
	if (!team) error(404, '這一季沒有這支球隊');
	return {
		season,
		group: params.group,
		team,
		currentSeasonTeam: seasonTeam(team, currentSeason.cup_id),
		seasonTeam: seasonTeam(team, season.cup_id),
		formerNames: formerNames(team),
		standing: teamStanding(season.cup_id, params.group, teamId),
		matches: teamMatches(season.cup_id, params.group, teamId),
		roster: latestRoster(season.cup_id, params.group, teamId)
	};
};
