import { error } from '@sveltejs/kit';
import { findPlayerSeason, findSeason, isGroup, playerEntries } from '#lib/site.js';
import type { EntryGenerator, PageLoad } from './$types';

export const entries: EntryGenerator = playerEntries;

export const load: PageLoad = ({ params }) => {
	const season = findSeason(params.season);
	if (!season || !isGroup(params.group)) error(404, '找不到這個賽季或組別');
	const playerId = Number(params.player);
	const found = findPlayerSeason(season.cup_id, params.group, playerId);
	if (!found) error(404, '這一季沒有這位球員的登錄');
	return {
		season,
		group: params.group,
		player: found.player,
		stats: found.season,
		team: found.team
	};
};
