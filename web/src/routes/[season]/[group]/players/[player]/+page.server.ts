import { error } from '@sveltejs/kit';
import { findAdvanced, findDistribution, playerGameLog } from '#lib/advanced.js';
import advancedJson from '#lib/data/player_advanced.json';
import boxScoresJson from '#lib/data/box_scores.json';
import type { BoxScoresData, PlayerAdvancedData } from '#lib/data/types.js';
import { findPlayerSeason, findSeason, isGroup, matchesFor, playerEntries } from '#lib/site.js';
import type { EntryGenerator, PageServerLoad } from './$types';

export const entries: EntryGenerator = playerEntries;

export const load: PageServerLoad = ({ params }) => {
	const season = findSeason(params.season);
	if (!season || !isGroup(params.group)) error(404, '找不到這個賽季或組別');
	const playerId = Number(params.player);
	const found = findPlayerSeason(season.cup_id, params.group, playerId);
	if (!found) error(404, '這一季沒有這位球員的登錄');
	const advanced = findAdvanced(advancedJson as PlayerAdvancedData, season.cup_id, params.group, playerId);
	const distribution =
		advanced?.position && !advanced.insufficient_sets
			? findDistribution(advancedJson as PlayerAdvancedData, season.cup_id, params.group, advanced.position)
			: undefined;
	return {
		season,
		group: params.group,
		player: found.player,
		stats: found.season,
		team: found.team,
		advanced: advanced ?? null,
		distribution: distribution ?? null,
		gameLog: playerGameLog(
			boxScoresJson as BoxScoresData,
			matchesFor(season.cup_id, params.group),
			season.cup_id,
			params.group,
			playerId
		)
	};
};
