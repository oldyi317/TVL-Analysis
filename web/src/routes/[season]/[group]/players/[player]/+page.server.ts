import { error } from '@sveltejs/kit';
import { findAdvanced, findDistribution, playerGameLog } from '#lib/advanced.js';
import advancedJson from '#lib/data/player_advanced.json';
import boxScoresJson from '#lib/data/box_scores.json';
import type { BoxScoresData, PlayerAdvancedData } from '#lib/data/types.js';
import { parseId, requireSeasonGroup } from '#lib/route.js';
import { findPlayerSeason, matchesFor, playerEntries } from '#lib/site.js';
import type { EntryGenerator, PageServerLoad } from './$types';

export const entries: EntryGenerator = playerEntries;

export const load: PageServerLoad = ({ params }) => {
	const { season, group } = requireSeasonGroup(params);
	const playerId = parseId(params.player);
	const found = playerId === undefined ? undefined : findPlayerSeason(season.cup_id, group, playerId);
	if (!found) error(404, '這一季沒有這位球員的登錄');
	const advanced = findAdvanced(advancedJson as PlayerAdvancedData, season.cup_id, group, found.player.player_id);
	const distribution =
		advanced?.position && !advanced.insufficient_sets
			? findDistribution(advancedJson as PlayerAdvancedData, season.cup_id, group, advanced.position)
			: undefined;
	return {
		season,
		group,
		player: found.player,
		stats: found.season,
		team: found.team,
		advanced: advanced ?? null,
		distribution: distribution ?? null,
		gameLog: playerGameLog(
			boxScoresJson as BoxScoresData,
			matchesFor(season.cup_id, group),
			season.cup_id,
			group,
			found.player.player_id
		)
	};
};
