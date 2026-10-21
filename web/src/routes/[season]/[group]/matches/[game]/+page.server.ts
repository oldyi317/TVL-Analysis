import { error } from '@sveltejs/kit';
import { findBoxScore, toBoxScoreView } from '#lib/boxScore.js';
import boxScoresJson from '#lib/data/box_scores.json';
import type { BoxScoresData } from '#lib/data/types.js';
import { parseId, requireSeasonGroup } from '#lib/route.js';
import { findMatch, matchEntries, players } from '#lib/site.js';
import type { EntryGenerator, PageServerLoad } from './$types';

export const entries: EntryGenerator = matchEntries;

export const load: PageServerLoad = ({ params }) => {
	const { season, group } = requireSeasonGroup(params);
	const gameId = parseId(params.game);
	const match = gameId === undefined ? undefined : findMatch(season.cup_id, group, gameId);
	if (!match) error(404, '找不到這場賽事');
	const game = findBoxScore(boxScoresJson as BoxScoresData, season.cup_id, group, match.game_id);
	return {
		season,
		group,
		match,
		boxScore: game ? toBoxScoreView(game, group, players) : null
	};
};
