import { error } from '@sveltejs/kit';
import { findBoxScore, toBoxScoreView } from '#lib/boxScore.js';
import boxScoresJson from '#lib/data/box_scores.json';
import type { BoxScoresData } from '#lib/data/types.js';
import { findMatch, findSeason, isGroup, matchEntries, players } from '#lib/site.js';
import type { EntryGenerator, PageServerLoad } from './$types';

export const entries: EntryGenerator = matchEntries;

export const load: PageServerLoad = ({ params }) => {
	const season = findSeason(params.season);
	if (!season || !isGroup(params.group)) error(404, '找不到這個賽季或組別');
	const match = findMatch(season.cup_id, params.group, Number(params.game));
	if (!match) error(404, '找不到這場賽事');
	const game = findBoxScore(boxScoresJson as BoxScoresData, season.cup_id, params.group, match.game_id);
	return {
		season,
		group: params.group,
		match,
		boxScore: game ? toBoxScoreView(game, params.group, players) : null
	};
};
