import { error } from '@sveltejs/kit';
import { findBoxScore, toBoxScoreView, toMatchTopView } from '#lib/boxScore.js';
import boxScoresJson from '#lib/data/box_scores.json';
import teamAdvancedJson from '#lib/data/team_advanced.json';
import type { BoxScoresData, TeamAdvancedData } from '#lib/data/types.js';
import { headToHead, isPlayed } from '#lib/match.js';
import { parseId, requireSeasonGroup } from '#lib/route.js';
import { findMatch, matchEntries, matchesFor, players } from '#lib/site.js';
import { postgameComparison, pregameComparison } from '#lib/teamAdvanced.js';
import type { EntryGenerator, PageServerLoad } from './$types';

export const entries: EntryGenerator = matchEntries;

export const load: PageServerLoad = ({ params }) => {
	const { season, group } = requireSeasonGroup(params);
	const gameId = parseId(params.game);
	const match = gameId === undefined ? undefined : findMatch(season.cup_id, group, gameId);
	if (!match) error(404, '找不到這場賽事');
	const played = isPlayed(match);
	const teamAdvanced = teamAdvancedJson as TeamAdvancedData;
	const pregame = played
		? null
		: {
				comparison: pregameComparison(teamAdvanced, season.cup_id, group, match.home.team_id, match.away.team_id) ?? null,
				record: headToHead(matchesFor(season.cup_id, group), match.home.team_id, match.away.team_id)
			};
	const postgame = played
		? (postgameComparison(teamAdvanced, season.cup_id, group, match.game_id, match.home.team_id, match.away.team_id) ?? null)
		: null;
	const game = findBoxScore(boxScoresJson as BoxScoresData, season.cup_id, group, match.game_id);
	return {
		season,
		group,
		match,
		pregame,
		postgame,
		topPerformers: played && game ? toMatchTopView(game.top_performers, group, season.cup_id, players) : [],
		boxScore: game ? toBoxScoreView(game, group, players) : null
	};
};
