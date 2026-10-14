import type { BoxScoresData, GameBoxScore, PlayerEntry, StatRow } from '#lib/data/types.js';
import { playerName, type Group } from '#lib/site.js';

export type BoxRow = {
	player_id: number;
	name: string;
	number: number | null;
	position: string | null;
	sets_played: number;
	total_points: number;
	attack_points: number;
	attack_total: number;
	block_points: number;
	serve_points: number;
	serve_total: number;
	receive_excellent: number;
	receive_total: number;
	dig_excellent: number;
	dig_total: number;
	set_excellent: number;
	set_total: number;
};

export type BoxScoreView = {
	home: BoxRow[];
	away: BoxRow[];
	golden_set: { home: BoxRow[]; away: BoxRow[] } | null;
};

export function findBoxScore(
	data: BoxScoresData,
	cupId: number,
	group: Group,
	gameId: number
): GameBoxScore | undefined {
	return data.box_scores
		.find((b) => b.cup_id === cupId && b.group === group)
		?.games.find((g) => g.game_id === gameId);
}

function toRow(row: StatRow, group: Group, players: readonly PlayerEntry[]): BoxRow {
	return {
		player_id: row.player_id,
		name: playerName(group, row.player_id, players),
		number: row.jersey_number,
		position: row.position,
		sets_played: row.sets_played,
		total_points: row.total_points,
		attack_points: row.attack_points,
		attack_total: row.attack_total,
		block_points: row.block_points,
		serve_points: row.serve_points,
		serve_total: row.serve_total,
		receive_excellent: row.receive_excellent,
		receive_total: row.receive_total,
		dig_excellent: row.dig_excellent,
		dig_total: row.dig_total,
		set_excellent: row.set_excellent,
		set_total: row.set_total
	};
}

export function toBoxScoreView(
	game: GameBoxScore,
	group: Group,
	players: readonly PlayerEntry[]
): BoxScoreView {
	const rows = (list: StatRow[]) => list.map((r) => toRow(r, group, players));
	return {
		home: rows(game.home),
		away: rows(game.away),
		golden_set: game.golden_set
			? { home: rows(game.golden_set.home), away: rows(game.golden_set.away) }
			: null
	};
}
