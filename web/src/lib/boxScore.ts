import type { BoxScoresData, GameBoxScore, MatchTopBoard, PlayerEntry, StatRow } from '#lib/data/types.js';
import { playerDisplayName, playerName, teamOf, type Group, type SeasonTeam } from '#lib/site.js';

export type BoxRow = {
	player_id: number;
	name: string;
	number: number | null;
	position: string | null;
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

type DetailStats = Pick<
	BoxRow,
	| 'attack_total'
	| 'serve_total'
	| 'receive_excellent'
	| 'receive_total'
	| 'dig_excellent'
	| 'dig_total'
	| 'set_excellent'
	| 'set_total'
>;

type DetailCol ={ label: string; value: (s: DetailStats) => string };

// zeroTotal：總數為 0 時顯示的字；不給就照樣組成「0／0」
export const goodOverTotal = (good: number, total: number, zeroTotal?: string): string =>
	total === 0 && zeroTotal !== undefined ? zeroTotal : `${good}／${total}`;

export function detailCols({ receiveLabel = '接發球', zeroTotal }: { receiveLabel?: string; zeroTotal?: string } = {}): DetailCol[] {
	return [
		{ label: '攻擊次數', value: (s) => String(s.attack_total) },
		{ label: '發球次數', value: (s) => String(s.serve_total) },
		{ label: receiveLabel, value: (s) => goodOverTotal(s.receive_excellent, s.receive_total, zeroTotal) },
		{ label: '防守', value: (s) => goodOverTotal(s.dig_excellent, s.dig_total, zeroTotal) },
		{ label: '舉球', value: (s) => goodOverTotal(s.set_excellent, s.set_total, zeroTotal) }
	];
}

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

export type MatchTopRowView = {
	rank: number;
	player_id: number;
	name: string;
	linked: boolean;
	team: SeasonTeam;
	value: number;
};

export type MatchTopBoardView = {
	key: MatchTopBoard['key'];
	label: string;
	rows: MatchTopRowView[];
};

export function toMatchTopView(
	boards: readonly MatchTopBoard[],
	group: Group,
	cupId: number,
	players: readonly PlayerEntry[]
): MatchTopBoardView[] {
	return boards.map((board) => ({
		key: board.key,
		label: board.label,
		rows: board.rows.map((row) => {
			const entry = players.find((p) => p.group === group && p.player_id === row.player_id);
			return {
				rank: row.rank,
				player_id: row.player_id,
				name: playerDisplayName(entry ?? { player_id: row.player_id, name: null }),
				linked: Boolean(entry?.name?.trim()),
				team: teamOf(row.team_id, group, cupId, '單場前三名'),
				value: row.value
			};
		})
	}));
}
