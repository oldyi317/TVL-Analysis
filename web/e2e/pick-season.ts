import { isPlayed } from '../src/lib/match.js';

export type Kind = 'played' | 'players' | 'teams' | 'rosters' | 'boxscores';
export type Source = {
	rows: any[];
	seasons: { cup_id: number; name: string }[];
	currentCupId: number;
};

// 各資料種類對應的 JSON 檔，以及「這一列算有資料」的 cup_id 取法
export const KINDS: Record<Kind, { file: string; cups: (r: any) => number[] }> = {
	played: {
		file: 'matches',
		cups: (r) => (r.matches.some((m: { sets: unknown[] }) => isPlayed(m)) ? [r.cup_id] : [])
	},
	players: { file: 'players', cups: (r) => r.seasons.map((s: { cup_id: number }) => s.cup_id) },
	teams: { file: 'teams', cups: (r) => r.cup_ids },
	rosters: { file: 'rosters', cups: (r) => (r.weeks.length > 0 ? [r.cup_id] : []) },
	boxscores: { file: 'box_scores', cups: (r) => (r.games.length > 0 ? [r.cup_id] : []) }
};

// 回傳最近一個有該資料的賽季：當季有就當季，否則取有資料的最大 cup_id，都沒有回 undefined
export function pickSeason(kind: Kind, group: string, src: Source) {
	const cups = new Set(src.rows.filter((r) => r.group === group).flatMap(KINDS[kind].cups));
	if (!cups.size) return undefined;
	const cupId = cups.has(src.currentCupId) ? src.currentCupId : Math.max(...cups);
	const season = src.seasons.find((s) => s.cup_id === cupId)?.name;
	return season === undefined ? undefined : { cupId, season };
}
