import type { Team } from '#lib/data/types.js';

export type SetScore = [number, number];
export type Side = 'home' | 'away';

// 各局比分 → 雙方贏得的局數
export function setsWon(sets: readonly SetScore[]): [number, number] {
	let home = 0;
	let away = 0;
	for (const [h, a] of sets) {
		if (h > a) home++;
		else if (a > h) away++;
	}
	return [home, away];
}

// 先拿三局的是勝方；未開打或資料不全回 null
export function winnerSide(sets: readonly SetScore[]): Side | null {
	const [home, away] = setsWon(sets);
	if (home >= 3 && home > away) return 'home';
	if (away >= 3 && away > home) return 'away';
	return null;
}

export type StyleMatch = {
	date: string;
	weekday: string;
	venue: string;
	home: Team;
	away: Team;
	sets: SetScore[];
	golden_set?: SetScore | null;
	start_time?: string | null;
};
