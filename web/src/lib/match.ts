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

// 「總決賽 第 19 週」；沒有階段就只有週次，兩者都沒有回 null
export function stageLabel(stage: string | null, week: number | null): string | null {
	const parts = [stage, week === null ? null : `第 ${week} 週`].filter((p) => p !== null);
	return parts.length ? parts.join(' ') : null;
}

export type MonthGroup<T extends { date: string }> = {
	key: string;
	label: string;
	short: string;
	matches: T[];
};

// 依「YYYY-MM」分段，輸入已依日期排序時輸出順序不變
export function groupByMonth<T extends { date: string }>(matches: readonly T[]): MonthGroup<T>[] {
	const groups: MonthGroup<T>[] = [];
	for (const match of matches) {
		const key = match.date.slice(0, 7);
		const last = groups.at(-1);
		if (last && last.key === key) {
			last.matches.push(match);
			continue;
		}
		const [year, month] = key.split('-');
		const m = Number(month);
		groups.push({ key, label: `${year} 年 ${m} 月`, short: `${m}月`, matches: [match] });
	}
	return groups;
}

type Dated = { date: string; sets: readonly unknown[] };

// 最近一個有已完賽場次的比賽日，回該日全部已完賽場次（保持原順序）
export function latestMatchDay<T extends Dated>(
	matches: readonly T[]
): { date: string; matches: T[] } | null {
	const played = matches.filter((m) => m.sets.length > 0);
	if (played.length === 0) return null;
	const date = played.reduce((max, m) => (m.date > max ? m.date : max), played[0].date);
	return { date, matches: played.filter((m) => m.date === date) };
}

export function nextMatch<T extends Dated>(matches: readonly T[]): T | null {
	const unplayed = matches.filter((m) => m.sets.length === 0);
	if (unplayed.length === 0) return null;
	return unplayed.reduce((min, m) => (m.date < min.date ? m : min));
}
