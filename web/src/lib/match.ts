import type { SeasonTeam } from '#lib/site.js';

export type SetScore = [number, number];
export type Side = 'home' | 'away';

export function isPlayed(match: { sets: readonly unknown[] }): boolean {
	return match.sets.length > 0;
}

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
	home: SeasonTeam;
	away: SeasonTeam;
	sets: SetScore[];
	golden_set?: SetScore | null;
	start_time?: string | null;
	home_win_prob?: number | null;
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

type Dated = { date: string; start_time?: string | null; sets: readonly unknown[] };

// 最近一個有已完賽場次的比賽日，回該日全部已完賽場次（保持原順序）
export function latestMatchDay<T extends Dated>(
	matches: readonly T[]
): { date: string; matches: T[] } | null {
	const played = matches.filter(isPlayed);
	if (played.length === 0) return null;
	const date = played.reduce((max, m) => (m.date > max ? m.date : max), played[0].date);
	return { date, matches: played.filter((m) => m.date === date) };
}

function startsBefore(a: Dated, b: Dated): boolean {
	if (a.date !== b.date) return a.date < b.date;
	if (!a.start_time) return false;
	return !b.start_time || a.start_time < b.start_time;
}

export function nextMatch<T extends Dated>(matches: readonly T[]): T | null {
	const unplayed = matches.filter((m) => !isPlayed(m));
	if (unplayed.length === 0) return null;
	return unplayed.reduce((min, m) => (startsBefore(m, min) ? m : min));
}

// 主隊賽前勝率 (0–1) → 兩隊整數百分比，相加必為 100；算不出回 null
export function winProbPercent(p: number | null | undefined): [number, number] | null {
	if (p === null || p === undefined) return null;
	const home = Math.round(p * 100);
	return [home, 100 - home];
}

export const WIN_PROB_NOTE =
	'依兩隊近 3 至 5 場技術統計推估，未考慮對手強度與主客場；開季初以上季末近況為準，僅供參考。';
