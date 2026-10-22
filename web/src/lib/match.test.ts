import { describe, expect, it } from 'vitest';
import { groupByMonth, isPlayed, latestMatchDay, nextMatch, setsWon, winnerSide, winProbPercent } from './match.js';

const sets: [number, number][] = [
	[25, 20],
	[23, 25],
	[25, 18],
	[25, 27],
	[15, 11],
];

describe('isPlayed', () => {
	it('沒有任何局比分算未開打', () => {
		expect(isPlayed({ sets: [] })).toBe(false);
	});
	it('有局比分算已開打', () => {
		expect(isPlayed({ sets: [[25, 20]] })).toBe(true);
	});
});

describe('setsWon', () => {
	it('數每一局贏家', () => {
		expect(setsWon(sets)).toEqual([3, 2]);
	});
	it('沒有局數回 0:0', () => {
		expect(setsWon([])).toEqual([0, 0]);
	});
});

describe('winnerSide', () => {
	it('贏三局的一方是勝方', () => {
		expect(winnerSide(sets)).toBe('home');
		expect(winnerSide([[20, 25], [18, 25], [25, 23], [22, 25]])).toBe('away');
	});
	it('未開打沒有勝方', () => {
		expect(winnerSide([])).toBeNull();
	});
});

describe('groupByMonth', () => {
	const at = (date: string) => ({ date });
	it('依月份分段並保留順序，標籤含年月', () => {
		const months = groupByMonth([at('2025-11-01'), at('2025-11-08'), at('2026-01-10')]);
		expect(months.map((m) => [m.key, m.label, m.short, m.matches.length])).toEqual([
			['2025-11', '2025 年 11 月', '11月', 2],
			['2026-01', '2026 年 1 月', '1月', 1]
		]);
	});
	it('沒有賽事回空陣列', () => {
		expect(groupByMonth([])).toEqual([]);
	});
});

describe('latestMatchDay', () => {
	const played = (date: string, id: number) => ({ date, id, sets: [[25, 20]] });
	const pending = (date: string, id: number) => ({ date, id, sets: [] });
	it('取最晚比賽日並含該日全部場次', () => {
		const result = latestMatchDay([played('2026-03-01', 1), played('2026-03-22', 2), played('2026-03-22', 3)]);
		expect(result?.date).toBe('2026-03-22');
		expect(result?.matches.map((m) => m.id)).toEqual([2, 3]);
	});
	it('忽略未開打場次', () => {
		const result = latestMatchDay([played('2026-03-01', 1), pending('2026-04-01', 2)]);
		expect(result?.date).toBe('2026-03-01');
		expect(result?.matches.map((m) => m.id)).toEqual([1]);
	});
	it('沒有已完賽場次回 null', () => {
		expect(latestMatchDay([])).toBeNull();
		expect(latestMatchDay([pending('2026-04-01', 1)])).toBeNull();
	});
	it('輸入未排序也正確並保持原順序', () => {
		const result = latestMatchDay([played('2026-03-22', 1), played('2026-03-01', 2), played('2026-03-22', 3)]);
		expect(result?.matches.map((m) => m.id)).toEqual([1, 3]);
	});
});

describe('nextMatch', () => {
	it('回日期最早的未開打，不依賴輸入順序', () => {
		const list = [
			{ date: '2026-03-01', id: 1, sets: [[25, 20]] },
			{ date: '2026-04-08', id: 3, sets: [] },
			{ date: '2026-04-01', id: 2, sets: [] }
		];
		expect(nextMatch(list)?.id).toBe(2);
	});
	it('同日依開賽時間，已知時間優先於 null，且只回一場', () => {
		const list = [
			{ date: '2026-04-01', id: 1, start_time: null, sets: [] },
			{ date: '2026-04-01', id: 2, start_time: '19:00', sets: [] },
			{ date: '2026-04-01', id: 3, start_time: '14:00', sets: [] }
		];
		expect(nextMatch(list)?.id).toBe(3);
		expect(nextMatch(list.filter((m) => m.id !== 3))?.id).toBe(2);
		expect(nextMatch(list.filter((m) => m.id === 1))?.id).toBe(1);
	});
	it('沒有未開打回 null', () => {
		expect(nextMatch([{ date: '2026-03-01', sets: [[25, 20]] }])).toBeNull();
		expect(nextMatch([])).toBeNull();
	});
});

describe('winProbPercent', () => {
	it('null 與 undefined 回 null', () => {
		expect(winProbPercent(null)).toBeNull();
		expect(winProbPercent(undefined)).toBeNull();
	});
	it('整數百分比且相加為 100', () => {
		expect(winProbPercent(0.62)).toEqual([62, 38]);
		expect(winProbPercent(0.5)).toEqual([50, 50]);
		expect(winProbPercent(0)).toEqual([0, 100]);
		expect(winProbPercent(1)).toEqual([100, 0]);
	});
	it('四捨五入邊界', () => {
		expect(winProbPercent(0.615)).toEqual([62, 38]);
		expect(winProbPercent(0.614)).toEqual([61, 39]);
	});
});
