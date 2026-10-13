import { describe, expect, it } from 'vitest';
import { groupByMonth, setsWon, winnerSide } from './match.js';

const sets: [number, number][] = [
	[25, 20],
	[23, 25],
	[25, 18],
	[25, 27],
	[15, 11],
];

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
