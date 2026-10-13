import { describe, expect, it } from 'vitest';
import { setsWon, winnerSide } from './match.js';

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
