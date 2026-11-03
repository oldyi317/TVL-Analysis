import { describe, expect, it } from 'vitest';
import { formatThroughDate, visibleRows } from './leaders.js';

const rows = (ranks: number[]) => ranks.map((rank, i) => ({ rank, player_id: i + 1 }));

describe('排行榜前 10 名截斷', () => {
	it('超過 10 人只留前 10 名', () => {
		const shown = visibleRows(rows([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
		expect(shown.map((r) => r.rank)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
	});

	it('第 10 名並列時全部顯示', () => {
		const shown = visibleRows(rows([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 13]));
		expect(shown.map((r) => r.rank)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10]);
	});

	it('不足 10 人全數顯示', () => {
		expect(visibleRows(rows([1, 1, 3]))).toHaveLength(3);
	});
});

describe('統計至日期', () => {
	it('ISO 日期轉成「X 月 X 日」', () => {
		expect(formatThroughDate('2026-11-02')).toBe('11 月 2 日');
	});
});
