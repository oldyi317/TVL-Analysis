import { describe, expect, it } from 'vitest';
import { formatTaipeiMinute } from './datetime.js';

describe('formatTaipeiMinute', () => {
	it('台北偏移的字串照原值顯示到分', () => {
		expect(formatTaipeiMinute('2026-10-28T21:05:33+08:00')).toBe('2026-10-28 21:05');
	});

	it('UTC 字串轉成台北時間', () => {
		expect(formatTaipeiMinute('2026-10-28T13:05:00Z')).toBe('2026-10-28 21:05');
	});

	it('跨日：UTC 深夜對應台北隔日清晨', () => {
		expect(formatTaipeiMinute('2026-12-31T18:30:00-05:00')).toBe('2027-01-01 07:30');
	});

	it('個位數月日時分補零', () => {
		expect(formatTaipeiMinute('2026-03-04T05:06:07+08:00')).toBe('2026-03-04 05:06');
	});
});
