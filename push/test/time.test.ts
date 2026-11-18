import { describe, expect, it } from 'vitest';
import { daysBetween, inWindow, parseWindow, taipeiTime } from '../src/time';

describe('taipeiTime', () => {
	it('UTC 日期與台北日期不同時，用台北的日期與分鐘數', () => {
		expect(taipeiTime(new Date('2026-11-07T23:50:00Z'))).toEqual({ date: '2026-11-08', minutes: 470 });
	});

	it('台北午夜整點是新的一天、分鐘數 0', () => {
		expect(taipeiTime(new Date('2026-11-08T16:00:00Z'))).toEqual({ date: '2026-11-09', minutes: 0 });
	});

	it('台北 23:59 還是同一天', () => {
		expect(taipeiTime(new Date('2026-11-08T15:59:00Z'))).toEqual({ date: '2026-11-08', minutes: 1439 });
	});

	it('跨年：UTC 12/31 16:00 是台北隔年 1/1 00:00', () => {
		expect(taipeiTime(new Date('2026-12-31T16:00:00Z'))).toEqual({ date: '2027-01-01', minutes: 0 });
	});

	it('台北 10:00 是 UTC 02:00', () => {
		expect(taipeiTime(new Date('2026-11-08T02:00:00Z'))).toEqual({ date: '2026-11-08', minutes: 600 });
	});
});

describe('daysBetween', () => {
	it('同一天是 0', () => {
		expect(daysBetween('2026-11-08', '2026-11-08')).toBe(0);
	});

	it('2026-11-01 到 2026-11-08 是 7', () => {
		expect(daysBetween('2026-11-01', '2026-11-08')).toBe(7);
	});

	it('2026-10-31 到 2026-11-08 是 8', () => {
		expect(daysBetween('2026-10-31', '2026-11-08')).toBe(8);
	});

	it('跨月：2026-10-30 到 2026-11-02 是 3', () => {
		expect(daysBetween('2026-10-30', '2026-11-02')).toBe(3);
	});

	it('跨年：2026-12-30 到 2027-01-02 是 3', () => {
		expect(daysBetween('2026-12-30', '2027-01-02')).toBe(3);
	});

	it('閏年二月：2028-02-28 到 2028-03-01 是 2', () => {
		expect(daysBetween('2028-02-28', '2028-03-01')).toBe(2);
	});

	it('to 早於 from 時是負數', () => {
		expect(daysBetween('2026-11-08', '2026-11-05')).toBe(-3);
	});
});

describe('parseWindow', () => {
	it('08:00-22:00 轉成分鐘數', () => {
		expect(parseWindow('08:00-22:00')).toEqual({ start: 480, end: 1320 });
	});

	it('分鐘也算進去', () => {
		expect(parseWindow('10:00-10:30')).toEqual({ start: 600, end: 630 });
	});

	it('開始與結束相同可以', () => {
		expect(parseWindow('10:00-10:00')).toEqual({ start: 600, end: 600 });
	});

	it.each(['', '8:00-22:00', '08:00-22:0', '08:00~22:00', '08:00-22:00 ', '08:00', 'abc', '08-22'])('格式不對 %j 回 null', (text) => {
		expect(parseWindow(text)).toBeNull();
	});

	it.each(['24:00-25:00', '08:60-22:00', '08:00-22:60', '08:00-24:00'])('時分超出範圍 %j 回 null', (text) => {
		expect(parseWindow(text)).toBeNull();
	});

	it('開始晚於結束回 null', () => {
		expect(parseWindow('22:00-08:00')).toBeNull();
	});
});

describe('inWindow', () => {
	const window = { start: 480, end: 1320 };

	it('07:59 不在時段內', () => {
		expect(inWindow(window, 479)).toBe(false);
	});

	it('08:00 在時段內', () => {
		expect(inWindow(window, 480)).toBe(true);
	});

	it('22:00 在時段內', () => {
		expect(inWindow(window, 1320)).toBe(true);
	});

	it('22:01 不在時段內', () => {
		expect(inWindow(window, 1321)).toBe(false);
	});

	it('時段中間在內', () => {
		expect(inWindow(window, 720)).toBe(true);
	});
});
