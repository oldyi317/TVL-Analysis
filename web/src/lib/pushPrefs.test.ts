import { describe, expect, it } from 'vitest';
import {
	DEFAULT_PREFS,
	PREFS_KEY,
	REGISTERED_KEY,
	checksOf,
	groupOf,
	isDue,
	readPrefs,
	readRegisteredAt,
	registerBody,
	urlBase64ToBytes,
	writePrefs,
	writeRegisteredAt
} from './pushPrefs.js';

function fakeStorage(initial: Record<string, string> = {}) {
	const data = { ...initial };
	return {
		data,
		getItem: (key: string) => data[key] ?? null,
		setItem: (key: string, value: string) => {
			data[key] = value;
		}
	};
}

const throwing = () => {
	throw new Error('denied');
};

describe('readPrefs', () => {
	it('讀到合法偏好', () => {
		const stored = { results: true, today: false, group: 'women' };
		expect(readPrefs(fakeStorage({ [PREFS_KEY]: JSON.stringify(stored) }))).toEqual(stored);
	});
	it.each([
		['沒有紀錄', undefined],
		['不是 JSON', 'oops'],
		['欄位型別不對', JSON.stringify({ results: 1, today: false, group: 'both' })],
		['組別不認得', JSON.stringify({ results: true, today: false, group: 'mixed' })],
		['是 null', 'null']
	])('%s 回預設值（兩種都關、兩組）', (_name, value) => {
		const storage = fakeStorage(value === undefined ? {} : { [PREFS_KEY]: value });
		expect(readPrefs(storage)).toEqual({ results: false, today: false, group: 'both' });
	});
	it('storage 為 null 或丟例外回預設值', () => {
		expect(readPrefs(null)).toEqual(DEFAULT_PREFS);
		expect(readPrefs({ getItem: throwing })).toEqual(DEFAULT_PREFS);
	});
});

describe('writePrefs', () => {
	it('寫成 JSON，讀得回來', () => {
		const storage = fakeStorage();
		const prefs = { results: false, today: true, group: 'men' } as const;
		writePrefs(storage, prefs);
		expect(readPrefs(storage)).toEqual(prefs);
	});
	it('storage 為 null 或丟例外不中斷', () => {
		expect(() => writePrefs(null, DEFAULT_PREFS)).not.toThrow();
		expect(() => writePrefs({ setItem: throwing }, DEFAULT_PREFS)).not.toThrow();
	});
});

describe('最後登記時間', () => {
	it('讀寫', () => {
		const storage = fakeStorage();
		writeRegisteredAt(storage, 1234);
		expect(storage.data[REGISTERED_KEY]).toBe('1234');
		expect(readRegisteredAt(storage)).toBe(1234);
	});
	it('沒有紀錄、壞掉、storage 壞了都是 null', () => {
		expect(readRegisteredAt(fakeStorage())).toBeNull();
		expect(readRegisteredAt(fakeStorage({ [REGISTERED_KEY]: 'abc' }))).toBeNull();
		expect(readRegisteredAt(null)).toBeNull();
		expect(readRegisteredAt({ getItem: throwing })).toBeNull();
	});
	it('寫入失敗不中斷', () => {
		expect(() => writeRegisteredAt({ setItem: throwing }, 1)).not.toThrow();
	});
});

describe('isDue（距上次登記滿 24 小時才重新登記）', () => {
	const now = 10 * 24 * 3600_000;
	const hour = 3600_000;
	it.each([
		['沒有紀錄', null, true],
		['23 小時前', now - 23 * hour, false],
		['剛好 24 小時前', now - 24 * hour, true],
		['25 小時前', now - 25 * hour, true],
		['在未來', now + hour, true]
	])('%s', (_name, at, expected) => {
		expect(isDue(at, now)).toBe(expected);
	});
});

describe('組別與兩個勾選互轉', () => {
	it.each([
		[true, true, 'both'],
		[true, false, 'men'],
		[false, true, 'women']
	] as const)('男 %s 女 %s → %s', (men, women, group) => {
		expect(groupOf(men, women)).toBe(group);
		expect(checksOf(group)).toEqual({ men, women });
	});
	it('兩個都沒勾退回兩組', () => {
		expect(groupOf(false, false)).toBe('both');
	});
});

describe('registerBody', () => {
	it('展開訂閱內容，再用偏好覆寫三個欄位', () => {
		const sub = { endpoint: 'https://x', keys: { p256dh: 'a', auth: 'b' }, results: false, group: 'men' };
		expect(registerBody(sub, { results: true, today: false, group: 'both' })).toEqual({
			endpoint: 'https://x',
			keys: { p256dh: 'a', auth: 'b' },
			results: true,
			today: false,
			group: 'both'
		});
	});
});

describe('urlBase64ToBytes', () => {
	it('base64url 轉位元組，不需填充', () => {
		expect([...urlBase64ToBytes('AQID')]).toEqual([1, 2, 3]);
		expect([...urlBase64ToBytes('-_8')]).toEqual([251, 255]);
	});
	it('正式公鑰解開是 65 bytes、第一個位元組 4', () => {
		const key = 'BGDlCqN1KpK7X7zAF4M5gsbj4zJ8lzHYDut1DEqohJOpaAUqlDZJyshcc5x2-p_CEG03u5EjoBDZfUeaqI0kIR0';
		const bytes = urlBase64ToBytes(key);
		expect(bytes.length).toBe(65);
		expect(bytes[0]).toBe(4);
	});
});
