import { describe, expect, it } from 'vitest';
import { GROUP_KEY, readGroup, writeGroup } from './groupMemory.js';

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

describe('readGroup', () => {
	it('讀到合法組別', () => {
		expect(readGroup(fakeStorage({ [GROUP_KEY]: 'women' }))).toBe('women');
	});
	it('讀不到回 men', () => {
		expect(readGroup(fakeStorage())).toBe('men');
	});
	it('非法值回 men', () => {
		expect(readGroup(fakeStorage({ [GROUP_KEY]: 'mixed' }))).toBe('men');
	});
	it('storage 為 null 或 undefined 回 men', () => {
		expect(readGroup(null)).toBe('men');
		expect(readGroup(undefined)).toBe('men');
	});
	it('getItem 拋錯回 men', () => {
		expect(readGroup({ getItem: throwing })).toBe('men');
	});
});

describe('writeGroup', () => {
	it('寫入後可讀回', () => {
		const storage = fakeStorage();
		writeGroup(storage, 'women');
		expect(storage.data[GROUP_KEY]).toBe('women');
		expect(readGroup(storage)).toBe('women');
	});
	it('setItem 拋錯不外洩', () => {
		expect(() => writeGroup({ setItem: throwing }, 'women')).not.toThrow();
	});
	it('storage 為 null 不拋錯', () => {
		expect(() => writeGroup(null, 'women')).not.toThrow();
		expect(() => writeGroup(undefined, 'women')).not.toThrow();
	});
});
