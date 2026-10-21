import { describe, expect, it } from 'vitest';
import { parseId, requireSeasonGroup, resolveSeasonGroup } from '#lib/route.js';
import { seasons } from '#lib/site.js';

const known = seasons[0];

describe('resolveSeasonGroup', () => {
	it('合法賽季與組別回傳 season 與 group', () => {
		const result = resolveSeasonGroup({ season: known.name, group: 'men' });
		expect(result?.season).toBe(known);
		expect(result?.group).toBe('men');
	});

	it('不存在的賽季回傳 undefined', () => {
		expect(resolveSeasonGroup({ season: 'no-such-season', group: 'men' })).toBeUndefined();
	});

	it('非法或缺少的組別回傳 undefined', () => {
		expect(resolveSeasonGroup({ season: known.name, group: 'mixed' })).toBeUndefined();
		expect(resolveSeasonGroup({ season: known.name, group: undefined })).toBeUndefined();
	});
});

describe('parseId', () => {
	it('純數字字串轉成數字', () => {
		expect(parseId('12')).toBe(12);
		expect(parseId('0')).toBe(0);
		expect(parseId('007')).toBe(7);
	});

	it.each(['', '1e2', ' 5 ', 'abc', '-1'])('%j 回傳 undefined', (raw) => {
		expect(parseId(raw)).toBeUndefined();
	});
});

describe('requireSeasonGroup', () => {
	it('合法參數回傳 season 與 group', () => {
		expect(requireSeasonGroup({ season: known.name, group: 'women' }).group).toBe('women');
	});

	it('找不到時丟出 404', () => {
		expect.assertions(2);
		try {
			requireSeasonGroup({ season: 'no-such-season', group: 'men' });
		} catch (e) {
			const err = e as { status: number; body: { message: string } };
			expect(err.status).toBe(404);
			expect(err.body.message).toBe('找不到這個賽季或組別');
		}
	});
});
