import { describe, expect, it } from 'vitest';
import { NAV_ITEMS, detailListOf, listHref, nextDepth, sectionOf, tabOf } from '#lib/nav.js';

describe('sectionOf', () => {
	it.each([
		['/[season]/[group]', 'home'],
		['/[season]/[group]/matches', 'matches'],
		['/[season]/[group]/matches/[game]', 'matches'],
		['/[season]/[group]/standings', 'standings'],
		['/[season]/[group]/teams', 'teams'],
		['/[season]/[group]/teams/[team]', 'teams'],
		['/[season]/[group]/players', 'players'],
		['/[season]/[group]/players/[player]', 'players'],
		['/[season]/[group]/leaders', 'leaders'],
		['/records/[group]/[[kind=recordKind]]', 'records']
	])('%s 屬於 %s', (routeId, section) => {
		expect(sectionOf(routeId)).toBe(section);
	});

	it.each(['/', '/404', '/styleguide', null])('%s 沒有區段', (routeId) => {
		expect(sectionOf(routeId)).toBeNull();
	});
});

describe('tabOf', () => {
	it.each([
		['/[season]/[group]', 'home'],
		['/[season]/[group]/matches/[game]', 'matches'],
		['/[season]/[group]/standings', 'standings'],
		['/[season]/[group]/teams/[team]', 'teams'],
		['/[season]/[group]/players/[player]', 'more'],
		['/[season]/[group]/leaders', 'more'],
		['/records/[group]/[[kind=recordKind]]', 'more']
	])('%s 對應分頁 %s', (routeId, tab) => {
		expect(tabOf(routeId)).toBe(tab);
	});

	it('沒有區段時沒有目前分頁', () => {
		expect(tabOf('/404')).toBeNull();
		expect(tabOf(null)).toBeNull();
	});
});

describe('detailListOf', () => {
	it.each([
		['/[season]/[group]/matches/[game]', 'matches'],
		['/[season]/[group]/teams/[team]', 'teams'],
		['/[season]/[group]/players/[player]', 'players']
	])('%s 的上一層是 %s 列表', (routeId, list) => {
		expect(detailListOf(routeId)).toBe(list);
	});

	it.each(['/[season]/[group]', '/[season]/[group]/matches', '/[season]/[group]/teams', '/records/[group]/[[kind=recordKind]]', '/', null])(
		'%s 不是詳細頁',
		(routeId) => {
			expect(detailListOf(routeId)).toBeNull();
		}
	);
});

describe('listHref', () => {
	const base = '/2025-26/men/';

	it('單場頁到賽事列表的該月份錨點', () => {
		expect(listHref('matches', base, '2025-12-07')).toBe('/2025-26/men/matches/#m-2025-12');
	});

	it('賽事列表沒有日期時不帶錨點', () => {
		expect(listHref('matches', base)).toBe('/2025-26/men/matches/');
	});

	it('球隊頁到球隊列表', () => {
		expect(listHref('teams', base)).toBe('/2025-26/men/teams/');
	});

	it('球員頁到球員列表', () => {
		expect(listHref('players', base)).toBe('/2025-26/men/players/');
	});
});

describe('nextDepth', () => {
	it('進站歸零', () => {
		expect(nextDepth(3, 'enter')).toBe(0);
	});

	it('站內導頁累加', () => {
		expect(nextDepth(0, 'link')).toBe(1);
		expect(nextDepth(1, 'goto')).toBe(2);
	});

	it('瀏覽紀錄後退與前進跟著增減', () => {
		expect(nextDepth(2, 'popstate', -1)).toBe(1);
		expect(nextDepth(1, 'popstate', 1)).toBe(2);
	});

	it('後退超過進站點時不低於零', () => {
		expect(nextDepth(0, 'popstate', -1)).toBe(0);
	});
});

describe('NAV_ITEMS', () => {
	it('七項依序為首頁、賽事、戰績、球隊、球員、排行榜、歷史紀錄', () => {
		expect(NAV_ITEMS.map((i) => i.label)).toEqual(['首頁', '賽事', '戰績', '球隊', '球員', '排行榜', '歷史紀錄']);
	});
});
