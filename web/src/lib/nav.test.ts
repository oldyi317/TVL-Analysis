import { describe, expect, it } from 'vitest';
import { NAV_ITEMS, sectionOf, tabOf } from '#lib/nav.js';

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

describe('NAV_ITEMS', () => {
	it('七項依序為首頁、賽事、戰績、球隊、球員、排行榜、歷史紀錄', () => {
		expect(NAV_ITEMS.map((i) => i.label)).toEqual(['首頁', '賽事', '戰績', '球隊', '球員', '排行榜', '歷史紀錄']);
	});
});
