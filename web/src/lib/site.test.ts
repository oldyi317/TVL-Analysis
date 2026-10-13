import { describe, expect, it } from 'vitest';
import type { StandingsEntry, Team } from '#lib/data/types.js';
import { standingsFor, toStandingsRow } from './site.js';

const team = (team_id: number, group: 'men' | 'women'): Team => ({
	team_id,
	group,
	name: `隊${team_id}`,
	short_name: `隊${team_id}`,
	color: '#123456',
	is_active: true,
	cup_ids: [21]
});

const entry: StandingsEntry = {
	rank: 1,
	team_id: 3,
	wins: 5,
	losses: 1,
	win_pct: 0.833,
	set_record: { '3-0': 2, '3-1': 2, '3-2': 1, '2-3': 0, '1-3': 1, '0-3': 0 },
	points: 14,
	sets_won: 16,
	sets_lost: 7,
	set_ratio: null,
	points_for: 400,
	points_against: 350,
	point_ratio: 1.143,
	last5: ['W', 'W', 'L', 'W', 'W']
};

describe('toStandingsRow', () => {
	it('以 group + team_id 對回球隊並帶過數值', () => {
		const list = [team(3, 'men'), team(3, 'women')];
		const row = toStandingsRow(entry, 'women', list);
		expect(row.team).toBe(list[1]);
		expect(row).toMatchObject({ rank: 1, wins: 5, losses: 1, points: 14, set_ratio: null });
		expect(row.last5).toEqual(['W', 'W', 'L', 'W', 'W']);
	});
	it('對不到球隊就丟錯', () => {
		expect(() => toStandingsRow(entry, 'men', [team(4, 'men')])).toThrow('team_id 3');
	});
});

describe('standingsFor', () => {
	it('找不到的賽季回 undefined', () => {
		expect(standingsFor(-1, 'men')).toBeUndefined();
	});
});
