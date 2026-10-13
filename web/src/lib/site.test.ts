import { describe, expect, it } from 'vitest';
import type { StandingsEntry } from '#lib/data/types.js';
import { standingsFor, teams, toStandingsRow } from './site.js';

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
		const real = teams.find((t) => t.group === 'women')!;
		const row = toStandingsRow({ ...entry, team_id: real.team_id }, 'women');
		expect(row.team).toBe(real);
		expect(row).toMatchObject({ rank: 1, wins: 5, losses: 1, points: 14, set_ratio: null });
		expect(row).not.toHaveProperty('team_id');
		expect(row.last5).toEqual(['W', 'W', 'L', 'W', 'W']);
	});
	it('帶過勝率、局數與得失分欄位', () => {
		const real = teams.find((t) => t.group === 'men')!;
		const row = toStandingsRow({ ...entry, team_id: real.team_id }, 'men');
		expect(row).toMatchObject({
			win_pct: 0.833,
			sets_won: 16,
			sets_lost: 7,
			points_for: 400,
			points_against: 350
		});
	});
	it('對不到球隊就丟錯', () => {
		expect(() => toStandingsRow({ ...entry, team_id: -1 }, 'men')).toThrow('team_id -1');
	});
});

describe('standingsFor', () => {
	it('找不到的賽季回 undefined', () => {
		expect(standingsFor(-1, 'men')).toBeUndefined();
	});
});
