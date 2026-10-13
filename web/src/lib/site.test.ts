import { describe, expect, it } from 'vitest';
import type { MatchEntry, StandingsEntry } from '#lib/data/types.js';
import { findMatch, matchesFor, standingsFor, teams, toMatchView, toStandingsRow } from './site.js';

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

describe('toMatchView', () => {
	const base: MatchEntry = {
		game_id: 327,
		date: '2026-03-22',
		weekday: '日',
		start_time: null,
		venue: '國立臺灣大學綜合體育館',
		stage: '總決賽',
		week: 19,
		home_team_id: 1,
		away_team_id: 2,
		sets: [[25, 20], [25, 18], [25, 22]],
		home_sets_won: 3,
		away_sets_won: 0,
		golden_set: [13, 15]
	};
	it('以 group + team_id 對回兩隊並帶過比分、階段與黃金決勝局', () => {
		const men = teams.filter((t) => t.group === 'men');
		const view = toMatchView({ ...base, home_team_id: men[0].team_id, away_team_id: men[1].team_id }, 'men');
		expect(view.home).toBe(men[0]);
		expect(view.away).toBe(men[1]);
		expect(view).toMatchObject({
			game_id: 327,
			date: '2026-03-22',
			weekday: '日',
			venue: '國立臺灣大學綜合體育館',
			stage: '總決賽',
			week: 19,
			golden_set: [13, 15],
			start_time: null
		});
		expect(view.sets).toEqual([[25, 20], [25, 18], [25, 22]]);
	});
	it('場館缺值顯示「場館未定」', () => {
		const men = teams.filter((t) => t.group === 'men');
		const view = toMatchView(
			{ ...base, venue: null, home_team_id: men[0].team_id, away_team_id: men[1].team_id },
			'men'
		);
		expect(view.venue).toBe('場館未定');
	});
	it('對不到球隊就丟錯', () => {
		expect(() => toMatchView({ ...base, home_team_id: -1 }, 'men')).toThrow('team_id -1');
	});
});

describe('matchesFor / findMatch', () => {
	it('找不到的賽季回空陣列與 undefined', () => {
		expect(matchesFor(-1, 'men')).toEqual([]);
		expect(findMatch(-1, 'men', 1)).toBeUndefined();
	});
});
