import { describe, expect, it } from 'vitest';
import { pickSeason } from './pick-season.js';

const seasons = [
	{ cup_id: 20, name: '2024-25' },
	{ cup_id: 21, name: '2025-26' },
	{ cup_id: 22, name: '2026-27' }
];
const match = (sets: number[][]) => ({ sets });
const src = (rows: unknown[], currentCupId = 22) => ({ rows, seasons, currentCupId });

describe('pickSeason', () => {
	it('當季有資料時優先當季', () => {
		const rows = [
			{ cup_id: 21, group: 'men', matches: [match([[25, 20]])] },
			{ cup_id: 22, group: 'men', matches: [match([[25, 20]])] }
		];
		expect(pickSeason('played', 'men', src(rows))).toEqual({ cupId: 22, season: '2026-27' });
	});

	it('當季沒有資料時退回有資料的最大 cup_id', () => {
		const rows = [
			{ cup_id: 20, group: 'men', matches: [match([[25, 20]])] },
			{ cup_id: 21, group: 'men', matches: [match([[25, 20]])] },
			{ cup_id: 22, group: 'men', matches: [match([])] }
		];
		expect(pickSeason('played', 'men', src(rows))).toEqual({ cupId: 21, season: '2025-26' });
	});

	it('完全沒有資料時回 undefined', () => {
		const rows = [{ cup_id: 22, group: 'men', matches: [match([])] }];
		expect(pickSeason('played', 'men', src(rows))).toBeUndefined();
		expect(pickSeason('players', 'men', src([]))).toBeUndefined();
	});

	it('組別互相隔離', () => {
		const rows = [
			{ player_id: 1, group: 'women', seasons: [{ cup_id: 22 }] },
			{ player_id: 2, group: 'men', seasons: [{ cup_id: 21 }] }
		];
		expect(pickSeason('players', 'men', src(rows))).toEqual({ cupId: 21, season: '2025-26' });
		expect(pickSeason('players', 'women', src(rows))).toEqual({ cupId: 22, season: '2026-27' });
	});

	it('球隊與週登錄各看自己的欄位', () => {
		const teams = [{ team_id: 1, group: 'men', cup_ids: [20, 21] }];
		expect(pickSeason('teams', 'men', src(teams))).toEqual({ cupId: 21, season: '2025-26' });
		const rosters = [
			{ cup_id: 21, group: 'men', team_id: 1, weeks: [{}] },
			{ cup_id: 22, group: 'men', team_id: 1, weeks: [] }
		];
		expect(pickSeason('rosters', 'men', src(rosters))).toEqual({ cupId: 21, season: '2025-26' });
	});
});
