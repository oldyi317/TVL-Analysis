import { describe, expect, it } from 'vitest';
import type { RecordBoard, SeasonRecordRow } from '#lib/data/types.js';
import { teams } from '#lib/site.js';
import { coverageText, recordsFor, rowDetail, toBoardView } from './records.js';

const men = teams.find((t) => t.group === 'men')!;
const row = (over: Partial<SeasonRecordRow> = {}): SeasonRecordRow => ({
	rank: 1, player_id: 7, name: '王小明', value: 30, cup_id: 21, season: '2025-26', games: 12,
	team_id: men.team_id, team_short: men.short_name, in_progress: false, ...over
});

describe('歷史紀錄資料整理', () => {
	it('男女組各有一筆', () => {
		expect(recordsFor('men')?.group).toBe('men');
		expect(recordsFor('women')?.group).toBe('women');
	});

	it('涵蓋範圍文字由資料組出', () => {
		expect(coverageText({ first_season: '2024-25', season_count: 3, through_date: '2026-11-01' })).toBe(
			'資料自 2024-25 起，共 3 季'
		);
	});

	it('列附上當季隊伍，欄位改名為 seasonRecords', () => {
		const board: RecordBoard = { key: 'total_points', label: '總得分', season_records: [row()] };
		const view = toBoardView(board, 'men');
		expect(view.seasonRecords[0].team.team_id).toBe(men.team_id);
		expect(view.seasonRecords[0].team.cup_id).toBe(21);
		expect('season_records' in view).toBe(false);
	});

	it('對不到球隊就丟錯', () => {
		const board: RecordBoard = { key: 'total_points', label: '總得分', season_records: [row({ team_id: -1 })] };
		expect(() => toBoardView(board, 'men')).toThrow('team_id -1');
	});

	it('列說明含賽季、當季隊名、出賽場數', () => {
		expect(rowDetail(row({ team_short: '桃園臺灣產險' }))).toBe('2025-26 · 桃園臺灣產險 · 出賽 12 場');
	});
});
