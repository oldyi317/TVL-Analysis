import { describe, expect, it } from 'vitest';
import type { CumulativeRecordRow, RecordBoard, SeasonRecordRow } from '#lib/data/types.js';
import { teams } from '#lib/site.js';
import { params } from '../params.js';
import { coverageText, kindFromParam, liveLabel, RECORD_KINDS, recordsFor, rowDetail, toBoardView } from './records.js';

const men = teams.find((t) => t.group === 'men')!;
const row = (over: Partial<SeasonRecordRow> = {}): SeasonRecordRow => ({
	rank: 1, player_id: 7, name: '王小明', value: 30, cup_id: 21, season: '2025-26', games: 12,
	team_id: men.team_id, team_short: men.short_name, in_progress: false, ...over
});
const cumulativeRow = (over: Partial<CumulativeRecordRow> = {}): CumulativeRecordRow => ({
	...row({ cup_id: 22, season: '2026-27', games: 40 }), season_count: 3, ...over
});
const board = (over: Partial<RecordBoard> = {}): RecordBoard => ({
	key: 'total_points', label: '總得分', season_records: [row()], cumulative_records: [cumulativeRow()], ...over
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

	it('列附上當季隊伍，欄位改名為 seasonRecords、cumulativeRecords', () => {
		const view = toBoardView(board(), 'men');
		expect(view.seasonRecords[0].team.team_id).toBe(men.team_id);
		expect(view.seasonRecords[0].team.cup_id).toBe(21);
		expect('season_records' in view).toBe(false);
	});

	it('累計紀錄列附上最近一季的隊伍', () => {
		const view = toBoardView(board(), 'men');
		expect(view.cumulativeRecords[0].team.team_id).toBe(men.team_id);
		expect(view.cumulativeRecords[0].team.cup_id).toBe(22);
	});

	it('對不到球隊就丟錯', () => {
		expect(() => toBoardView(board({ season_records: [row({ team_id: -1 })] }), 'men')).toThrow('team_id -1');
		expect(() => toBoardView(board({ cumulative_records: [cumulativeRow({ team_id: -2 })] }), 'men')).toThrow('team_id -2');
	});

	it('列說明含賽季、當季隊名、出賽場數', () => {
		expect(rowDetail(row({ team_short: '桃園臺灣產險' }))).toBe('2025-26 · 桃園臺灣產險 · 出賽 12 場');
	});

	it('累計紀錄列說明含涵蓋季數、出賽場數、最近一季隊名', () => {
		expect(rowDetail(cumulativeRow({ team_short: '臺北鳳凰' }))).toBe('3 季 · 出賽 40 場 · 最近一季 臺北鳳凰');
	});

	it('進行中標示依紀錄類型用詞', () => {
		expect(liveLabel(row())).toBe('進行中');
		expect(liveLabel(cumulativeRow())).toBe('含進行中賽季');
	});

	it('紀錄類型依序為單季紀錄、累計紀錄，各有子路徑', () => {
		expect(RECORD_KINDS.map((k) => [k.label, k.path])).toEqual([['單季紀錄', ''], ['累計紀錄', 'cumulative']]);
	});

	it('路由參數對回紀錄類型，省略時為單季紀錄', () => {
		expect(kindFromParam(undefined)?.key).toBe('season');
		expect(kindFromParam('cumulative')?.key).toBe('cumulative');
		expect(kindFromParam('career')).toBeUndefined();
	});

	it('路由參數比對器只接受有子路徑的紀錄類型', () => {
		const accepts = (value: string) => {
			const result = params.recordKind['~standard'].validate(value);
			return !(result instanceof Promise) && !result.issues;
		};
		for (const k of RECORD_KINDS) expect(accepts(k.path)).toBe(k.path !== '');
		expect(accepts('career')).toBe(false);
	});
});
