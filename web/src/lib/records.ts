import recordsJson from '#lib/data/records.json';
import type { CumulativeRecordRow, GroupRecords, RecordBoard, RecordsCoverage, RecordsData, SeasonRecordRow } from '#lib/data/types.js';
import { teamOf, type Group, type SeasonTeam } from '#lib/site.js';

export type RecordKind = 'season' | 'cumulative';
export type RecordKindInfo = { key: RecordKind; label: string; path: '' | 'cumulative'; intro: string };
export const RECORD_KINDS: RecordKindInfo[] = [
	{ key: 'season', label: '單季紀錄', path: '', intro: '一名球員在一個賽季內的總和，同一人在不同賽季可各自上榜。' },
	{
		key: 'cumulative',
		label: '累計紀錄',
		path: 'cumulative',
		intro: '一名球員在資料涵蓋的所有賽季內的總和，每人只列一次；隊伍為最近一季所屬隊伍。'
	}
];

export function kindFromParam(param: string | undefined): RecordKindInfo | undefined {
	return RECORD_KINDS.find((k) => k.path === (param ?? ''));
}

export type RecordRow = SeasonRecordRow | CumulativeRecordRow;
export type RecordRowView<T extends RecordRow = RecordRow> = T & { team: SeasonTeam };
export type RecordBoardView = Omit<RecordBoard, 'season_records' | 'cumulative_records'> & {
	seasonRecords: RecordRowView<SeasonRecordRow>[];
	cumulativeRecords: RecordRowView<CumulativeRecordRow>[];
};

export function recordsFor(group: Group): GroupRecords | undefined {
	return (recordsJson as RecordsData).records.find((r) => r.group === group);
}

export function coverageText(coverage: NonNullable<RecordsCoverage>): string {
	return `資料自 ${coverage.first_season} 起，共 ${coverage.season_count} 季`;
}

export function toBoardView(board: RecordBoard, group: Group): RecordBoardView {
	const { season_records, cumulative_records, ...rest } = board;
	const withTeam = <T extends RecordRow>(row: T): RecordRowView<T> => ({
		...row,
		team: teamOf(row.team_id, group, row.cup_id, '歷史紀錄')
	});
	return {
		...rest,
		seasonRecords: season_records.map(withTeam),
		cumulativeRecords: cumulative_records.map(withTeam)
	};
}

const isCumulative = (row: RecordRow): row is CumulativeRecordRow => 'season_count' in row;

export function rowDetail(row: RecordRow): string {
	if (isCumulative(row)) return `${row.season_count} 季 · 出賽 ${row.games} 場 · 最近一季 ${row.team_short}`;
	return `${row.season} · ${row.team_short} · 出賽 ${row.games} 場`;
}

export function liveLabel(row: RecordRow): string {
	return isCumulative(row) ? '含進行中賽季' : '進行中';
}
