import recordsJson from '#lib/data/records.json';
import type { GroupRecords, RecordBoard, RecordsCoverage, RecordsData, SeasonRecordRow } from '#lib/data/types.js';
import { teamOf, type Group, type SeasonTeam } from '#lib/site.js';

export type SeasonRecordRowView = SeasonRecordRow & { team: SeasonTeam };
export type RecordBoardView = Omit<RecordBoard, 'season_records'> & { seasonRecords: SeasonRecordRowView[] };

export function recordsFor(group: Group): GroupRecords | undefined {
	return (recordsJson as RecordsData).records.find((r) => r.group === group);
}

export function coverageText(coverage: NonNullable<RecordsCoverage>): string {
	return `資料自 ${coverage.first_season} 起，共 ${coverage.season_count} 季`;
}

export function toBoardView(board: RecordBoard, group: Group): RecordBoardView {
	const { season_records, ...rest } = board;
	return { ...rest, seasonRecords: season_records.map((row) => ({ ...row, team: teamOf(row.team_id, group, row.cup_id, '歷史紀錄') })) };
}

export function rowDetail(row: SeasonRecordRow): string {
	return `${row.season} · ${row.team_short} · 出賽 ${row.games} 場`;
}
