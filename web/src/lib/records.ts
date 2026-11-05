import recordsJson from '#lib/data/records.json';
import type {
	CumulativeRecordRow,
	GroupRecords,
	MatchRecordRow,
	RecordBoard,
	RecordsCoverage,
	RecordsData,
	SeasonRecordRow
} from '#lib/data/types.js';
import { formatThroughDate } from '#lib/leaders.js';
import { teamOf, type Group, type SeasonTeam } from '#lib/site.js';

export type RecordKind = 'season' | 'cumulative' | 'match';
export type RecordKindInfo = { key: RecordKind; label: string; path: '' | 'cumulative' | 'match'; intro: string };
export const RECORD_KINDS: RecordKindInfo[] = [
	{ key: 'season', label: '單季紀錄', path: '', intro: '一名球員在一個賽季內的總和，同一人在不同賽季可各自上榜。' },
	{
		key: 'cumulative',
		label: '累計紀錄',
		path: 'cumulative',
		intro: '一名球員在資料涵蓋的所有賽季內的總和，每人只列一次；隊伍為最近一季所屬隊伍。'
	},
	{
		key: 'match',
		label: '單場紀錄',
		path: 'match',
		intro: '一名球員在一場賽事中的數據，同一人可多場上榜；點日期與對手可進入該場賽事的單場數據。'
	}
];

export function kindFromParam(param: string | undefined): RecordKindInfo | undefined {
	return RECORD_KINDS.find((k) => k.path === (param ?? ''));
}

export type RecordRow = SeasonRecordRow | CumulativeRecordRow | MatchRecordRow;
export type RecordRowView<T extends RecordRow = RecordRow> = T & { team: SeasonTeam };
export type RecordBoardView = Omit<RecordBoard, 'season_records' | 'cumulative_records' | 'match_records'> & {
	seasonRecords: RecordRowView<SeasonRecordRow>[];
	cumulativeRecords: RecordRowView<CumulativeRecordRow>[];
	matchRecords: RecordRowView<MatchRecordRow>[];
};

export function recordsFor(group: Group): GroupRecords | undefined {
	return (recordsJson as RecordsData).records.find((r) => r.group === group);
}

export function coverageText(coverage: NonNullable<RecordsCoverage>): string {
	return `資料自 ${coverage.first_season} 起，共 ${coverage.season_count} 季`;
}

export function toBoardView(board: RecordBoard, group: Group): RecordBoardView {
	const { season_records, cumulative_records, match_records, ...rest } = board;
	const withTeam = <T extends RecordRow>(row: T): RecordRowView<T> => ({
		...row,
		team: teamOf(row.team_id, group, row.cup_id, '歷史紀錄')
	});
	return {
		...rest,
		seasonRecords: season_records.map(withTeam),
		cumulativeRecords: cumulative_records.map(withTeam),
		matchRecords: match_records.map(withTeam)
	};
}

export function rowsOf(board: RecordBoardView, kind: RecordKind): RecordRowView[] {
	if (kind === 'cumulative') return board.cumulativeRecords;
	if (kind === 'match') return board.matchRecords;
	return board.seasonRecords;
}

const isCumulative = (row: RecordRow): row is CumulativeRecordRow => 'season_count' in row;
export const isMatch = (row: RecordRow): row is MatchRecordRow => 'match_date' in row;

export function rowKey(row: RecordRow): string {
	const kind = isCumulative(row) ? 'c' : isMatch(row) ? `m${row.match_date}` : 's';
	return `${kind}-${row.player_id}-${row.cup_id}`;
}

export function rowDetail(row: RecordRow): string {
	if (isCumulative(row)) return `${row.season_count} 季 · 出賽 ${row.games} 場 · 最近一季 ${row.team_short}`;
	if (isMatch(row)) return `${row.season} · ${row.team_short}`;
	return `${row.season} · ${row.team_short} · 出賽 ${row.games} 場`;
}

export function matchLabel(row: MatchRecordRow): string {
	return `${formatThroughDate(row.match_date)} 對 ${row.opponent}`;
}

export function liveLabel(row: RecordRow): string {
	return isCumulative(row) ? '含進行中賽季' : '進行中';
}
