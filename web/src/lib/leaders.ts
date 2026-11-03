import leadersJson from '#lib/data/leaders.json';
import type { GroupLeaders, LeaderBoard, LeaderRow, LeadersData } from '#lib/data/types.js';
import { seasonTeam, teams, type Group, type SeasonTeam } from '#lib/site.js';

export const TOP_N = 10;

export type LeaderRowView = LeaderRow & { team: SeasonTeam };
export type LeaderBoardView = Omit<LeaderBoard, 'rows'> & { rows: LeaderRowView[] };

export function leadersFor(cupId: number, group: Group): GroupLeaders | undefined {
	return (leadersJson as LeadersData).leaders.find((s) => s.cup_id === cupId && s.group === group);
}

// 對不到球隊代表資料檔壞了：直接丟錯讓 build 失敗，不默默少一列
export function toBoardView(board: LeaderBoard, group: Group, cupId: number): LeaderBoardView {
	const rows = board.rows.map((row) => {
		const team = teams.find((t) => t.group === group && t.team_id === row.team_id);
		if (!team) throw new Error(`排行榜的 team_id ${row.team_id}（${group}）不在 teams.json`);
		return { ...row, team: seasonTeam(team, cupId) };
	});
	return { ...board, rows };
}

// 前 10 名截斷；第 10 名並列時全部顯示
export function visibleRows<T extends { rank: number }>(rows: readonly T[], limit = TOP_N): T[] {
	if (rows.length <= limit) return [...rows];
	const cutoff = rows[limit - 1].rank;
	return rows.filter((r) => r.rank <= cutoff);
}

export function formatThroughDate(iso: string): string {
	const [, month, day] = iso.split('-').map(Number);
	return `${month} 月 ${day} 日`;
}
