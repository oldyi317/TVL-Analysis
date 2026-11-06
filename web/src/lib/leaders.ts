import leadersJson from '#lib/data/leaders.json';
import type { GroupLeaders, LeaderBoard, LeaderRow, LeadersData, VolumeRate, VolumeRatePoint } from '#lib/data/types.js';
import { teamOf, type Group, type SeasonTeam } from '#lib/site.js';
import { STAT_LABELS } from '#lib/statLabels.js';

export const TOP_N = 10;

export type LeaderRowView = LeaderRow & { team: SeasonTeam };
export type LeaderBoardView = Omit<LeaderBoard, 'rows'> & { rows: LeaderRowView[] };

export function leadersFor(cupId: number, group: Group): GroupLeaders | undefined {
	return (leadersJson as LeadersData).leaders.find((s) => s.cup_id === cupId && s.group === group);
}

export function toBoardView(board: LeaderBoard, group: Group, cupId: number): LeaderBoardView {
	return { ...board, rows: board.rows.map((row) => ({ ...row, team: teamOf(row.team_id, group, cupId, '排行榜') })) };
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

export type VolumeRateKey = VolumeRate['key'];

export const VOLUME_RATE_TECHNIQUES: readonly { key: VolumeRateKey; label: string }[] = [
	{ key: 'attack_success_rate', label: '攻擊' },
	{ key: 'receive_excellent_rate', label: '接發球' },
	{ key: 'dig_excellent_rate', label: '防守' },
	{ key: 'set_excellent_rate', label: '舉球' }
];

export type VolumeRatePointView = VolumeRatePoint & { team: string };
export type VolumeRateView = Omit<VolumeRate, 'points'> & {
	technique: string;
	countLabel: string;
	rateLabel: string;
	points: VolumeRatePointView[];
};

export function toVolumeRateView(v: VolumeRate, group: Group, cupId: number): VolumeRateView {
	const technique = VOLUME_RATE_TECHNIQUES.find((t) => t.key === v.key)!.label;
	return {
		...v,
		technique,
		countLabel: `${technique}次數`,
		rateLabel: STAT_LABELS[v.key],
		points: v.points.map((p) => ({ ...p, team: teamOf(p.team_id, group, cupId, '排行榜').short_name }))
	};
}

type VolumeRateDatum = { value: [number, number]; name: string; team: string };

export function volumeRateOption(v: VolumeRateView): object | null {
	if (v.points.length === 0) return null;
	const data: VolumeRateDatum[] = v.points.map((p) => ({ value: [p.count, p.rate], name: p.name, team: p.team }));
	return {
		legend: { show: false },
		grid: { top: 32, right: 24, bottom: 32 },
		tooltip: {
			trigger: 'item',
			formatter: ({ data: d }: { data: VolumeRateDatum }) =>
				`${d.name}（${d.team}）<br/>${v.countLabel}：${d.value[0]}<br/>${v.rateLabel}：${d.value[1]}%`
		},
		xAxis: { type: 'value', name: v.countLabel, nameLocation: 'middle', nameGap: 28, scale: true },
		yAxis: { type: 'value', name: `${v.rateLabel}（%）`, nameTextStyle: { align: 'left' }, scale: true },
		series: [
			{
				type: 'scatter',
				name: v.rateLabel,
				data,
				markLine: {
					silent: true,
					symbol: 'none',
					lineStyle: { type: 'dashed' },
					label: { show: false },
					data: [{ xAxis: v.median_count }, { yAxis: v.median_rate }]
				}
			}
		]
	};
}

export function volumeRateNote(v: VolumeRateView, threshold: string | null): string {
	const scope = threshold ? `只列達比率榜門檻的球員，${threshold.replace(/^門檻：/, '')}；` : '';
	return `${scope}虛線是兩軸中位數（${v.countLabel} ${v.median_count}、${v.rateLabel} ${v.median_rate}%），右上是量大又準、左下是量少又不準。`;
}
