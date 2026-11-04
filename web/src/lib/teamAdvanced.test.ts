import { describe, expect, it } from 'vitest';
import type { TeamAdvanced, TeamAdvancedData, TeamMetric, TeamMetrics } from '#lib/data/types.js';
import { metricsConclusion, metricsDotOption, metricsTable, teamComparison } from './teamAdvanced.js';

const metric = (value: number | null, rank: number | null): TeamMetric => ({
	numerator: 0,
	denominator: value === null ? 0 : 10,
	value,
	rank
});

const team = (team_id: number, ranks: Partial<Record<keyof TeamMetrics, number | null>>, games = 3): TeamAdvanced => {
	const m = (key: keyof TeamMetrics) => {
		const rank = ranks[key] === undefined ? 1 : ranks[key];
		return metric(rank === null ? null : 60 - rank * 10, rank);
	};
	return {
		team_id,
		games,
		metrics: {
			attack_success_rate: m('attack_success_rate'),
			serve_point_rate: m('serve_point_rate'),
			receive_excellent_rate: m('receive_excellent_rate'),
			dig_excellent_rate: m('dig_excellent_rate'),
			set_excellent_rate: m('set_excellent_rate'),
			blocks_per_set: m('blocks_per_set')
		}
	};
};

const data = (teams: TeamAdvanced[]): TeamAdvancedData => ({
	team_advanced: [{ cup_id: 21, group: 'men', teams }]
});

const others = (ranks: number[]) => ranks.map((r, i) => team(100 + i, {
	attack_success_rate: r, serve_point_rate: r, receive_excellent_rate: r,
	dig_excellent_rate: r, set_excellent_rate: r, blocks_per_set: r
}));

describe('teamComparison', () => {
	it('依賽季、組別、球隊找出本隊與組內各隊', () => {
		const t = team(4, {});
		const c = teamComparison(data([t, team(1, {})]), 21, 'men', 4);
		expect(c?.team).toBe(t);
		expect(c?.teams).toHaveLength(2);
		expect(teamComparison(data([t]), 21, 'women', 4)).toBeUndefined();
		expect(teamComparison(data([t]), 21, 'men', 9)).toBeUndefined();
	});
});

describe('metricsConclusion', () => {
	it('取名次最好的一項加上最差的一項', () => {
		const t = team(4, {
			attack_success_rate: 5, serve_point_rate: 3, receive_excellent_rate: 1,
			dig_excellent_rate: 2, set_excellent_rate: 4, blocks_per_set: 3
		});
		const c = teamComparison(data([t, ...others([2, 3, 4, 6, 6])]), 21, 'men', 4)!;
		expect(metricsConclusion(c)).toBe('接發球到位率 第 1 / 6，攻擊成功率 第 5 / 6');
	});

	it('名次並列時寫並列', () => {
		const t = team(4, {
			attack_success_rate: 1, serve_point_rate: 2, receive_excellent_rate: 2,
			dig_excellent_rate: 2, set_excellent_rate: 2, blocks_per_set: 3
		});
		const tied = team(5, {
			attack_success_rate: 1, serve_point_rate: 1, receive_excellent_rate: 1,
			dig_excellent_rate: 1, set_excellent_rate: 1, blocks_per_set: 3
		});
		const c = teamComparison(data([t, tied, ...others([4])]), 21, 'men', 4)!;
		expect(metricsConclusion(c)).toBe('攻擊成功率 並列第 1 / 3，每局攔網 並列第 3 / 3');
	});

	it('分母沒有值的指標不列入，名次分母只數有值的隊', () => {
		const t = team(4, {
			attack_success_rate: 2, serve_point_rate: 2, receive_excellent_rate: 2,
			dig_excellent_rate: 2, set_excellent_rate: null, blocks_per_set: 1
		});
		const a = team(1, {
			attack_success_rate: 1, serve_point_rate: 3, receive_excellent_rate: 3,
			dig_excellent_rate: 3, set_excellent_rate: 1, blocks_per_set: 3
		});
		const b = team(2, {
			attack_success_rate: null, serve_point_rate: 4, receive_excellent_rate: 4,
			dig_excellent_rate: 4, set_excellent_rate: 2, blocks_per_set: 4
		});
		const c = teamComparison(data([t, a, b]), 21, 'men', 4)!;
		expect(metricsConclusion(c)).toBe('每局攔網 第 1 / 3，攻擊成功率 第 2 / 2');
	});

	it('各項名次都相同時合成一句', () => {
		const c = teamComparison(data([team(4, {}), ...others([2])]), 21, 'men', 4)!;
		expect(metricsConclusion(c)).toBe('各項球隊指標都是第 1 / 2');
	});

	it('已賽未滿 3 場不產生', () => {
		const c = teamComparison(data([team(4, {}, 2), ...others([2])]), 21, 'men', 4)!;
		expect(metricsConclusion(c)).toBeNull();
	});
});

type DotPoint = { value: [number, number]; name: string; itemStyle: { opacity: number } };
type DotOption = {
	yAxis: { data: string[] };
	series: { data: DotPoint[] }[];
	tooltip: { formatter: (p: { data: DotPoint }) => string };
};

describe('metricsDotOption', () => {
	const withAttack = (team_id: number, value: number | null, rank: number | null) => {
		const t = team(team_id, {});
		t.metrics.attack_success_rate = metric(value, rank);
		return t;
	};
	const names: Record<number, string> = { 1: '屏東台電', 2: '雲林美津濃', 4: '臺北鳳凰', 5: '龍邁建設' };
	const option = (teams: TeamAdvanced[]) =>
		metricsDotOption(teamComparison(data(teams), 21, 'men', 4)!, (id) => names[id]) as DotOption;
	const attackRow = (o: DotOption) => o.series[0].data.filter((p) => p.value[1] === 0);

	it('每列一項指標，依固定順序', () => {
		expect(option([team(4, {})]).yAxis.data).toEqual([
			'攻擊成功率', '發球得分率', '接發球到位率', '防守到位率', '舉球到位率', '每局攔網'
		]);
	});

	it('各列以組內最低到最高換算位置，沒有值的隊不畫', () => {
		const o = option([withAttack(4, 45, 2), withAttack(1, 50, 1), withAttack(2, 40, 3), withAttack(5, null, null)]);
		const x = Object.fromEntries(attackRow(o).map((p) => [p.name, p.value[0]]));
		expect(x).toEqual({ 臺北鳳凰: 50, 屏東台電: 100, 雲林美津濃: 0 });
	});

	it('組內各隊同值時置中', () => {
		const o = option([withAttack(4, 45, 1), withAttack(1, 45, 1)]);
		expect(attackRow(o).map((p) => p.value[0])).toEqual([50, 50]);
	});

	it('本隊強調並畫在最上層，其他隊淡化', () => {
		const row = attackRow(option([withAttack(4, 45, 2), withAttack(1, 50, 1)]));
		expect(row.map((p) => [p.name, p.itemStyle.opacity])).toEqual([['屏東台電', 0.35], ['臺北鳳凰', 1]]);
	});

	it('tooltip 顯示隊名、數值與組內名次', () => {
		const o = option([withAttack(4, 45, 2), withAttack(1, 50, 1)]);
		const phoenix = attackRow(o).find((p) => p.name === '臺北鳳凰')!;
		expect(o.tooltip.formatter({ data: phoenix })).toBe('臺北鳳凰<br/>攻擊成功率：45%（第 2 / 2）');
		const blocks = o.series[0].data.find((p) => p.value[1] === 5 && p.name === '屏東台電')!;
		expect(o.tooltip.formatter({ data: blocks })).toBe('屏東台電<br/>每局攔網：50（並列第 1 / 2）');
	});
});

describe('metricsTable', () => {
	it('每隊一列、每項指標一欄，本隊在最前，其他隊依資料順序', () => {
		const t = team(4, { attack_success_rate: 2, set_excellent_rate: null });
		const names: Record<number, string> = { 4: '臺北鳳凰', 100: '甲隊', 101: '乙隊' };
		const rows = metricsTable(teamComparison(data([...others([1, 3]), t]), 21, 'men', 4)!, (id) => names[id]);
		expect(rows.map((r) => [r.name, r.isTeam])).toEqual([['臺北鳳凰', true], ['甲隊', false], ['乙隊', false]]);
		expect(rows[0].cells[0]).toEqual({ key: 'attack_success_rate', value: '40%', rank: '第 2 / 3' });
		expect(rows[0].cells[4]).toEqual({ key: 'set_excellent_rate', value: '—', rank: '—' });
		expect(rows[1].cells[0]).toEqual({ key: 'attack_success_rate', value: '50%', rank: '第 1 / 3' });
	});
});
