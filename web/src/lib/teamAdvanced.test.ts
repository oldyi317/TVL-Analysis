import { describe, expect, it } from 'vitest';
import { teamPalette } from '#lib/color.js';
import type { TeamAdvanced, TeamAdvancedData, TeamGameMetrics, TeamMetric, TeamMetrics, TeamTrendGame } from '#lib/data/types.js';
import {
	metricsConclusion,
	metricsDotOption,
	metricsTable,
	pregameBarOption,
	pregameComparison,
	pregameConclusion,
	pregameTable,
	SCORE_SOURCES,
	sourcesBarOption,
	sourcesConclusion,
	sourcesTable,
	teamComparison,
	trendConclusion,
	trendTable,
	teamTrendOption
} from './teamAdvanced.js';

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
		points: { attack: 40, block: 10, serve: 10, opponent_error: 40, total: 100 },
		metrics: {
			attack_success_rate: m('attack_success_rate'),
			serve_point_rate: m('serve_point_rate'),
			receive_excellent_rate: m('receive_excellent_rate'),
			dig_excellent_rate: m('dig_excellent_rate'),
			set_excellent_rate: m('set_excellent_rate'),
			blocks_per_set: m('blocks_per_set')
		},
		trend: []
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

	it('tooltip 的隊名轉義成文字', () => {
		const c = teamComparison(data([withAttack(4, 45, 1)]), 21, 'men', 4)!;
		const o = metricsDotOption(c, () => '<img src=x onerror=alert(1)>') as DotOption;
		expect(o.tooltip.formatter({ data: attackRow(o)[0] })).toBe('&lt;img src=x onerror=alert(1)&gt;<br/>攻擊成功率：45%（第 1 / 1）');
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

const withPoints = (t: TeamAdvanced, p: Partial<TeamAdvanced['points']>): TeamAdvanced => {
	const points = { ...t.points, ...p };
	points.total = p.total ?? points.attack + points.block + points.serve + points.opponent_error;
	return { ...t, points };
};
// 以對手失誤得分佔比控制名次；其餘三項固定 25 / 10 / 15（共 50）
const byError = (id: number, error: number, games = 3) =>
	withPoints(team(id, {}, games), { attack: 25, block: 10, serve: 15, opponent_error: error });
const shares = (id: number, attack: number, block: number, serve: number, opponent_error: number) =>
	withPoints(team(id, {}), { attack, block, serve, opponent_error });
const comparison = (teams: TeamAdvanced[]) => teamComparison(data(teams), 21, 'men', teams[0].team_id)!;

describe('sourcesConclusion', () => {
	it('四項正式名稱', () => {
		expect(SCORE_SOURCES.map((s) => s.label)).toEqual(['攻擊得分', '攔網得分', '發球得分', '對手失誤得分']);
	});

	it('佔比最高的一端：寫組內最高並四捨五入整數', () => {
		// 本隊：25/10/15/error=50 → 對手失誤 50%；對手失誤佔比最高
		const c = comparison([byError(4, 50), byError(1, 10), byError(2, 20)]);
		expect(sourcesConclusion(c)).toBe('對手失誤得分佔 50%，組內最高');
	});

	it('並列最低：其他項不在兩端時取最低端，並列寫並列', () => {
		// 各隊佔比（攻擊／攔網／發球／對手失誤，總分 100）；本隊攻擊 10% 與另一隊並列最低
		const c = comparison([
			shares(4, 10, 30, 30, 30),
			shares(1, 40, 20, 20, 20),
			shares(2, 20, 40, 40, 0),
			shares(3, 10, 10, 10, 70)
		]);
		expect(sourcesConclusion(c)).toBe('攻擊得分佔 10%，組內並列最低');
	});

	it('不在兩端時寫第 k 高，名次相同偏好由高端、再依項目順序', () => {
		const c = comparison([
			shares(4, 20, 20, 20, 40),
			shares(1, 40, 10, 30, 20),
			shares(2, 30, 30, 10, 30),
			shares(3, 5, 35, 10, 50)
		]);
		expect(sourcesConclusion(c)).toBe('發球得分佔 20%，組內第 2 高');
	});

	it('四項都並列最高時取第一項', () => {
		const c = comparison([byError(4, 50), byError(1, 50)]);
		expect(sourcesConclusion(c)).toBe('攻擊得分佔 25%，組內並列最高');
	});

	it('已賽未滿 3 場、可比較的隊少於 2、total 為 0 都不產生', () => {
		expect(sourcesConclusion(comparison([byError(4, 10, 2), byError(1, 20)]))).toBeNull();
		expect(sourcesConclusion(comparison([byError(4, 10)]))).toBeNull();
		const zero = withPoints(team(1, {}), { attack: 0, block: 0, serve: 0, opponent_error: 0 });
		expect(sourcesConclusion(comparison([byError(4, 10), zero]))).toBeNull();
	});
});

type BarOption = {
	yAxis: { data: string[]; inverse: boolean };
	xAxis: { min: number; max: number };
	legend: { data: string[] };
	series: { name: string; stack: string; data: { value: number; itemStyle: { opacity: number } }[] }[];
	tooltip: { formatter: (p: { dataIndex: number }[]) => string };
};

describe('sourcesBarOption', () => {
	const names: Record<number, string> = { 1: '屏東台電', 4: '臺北鳳凰', 2: '雲林美津濃' };
	const zero = withPoints(team(2, {}), { attack: 0, block: 0, serve: 0, opponent_error: 0 });
	const option = () =>
		sourcesBarOption(comparison([byError(4, 50), byError(1, 10), zero]), (id) => names[id]) as BarOption;

	it('本隊在最上面並註明本隊，total 為 0 的隊不畫', () => {
		const o = option();
		expect(o.yAxis.inverse).toBe(true);
		expect(o.yAxis.data).toEqual(['臺北鳳凰（本隊）', '屏東台電']);
	});

	it('四個 series 同一堆疊、佔比到小數 1 位、其他隊淡化', () => {
		const o = option();
		expect(o.series.map((s) => s.name)).toEqual(['攻擊得分', '攔網得分', '發球得分', '對手失誤得分']);
		expect(new Set(o.series.map((s) => s.stack)).size).toBe(1);
		expect(o.series[0].data.map((d) => d.value)).toEqual([25, 41.7]);
		expect(o.series[0].data.map((d) => d.itemStyle.opacity)).toEqual([1, 0.35]);
		expect(o.xAxis).toMatchObject({ min: 0, max: 100 });
		expect(o.legend.data).toHaveLength(4);
	});

	it('tooltip 顯示四項得分數與佔比，並註明推算值', () => {
		const text = option().tooltip.formatter([{ dataIndex: 0 }]);
		expect(text).toContain('臺北鳳凰');
		expect(text).toContain('對手失誤得分：50 分（50%）');
		expect(text.endsWith('對手失誤得分是推算值：比分總分 − 技術得分')).toBe(true);
	});

	it('tooltip 的隊名轉義成文字', () => {
		const o = sourcesBarOption(comparison([byError(4, 50)]), () => '<img src=x onerror=alert(1)>') as BarOption;
		expect(o.tooltip.formatter([{ dataIndex: 0 }]).startsWith('&lt;img src=x onerror=alert(1)&gt;<br/>')).toBe(true);
	});
});

describe('sourcesTable', () => {
	it('本隊在第一列，各列有四項得分數與佔比及比分總分', () => {
		const names: Record<number, string> = { 4: '臺北鳳凰', 1: '屏東台電' };
		const rows = sourcesTable(comparison([byError(1, 10), byError(4, 50)].reverse()), (id) => names[id]);
		expect(rows.map((r) => [r.name, r.isTeam, r.total])).toEqual([['臺北鳳凰', true, 100], ['屏東台電', false, 60]]);
		expect(rows[0].cells[3]).toEqual({ key: 'opponent_error', points: 50, share: '50%' });
		expect(rows[1].cells[3]).toEqual({ key: 'opponent_error', points: 10, share: '17%' });
	});
});

describe('pregameComparison', () => {
	it('依賽季、組別找出主客兩隊與組內各隊', () => {
		const home = team(4, {});
		const away = team(1, {});
		const m = pregameComparison(data([home, away, team(2, {})]), 21, 'men', 4, 1);
		expect(m?.home).toBe(home);
		expect(m?.away).toBe(away);
		expect(m?.teams).toHaveLength(3);
	});

	it('任一隊本季沒有單場數據就沒有對戰比較，也不拿別季代替', () => {
		const d: TeamAdvancedData = {
			team_advanced: [
				{ cup_id: 21, group: 'men', teams: [team(4, {}), team(1, {})] },
				{ cup_id: 22, group: 'men', teams: [team(4, {})] }
			]
		};
		expect(pregameComparison(d, 22, 'men', 4, 1)).toBeUndefined();
		expect(pregameComparison(d, 21, 'women', 4, 1)).toBeUndefined();
		expect(pregameComparison(data([team(4, {}), team(1, {}, 0)]), 21, 'men', 4, 1)).toBeUndefined();
	});
});

describe('pregameConclusion', () => {
	const pair = (home: TeamAdvanced, away: TeamAdvanced) => ({ home, away, teams: [home, away] });
	const better = { attack_success_rate: 1, serve_point_rate: 1, receive_excellent_rate: 1, dig_excellent_rate: 1 };
	const worse = { attack_success_rate: 2, serve_point_rate: 2, receive_excellent_rate: 2, dig_excellent_rate: 2 };

	it('領先項數多的隊伍先講，再講另一隊', () => {
		const home = team(4, { ...better, set_excellent_rate: 2, blocks_per_set: 2 });
		const away = team(1, { ...worse, set_excellent_rate: 1, blocks_per_set: 1 });
		expect(pregameConclusion(pair(home, away), '台電', '鳳凰')).toBe('台電 在 4 項指標領先，鳳凰 在 2 項領先');
		expect(pregameConclusion(pair(away, home), '鳳凰', '台電')).toBe('台電 在 4 項指標領先，鳳凰 在 2 項領先');
	});

	it('一隊全面領先時只講領先的一隊；數值相同與缺值的項目不算', () => {
		const home = team(4, { ...worse, set_excellent_rate: 1, blocks_per_set: null });
		const away = team(1, { ...better, set_excellent_rate: 1, blocks_per_set: 1 });
		expect(pregameConclusion(pair(home, away), '台電', '鳳凰')).toBe('鳳凰 在 4 項指標領先');
	});

	it('兩隊領先項數相同', () => {
		const home = team(4, { attack_success_rate: 1, serve_point_rate: 1, receive_excellent_rate: 1, dig_excellent_rate: 2, set_excellent_rate: 2, blocks_per_set: 2 });
		const away = team(1, { attack_success_rate: 2, serve_point_rate: 2, receive_excellent_rate: 2, dig_excellent_rate: 1, set_excellent_rate: 1, blocks_per_set: 1 });
		expect(pregameConclusion(pair(home, away), '台電', '鳳凰')).toBe('兩隊各在 3 項指標領先');
	});

	it('任一隊已賽未滿 3 場不產生結論', () => {
		expect(pregameConclusion(pair(team(4, better), team(1, worse, 2)), '台電', '鳳凰')).toBeNull();
		expect(pregameConclusion(pair(team(4, better, 2), team(1, worse)), '台電', '鳳凰')).toBeNull();
	});
});

describe('pregameBarOption', () => {
	const home = team(4, { attack_success_rate: 1, blocks_per_set: null });
	const away = team(1, { attack_success_rate: 2 });
	const m = { home, away, teams: [home, away, team(2, { attack_success_rate: 3 })] };
	const option = () =>
		pregameBarOption(m, { home: '台電', away: '鳳凰' }, { home: '#c00000', away: '#0050a0' }) as any;

	it('主隊在左、客隊在右', () => {
		const [h, a] = option().series;
		expect(h).toMatchObject({ name: '台電' });
		expect(a).toMatchObject({ name: '鳳凰' });
		expect(h.data.every((d: { value: number | null }) => d.value === null || d.value <= 0)).toBe(true);
		expect(a.data.every((d: { value: number | null }) => d.value === null || d.value >= 0)).toBe(true);
	});

	it('長條用 teamPalette().chart 圖表隊色，不用原始 hex（#175）', () => {
		const [h, a] = (pregameBarOption(m, { home: '台電', away: '鳳凰' }, { home: '#0B304B', away: '#0A4951' }) as any).series;
		expect(h.itemStyle.color).toBe(teamPalette('#0B304B').chart);
		expect(h.itemStyle.color).not.toBe('#0B304B');
		expect(a.itemStyle.color).toBe(teamPalette('#0A4951').chart);
		expect(a.itemStyle.color).not.toBe('#0A4951');
	});

	it('長條以組內最高值為滿格；缺值不畫', () => {
		const [h, a] = option().series;
		expect(h.data[0].value).toBe(-100);
		expect(a.data[0].value).toBe(80);
		expect(h.data[5].value).toBeNull();
	});

	it('tooltip 顯示兩隊數值', () => {
		const text = option().tooltip.formatter([{ dataIndex: 0 }]);
		expect(text).toContain('攻擊成功率');
		expect(text).toContain('台電：50%');
		expect(text).toContain('鳳凰：40%');
		expect(option().tooltip.formatter([{ dataIndex: 5 }])).toContain('台電：—');
	});

	it('tooltip 的隊名轉義成文字', () => {
		const o = pregameBarOption(m, { home: '<img src=x onerror=alert(1)>', away: '鳳凰' }, { home: '#c00000', away: '#0050a0' }) as any;
		expect(o.tooltip.formatter([{ dataIndex: 0 }])).toBe('攻擊成功率<br/>&lt;img src=x onerror=alert(1)&gt;：50%<br/>鳳凰：40%');
	});
});

describe('pregameTable', () => {
	it('每項指標一列，並列兩隊數值與組內名次', () => {
		const home = team(4, { attack_success_rate: 1, blocks_per_set: null });
		const away = team(1, { attack_success_rate: 2 });
		const rows = pregameTable({ home, away, teams: [home, away, team(2, { attack_success_rate: 3 })] });
		expect(rows).toHaveLength(6);
		expect(rows[0]).toEqual({
			key: 'attack_success_rate',
			label: '攻擊成功率',
			home: { value: '50%', rank: '第 1 / 3' },
			away: { value: '40%', rank: '第 2 / 3' }
		});
		expect(rows[5].home).toEqual({ value: '—', rank: '—' });
	});
});

describe('球隊逐場趨勢', () => {
	const rate = (numerator: number, denominator: number) => ({
		numerator,
		denominator,
		value: denominator === 0 ? null : Math.round((numerator / denominator) * 10000) / 100
	});
	const game = (game_id: number, date: string, opponent_id: number | null, attack: [number, number], blocks: [number, number] = [2, 3]): TeamTrendGame => {
		const zero = rate(0, 0);
		const metrics: TeamGameMetrics = {
			attack_success_rate: rate(...attack),
			serve_point_rate: zero,
			receive_excellent_rate: zero,
			dig_excellent_rate: zero,
			set_excellent_rate: zero,
			blocks_per_set: { numerator: blocks[0], denominator: blocks[1], value: Math.round((blocks[0] / blocks[1]) * 100) / 100 }
		};
		return { game_id, date, opponent_id, metrics };
	};
	const withTrend = (trend: TeamTrendGame[]): TeamAdvanced => {
		const sum = (key: keyof TeamGameMetrics, part: 'numerator' | 'denominator') =>
			trend.reduce((n, g) => n + g.metrics[key][part], 0);
		const total = (key: keyof TeamGameMetrics, scale: number) => {
			const den = sum(key, 'denominator');
			return { numerator: sum(key, 'numerator'), denominator: den, value: den ? Math.round((sum(key, 'numerator') / den) * scale * 100) / 100 : null, rank: 1 };
		};
		return {
			...team(4, {}, trend.length),
			metrics: {
				attack_success_rate: total('attack_success_rate', 100),
				serve_point_rate: total('serve_point_rate', 100),
				receive_excellent_rate: total('receive_excellent_rate', 100),
				dig_excellent_rate: total('dig_excellent_rate', 100),
				set_excellent_rate: total('set_excellent_rate', 100),
				blocks_per_set: total('blocks_per_set', 1)
			},
			trend
		};
	};
	const names: Record<number, string> = { 1: '台電', 2: '美津濃' };
	const teamName = (id: number) => names[id];
	// 前 2 場 攻擊 10/40 = 25%；近 3 場 30/60 = 50%；整季 40/100 = 40% → 高 10 個百分點
	const season = withTrend([
		game(1, '2026-01-03', 1, [5, 20]),
		game(2, '2026-01-10', 2, [5, 20]),
		game(3, '2026-01-17', 1, [10, 20]),
		game(4, '2026-01-24', 2, [10, 20]),
		game(5, '2026-01-31', null, [10, 20], [4, 4])
	]);

	type Fmt = (p: { dataIndex: number }[]) => string;
	type O = {
		xAxis: { data: string[] };
		yAxis: { min?: number; max?: number; name: string };
		series: { name: string; connectNulls: boolean; data: unknown[] }[];
		tooltip: { formatter: Fmt };
	};

	it('折線沿用球員逐場趨勢的做法：MM/DD、比率 y 軸 0–100、分母為 0 是空點', () => {
		const o = teamTrendOption(season, 'attack_success_rate', teamName) as O;
		expect(o.xAxis.data).toEqual(['01/03', '01/10', '01/17', '01/24', '01/31']);
		expect(o.yAxis).toMatchObject({ min: 0, max: 100, name: '%' });
		expect(o.series.map((s) => s.name)).toEqual(['攻擊成功率']);
		expect(o.series[0].connectNulls).toBe(false);
		expect(o.series[0].data).toEqual([25, 25, 50, 50, 50]);
		expect((teamTrendOption(season, 'serve_point_rate', teamName) as O).series[0].data).toEqual([null, null, null, null, null]);
	});

	it('提示框列出日期、對手與分子分母；對手不明時只列日期', () => {
		const o = teamTrendOption(season, 'attack_success_rate', teamName) as O;
		expect(o.tooltip.formatter([{ dataIndex: 1 }])).toBe('2026-01-10 對 美津濃<br/>攻擊成功率：25%（得分 5／20）');
		expect(o.tooltip.formatter([{ dataIndex: 4 }])).toBe('2026-01-31<br/>攻擊成功率：50%（得分 10／20）');
		const blocks = teamTrendOption(season, 'blocks_per_set', teamName) as O;
		expect(blocks.yAxis.max).toBeUndefined();
		expect(blocks.tooltip.formatter([{ dataIndex: 4 }])).toContain('每局攔網：1（攔網 4／4 局）');
	});

	it('提示框的對手隊名轉義成文字', () => {
		const o = teamTrendOption(season, 'attack_success_rate', () => '<img src=x onerror=alert(1)>') as O;
		expect(o.tooltip.formatter([{ dataIndex: 1 }])).toBe('2026-01-10 對 &lt;img src=x onerror=alert(1)&gt;<br/>攻擊成功率：25%（得分 5／20）');
	});

	it('結論是近 3 場加總後再除，與本季整體的差距', () => {
		expect(trendConclusion(season, 'attack_success_rate')).toBe('近 3 場攻擊成功率比本季整體高 10 個百分點');
		// 每局攔網：近 3 場 8/10 = 0.8；整季 12/16 = 0.75
		expect(trendConclusion(season, 'blocks_per_set')).toBe('近 3 場每局攔網比本季整體多 0.05 次');
	});

	it('比本季低時說低，差距四捨五入為 0 時說持平', () => {
		const down = withTrend([game(1, '2026-01-03', 1, [9, 10]), ...[2, 3, 4].map((i) => game(i, `2026-01-1${i}`, 1, [5, 10]))]);
		expect(trendConclusion(down, 'attack_success_rate')).toBe('近 3 場攻擊成功率比本季整體低 10 個百分點');
		const flat = withTrend([1, 2, 3, 4].map((i) => game(i, `2026-01-1${i}`, 1, [5, 10])));
		expect(trendConclusion(flat, 'attack_success_rate')).toBe('近 3 場攻擊成功率與本季整體持平');
	});

	it('已賽 3 場以下（近 3 場即整季）、或近 3 場分母為 0 時不產生結論', () => {
		const two = withTrend([game(1, '2026-01-03', 1, [5, 10]), game(2, '2026-01-10', 2, [6, 10])]);
		expect(trendConclusion(two, 'attack_success_rate')).toBeNull();
		const three = withTrend([game(1, '2026-01-03', 1, [5, 10]), game(2, '2026-01-10', 2, [6, 10]), game(3, '2026-01-17', 1, [9, 10])]);
		expect(trendConclusion(three, 'attack_success_rate')).toBeNull();
		expect(trendConclusion(season, 'serve_point_rate')).toBeNull();
	});

	it('表格每場一列，列出日期、對手與六項指標', () => {
		const rows = trendTable(season, teamName);
		expect(rows).toHaveLength(5);
		expect(rows[0]).toMatchObject({ game_id: 1, date: '2026-01-03', opponent: '台電' });
		expect(rows[4].opponent).toBe('—');
		expect(rows[0].cells.map((c) => c.value)).toEqual(['25%', '—', '—', '—', '—', '0.67']);
	});
});
