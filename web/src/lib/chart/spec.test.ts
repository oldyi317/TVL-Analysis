import { expect, it, vi } from 'vitest';
import { chartOption, type ChartSpec } from './spec.js';

const base: ChartSpec = {
	xAxis: {}, yAxis: {}, series: [],
	tooltip: { trigger: 'axis', lines: () => [] }
};

it('沒有圖例與橫軸名時採用預設留白，所有寬度共用主設定', () => {
	const { option } = chartOption(base);
	expect(option.grid).toEqual({ top: 24, right: 16, bottom: 8, left: 8, containLabel: true });
	expect(option.legend).toEqual({ show: false, bottom: 0 });
	expect(option).not.toHaveProperty('media');
	expect(option.xAxis).toBe(base.xAxis);
	expect(option.yAxis).toBe(base.yAxis);
	expect(option.series).toBe(base.series);
});

it('橫軸有軸名、沒有圖例時底部留 28', () => {
	expect(chartOption({ ...base, xAxis: { name: '得分' } }).option.grid).toMatchObject({ bottom: 28 });
});

it('只有一項圖例時各寬度都留一行，有軸名留 64、無軸名留 40', () => {
	const legend = ['屏東台電'];
	for (const [xAxis, bottom] of [[{}, 40], [{ name: '得分' }, 64]] as const) {
		const { option } = chartOption({ ...base, xAxis, legend });
		expect(option.grid).toMatchObject({ bottom });
		expect(option.legend).toEqual({ data: legend, bottom: 0 });
		expect(option).not.toHaveProperty('media');
	}
});

type Obj = Record<string, unknown>;
type Tier = { query: { minWidth?: number; maxWidth?: number }; option: Obj };

function overlay(main: Obj, changes: Obj): Obj {
	const result = { ...main };
	for (const [key, value] of Object.entries(changes)) {
		result[key] = value && typeof value === 'object' && !Array.isArray(value)
			? overlay(main[key] as Obj, value as Obj)
			: value;
	}
	return result;
}

function atWidth(option: Obj, width: number): Obj {
	if (!option.media) return option;
	const matches = (option.media as Tier[]).filter(({ query }) =>
		width >= (query.minWidth ?? 0) && width <= (query.maxWidth ?? Infinity)
	);
	expect(matches).toHaveLength(1);
	return overlay(option, matches[0].option);
}

const men = ['屏東台電', '雲林美津濃', '臺北國北獅', '桃園臺產', '獅子王'];

it('五隊圖例依寬度折成五至一行，各段依軸名留白', () => {
	const named = chartOption({ ...base, legend: men, xAxis: { name: '得分' } }).option;
	const unnamed = chartOption({ ...base, legend: men }).option;
	for (const [width, withName, withoutName] of [
		[0, 152, 128], [167, 152, 128], [168, 130, 106], [195, 130, 106],
		[196, 108, 84], [271, 108, 84], [272, 86, 62], [465, 86, 62], [466, 64, 40]
	]) {
		expect(atWidth(named, width).grid).toMatchObject({ bottom: withName });
		expect(atWidth(unnamed, width).grid).toMatchObject({ bottom: withoutName });
	}
});

it('圖寬 254 與 309 時依四隊或五隊的隊名留白，英數字以全形字計', () => {
	const women = ['臺北鯨華', '高雄台電', '新北中纖', '義力營造'];
	const bottom = (legend: string[], width: number) =>
		(atWidth(chartOption({ ...base, legend }).option, width).grid as Obj).bottom;
	expect([bottom(men, 254), bottom(women, 254)]).toEqual([84, 62]);
	expect([bottom(men, 309), bottom(women, 309)]).toEqual([62, 62]);
	expect([bottom(['臺北Conti', '連莊'], 195), bottom(['臺北Conti', '連莊'], 196)]).toEqual([62, 40]);
});

it('可指定繪圖區的上、左、右留白', () => {
	const grid = { top: 4, left: 12, right: 20 };
	expect(chartOption({ ...base, grid }).option.grid).toEqual({ ...grid, bottom: 8, containLabel: true });
});

it('480 用窄圖值，481 恢復主設定與預設留白，陣列與函式整個替換', () => {
	const wideLabel = () => '寬';
	const narrowLabel = () => '窄';
	const { option } = chartOption({
		...base,
		xAxis: { interval: 20 },
		yAxis: { axisLabel: { padding: [1, 2], formatter: wideLabel, margin: 8 } },
		narrow: {
			grid: { left: 2, right: 3 }, xAxis: { interval: 50 },
			yAxis: { axisLabel: { padding: [3], formatter: narrowLabel } }
		}
	});
	expect(atWidth(option, 480)).toMatchObject({
		grid: { left: 2, right: 3, bottom: 8 }, xAxis: { interval: 50 },
		yAxis: { axisLabel: { padding: [3], formatter: narrowLabel, margin: 8 } }
	});
	expect(atWidth(option, 481)).toMatchObject({
		grid: { left: 8, right: 16, bottom: 8 }, xAxis: { interval: 20 },
		yAxis: { axisLabel: { padding: [1, 2], formatter: wideLabel, margin: 8 } }
	});
	expect((option.media as Tier[])[1].option).toEqual({
		grid: { left: 8, right: 16, bottom: 8 }, xAxis: { interval: 20 },
		yAxis: { axisLabel: { padding: [1, 2], formatter: wideLabel } }
	});
});

it('五隊圖例與窄圖同時存在時每段都帶齊底部留白與軸間隔', () => {
	const { option } = chartOption({
		...base, legend: men, xAxis: { name: '得分', interval: 20 }, narrow: { xAxis: { interval: 50 } }
	});
	for (const [width, bottom, interval] of [
		[0, 152, 50], [168, 130, 50], [196, 108, 50], [272, 86, 50],
		[465, 86, 50], [466, 64, 50], [480, 64, 50], [481, 64, 20]
	]) {
		expect(atWidth(option, width)).toMatchObject({ grid: { bottom }, xAxis: { interval } });
	}
});

it('窄圖路徑缺少寬圖值時丟錯並指出路徑', () => {
	expect(() => chartOption({ ...base, narrow: { xAxis: { interval: 50 } } })).toThrow('xAxis.interval');
});

it('提示文字每行跳脫五個特殊字元，粗體行加粗並以換行接起', () => {
	const { option } = chartOption({
		...base, tooltip: { trigger: 'axis', lines: () => [`&<>"'`, { bold: `&<>"'` }] }
	});
	const tip = option.tooltip as { formatter: (params: { dataIndex: number }[]) => string };
	expect(tip.formatter([{ dataIndex: 0 }])).toBe('&amp;&lt;&gt;&quot;&#39;<br/><b>&amp;&lt;&gt;&quot;&#39;</b>');
});

it('座標軸提示取第一筆資料索引，帶出指標設定並限制在圖內', () => {
	const lines = vi.fn(() => ['座標軸']);
	const axisPointer = { type: 'shadow' } as const;
	const { option } = chartOption({ ...base, tooltip: { trigger: 'axis', axisPointer, lines } });
	expect(option.tooltip).toMatchObject({ trigger: 'axis', confine: true, axisPointer });
	const tip = option.tooltip as { formatter: (params: { dataIndex: number }[]) => string };
	expect(tip.formatter([{ dataIndex: 3 }, { dataIndex: 7 }])).toBe('座標軸');
	expect(lines).toHaveBeenCalledWith(3);
	expect(chartOption(base).option.tooltip).not.toHaveProperty('axisPointer');
});

it('資料點提示只傳名稱、值與原始資料，並限制在圖內', () => {
	const lines = vi.fn(() => ['資料點']);
	const { option } = chartOption({ ...base, tooltip: { trigger: 'item', lines } });
	const hit = { name: '主隊', value: 2, data: { team: '主隊' } };
	const tip = option.tooltip as { formatter: (params: Obj) => string };
	expect(tip).toMatchObject({ trigger: 'item', confine: true });
	expect(tip).not.toHaveProperty('axisPointer');
	expect(tip.formatter({ ...hit, dataIndex: 3 })).toBe('資料點');
	expect(lines).toHaveBeenCalledWith(hit);
});
