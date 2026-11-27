// narrow 只寫容器寬度 ≤ 480px 的差異，陣列整個替換。寬圖同一路徑缺值會丟錯，grid.left／right 可用預設。
// 橫軸有 name 會多留軸名高度，nameGap 由各圖指定。
// axis 的 lines 收到第一筆的 dataIndex，item 的 lines 收到該點的 name、value、data。
// chartOption 是純函式，可依任意順序呼叫，不匯入 echarts。
import { bottomLegendMedia } from './legend.js';
import { tooltipOption } from './tooltip.js';

type Obj = Record<string, unknown>;

export const NARROW_MAX = 480;
export type Line = string | { bold: string };

export type ChartSpec<D = unknown> = {
	series: Obj[];
	xAxis: Obj;
	yAxis: Obj;
	legend?: readonly string[];
	grid?: { top?: number; left?: number; right?: number };
	narrow?: { grid?: { left?: number; right?: number }; xAxis?: Obj; yAxis?: Obj };
	tooltip:
		| { trigger: 'item'; lines: (hit: { name: string; value: unknown; data: D }) => Line[] }
		| { trigger: 'axis'; axisPointer?: { type: 'shadow' }; lines: (dataIndex: number) => Line[] };
};

declare const sealed: unique symbol;
export type ChartOption = { readonly [sealed]: true; readonly option: Obj };

const LEGEND_GAP = 16;
const NAMED_AXIS_LEGEND_GAP = 40;
const EDGE_GAP = 8;
const NAMED_AXIS_EDGE_GAP = 28;

function isPlainObject(value: unknown): value is Obj {
	if (value === null || typeof value !== 'object') return false;
	const proto = Object.getPrototypeOf(value);
	return proto === Object.prototype || proto === null;
}

// 寬圖需逐一還原窄圖改過的值，避免容器放大後仍沿用窄圖設定。
function wideValues(narrow: Obj, main: unknown, parent = ''): Obj {
	return Object.fromEntries(Object.entries(narrow).map(([key, value]) => {
		const path = parent ? parent + '.' + key : key;
		const wide = isPlainObject(main) && Object.hasOwn(main, key) ? main[key] : undefined;
		if (isPlainObject(value)) return [key, wideValues(value, wide, path)];
		if (wide === undefined) throw new Error('主設定缺少 ' + path);
		return [key, wide];
	}));
}

export function chartOption<D>(spec: ChartSpec<D>): ChartOption {
	const tiers = spec.legend?.length
		? bottomLegendMedia(spec.legend, spec.xAxis.name ? NAMED_AXIS_LEGEND_GAP : LEGEND_GAP)
		: [];
	const bottoms = new Map(tiers.map(({ query, option }) => [query?.minWidth ?? 0, option.grid.bottom]));
	if (!bottoms.size) bottoms.set(0, spec.xAxis.name ? NAMED_AXIS_EDGE_GAP : EDGE_GAP);
	const cuts = new Set(bottoms.keys());
	if (spec.narrow) cuts.add(NARROW_MAX + 1);
	const starts = [...cuts].sort((a, b) => a - b);
	const bottom = bottoms.get(Math.max(...bottoms.keys()));
	const option: Obj = {
		tooltip: tooltipOption(spec.tooltip),
		xAxis: spec.xAxis, yAxis: spec.yAxis, series: spec.series,
		grid: { top: 24, right: 16, bottom, left: 8, containLabel: true, ...spec.grid },
		legend: spec.legend ? { data: [...spec.legend], bottom: 0 } : { show: false, bottom: 0 }
	};
	const narrow = spec.narrow;
	const wide = narrow && wideValues(narrow, option);
	if (starts.length > 1) {
		let currentBottom = bottoms.get(0);
		option.media = starts.map((start, i) => {
			currentBottom = bottoms.get(start) ?? currentBottom;
			const changes = start <= NARROW_MAX ? narrow : wide;
			return {
				query: {
					...(i > 0 && { minWidth: start }),
					...(i + 1 < starts.length && { maxWidth: starts[i + 1] - 1 })
				},
				option: { ...changes, grid: { ...(changes?.grid as Obj), bottom: currentBottom } }
			};
		});
	}
	return { option } as ChartOption;
}
