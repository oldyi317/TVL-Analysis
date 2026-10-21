// 色值與 src/app.css 的 token 同步；這是唯一允許重複色值的地方（ECharts 不吃 CSS 變數）
const FONT =
	"system-ui, -apple-system, 'Segoe UI', Roboto, 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', sans-serif";

export const CHART_COLORS = {
	light: {
		bg: '#f3f4f9',
		surface: '#ffffff',
		text: '#262b4a',
		muted: '#5f6788',
		line: '#c9cde2',
		series: ['#5258da', '#e2457f', '#cc6a00', '#6b72d4', '#0b9279']
	},
	dark: {
		bg: '#12141f',
		surface: '#1b1e2e',
		text: '#eceef8',
		muted: '#9ba1bc',
		line: '#3a405a',
		series: ['#7b82f0', '#f66fa0', '#f8b36a', '#5f66b8', '#3fd2b6']
	}
} as const;

export function chartTheme(mode: 'light' | 'dark'): object {
	const c = CHART_COLORS[mode];
	const axis = {
		axisLabel: { color: c.muted, fontSize: 12 },
		axisLine: { show: false },
		axisTick: { show: false },
		splitLine: { lineStyle: { color: c.line, type: 'solid' } }
	};
	return {
		color: [...c.series],
		backgroundColor: 'transparent',
		textStyle: { fontFamily: FONT, color: c.text },
		legend: { icon: 'circle', textStyle: { color: c.muted, fontSize: 14, fontFamily: FONT } },
		tooltip: {
			backgroundColor: c.surface,
			borderWidth: 0,
			borderRadius: 12,
			extraCssText: 'box-shadow: 0 12px 32px rgba(38, 43, 74, 0.14); border-radius: 12px;',
			textStyle: { color: c.text, fontSize: 16, fontFamily: FONT }
		},
		bar: { barMaxWidth: 18, itemStyle: { borderRadius: [6, 6, 0, 0] } },
		line: { smooth: 0.3, symbol: 'circle', symbolSize: 6, lineStyle: { width: 2.5 } },
		categoryAxis: { ...axis, splitLine: { show: false }, axisLabel: { ...axis.axisLabel, interval: 0 } },
		valueAxis: axis
	};
}
