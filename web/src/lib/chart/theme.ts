// 色值與 src/app.css 的 token 同步；這是唯一允許重複色值的地方（ECharts 不吃 CSS 變數）
const FONT =
	"system-ui, -apple-system, 'Segoe UI', Roboto, 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', sans-serif";

export const CHART_COLORS = {
	light: {
		bg: '#f2f3f5',
		surface: '#ffffff',
		text: '#15171a',
		muted: '#5b6168',
		line: '#c8ccd2',
		series: ['#15171a', '#5b6168', '#858c95', '#3d4248']
	},
	dark: {
		bg: '#121417',
		surface: '#1c1f23',
		text: '#eceef0',
		muted: '#a3a9b1',
		line: '#3a4048',
		series: ['#eceef0', '#a3a9b1', '#7c838c', '#d2d6da']
	}
} as const;

export function chartTheme(mode: 'light' | 'dark'): object {
	const c = CHART_COLORS[mode];
	const axis = {
		axisLabel: { color: c.muted, fontSize: 14 },
		axisLine: { lineStyle: { color: c.line } },
		axisTick: { lineStyle: { color: c.line } },
		splitLine: { lineStyle: { color: c.line } }
	};
	return {
		color: [...c.series],
		backgroundColor: 'transparent',
		textStyle: { fontFamily: FONT, color: c.text },
		legend: { textStyle: { color: c.muted, fontSize: 14, fontFamily: FONT } },
		tooltip: {
			backgroundColor: c.surface,
			borderColor: c.line,
			textStyle: { color: c.text, fontSize: 16, fontFamily: FONT }
		},
		categoryAxis: { ...axis, axisLabel: { ...axis.axisLabel, interval: 0 } },
		valueAxis: axis
	};
}
