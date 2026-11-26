import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { CHART_COLORS } from './chart/theme.js';
import { contrastRatio, SURFACE } from './color.js';

// 直接解析 app.css，確保寫在 CSS 裡的 token 本身達標，而不是另一份副本
const css = readFileSync(new URL('../app.css', import.meta.url), 'utf8');

function block(selector: RegExp): Record<string, string> {
	const match = css.match(selector);
	if (!match) throw new Error(`找不到區塊 ${selector}`);
	const vars: Record<string, string> = {};
	for (const [, name, value] of match[1].matchAll(/--([\w-]+):\s*(#[0-9a-fA-F]{6})\s*;/g)) {
		vars[name] = value;
	}
	return vars;
}

function oklab(hex: string): [number, number, number] {
	const [r, g, b] = [1, 3, 5].map((i) => {
		const c = parseInt(hex.slice(i, i + 2), 16) / 255;
		return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
	});
	const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
	const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
	const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
	return [
		0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
		1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
		0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
	];
}

function deltaE(a: string, b: string): number {
	const [x, y] = [oklab(a), oklab(b)];
	return 100 * Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]);
}

const light = block(/:root\s*\{([^}]*)\}/);
const dark = block(/prefers-color-scheme:\s*dark\)\s*\{\s*:root\s*\{([^}]*)\}/);

// 文字色 × 背景色：全部必須達 WCAG AA 4.5:1
const textPairs: [string, string][] = [
	['text', 'bg'],
	['text', 'surface'],
	['text-muted', 'bg'],
	['text-muted', 'surface'],
	['win', 'surface'],
	['win', 'bg'],
	['loss', 'surface'],
	['loss', 'bg'],
	['warn', 'surface'],
	['warn', 'bg'],
	['on-accent', 'accent'],
	['warn-ink', 'warn-tint'],
	['focus', 'bg'],
	['focus', 'surface'],
];

describe.each([
	['淺色', light],
	['深色', dark],
])('%s 模式 token', (_, vars) => {
	it('兩種模式定義同一組顏色 token', () => {
		expect(Object.keys(vars).sort()).toEqual(Object.keys(light).sort());
	});
	it.each(textPairs)('--%s 放在 --%s 上達 4.5:1', (fg, bg) => {
		expect(vars[fg], fg).toBeDefined();
		expect(vars[bg], bg).toBeDefined();
		expect(contrastRatio(vars[fg], vars[bg])).toBeGreaterThanOrEqual(4.5);
	});
	it('分隔線對背景至少 1.5:1（看得到但不搶眼）', () => {
		expect(contrastRatio(vars.line, vars.surface)).toBeGreaterThanOrEqual(1.5);
	});
});

// JS 端必須重複色值的兩處（隊色計算、ECharts 主題）要跟 app.css 同步
describe('JS 端色值與 app.css 同步', () => {
	it('color.ts 的 SURFACE', () => {
		expect(SURFACE.light).toBe(light.surface);
		expect(SURFACE.dark).toBe(dark.surface);
	});
	it.each([
		['light', light],
		['dark', dark],
	] as const)('chart/theme.ts 的 %s 模式', (mode, vars) => {
		const c = CHART_COLORS[mode];
		expect([c.bg, c.surface, c.text, c.muted, c.line]).toEqual([
			vars.bg,
			vars.surface,
			vars.text,
			vars['text-muted'],
			vars.line,
		]);
		expect([...c.series]).toEqual(c.series.map((_, i) => vars[`series-${i + 1}`]));
	});
	// 目前圖表最多同時畫四條序列（得分來源組成）；ΔE 15 是一般視覺分得開的下限
	it.each(['light', 'dark'] as const)('%s 模式的前四條圖表序列色兩兩分得開（OKLab ΔE ≥ 15）', (mode) => {
		const series = CHART_COLORS[mode].series.slice(0, 4);
		for (const [i, a] of series.entries()) {
			for (const b of series.slice(i + 1)) {
				expect(deltaE(a, b), `${a} ↔ ${b}`).toBeGreaterThanOrEqual(15);
			}
		}
	});
	it.each([
		['light', light],
		['dark', dark],
	] as const)('%s 模式的圖表中性序列色對背景至少 3:1（非文字對比）', (mode, vars) => {
		for (const color of CHART_COLORS[mode].series) {
			expect(contrastRatio(color, vars.bg), color).toBeGreaterThanOrEqual(3);
		}
	});
});

it('負向對照：檢查器會把不達標的組合判紅', () => {
	expect(contrastRatio('#999999', '#ffffff')).toBeLessThan(4.5);
});

const darkBlock = css.match(/prefers-color-scheme:\s*dark\)\s*\{\s*:root\s*\{([^}]*)\}/)![1];
const rootBlock = css.match(/:root\s*\{([^}]*)\}/)![1];

describe('殼層 token', () => {
	it.each(['top', 'right', 'bottom', 'left'])('--safe-%s 取安全區 inset 並以 0px 為後備', (side) => {
		expect(rootBlock).toContain(`--safe-${side}: env(safe-area-inset-${side}, 0px);`);
	});
	it('--tabbar-h 已定義', () => {
		expect(rootBlock).toMatch(/--tabbar-h:\s*[\d.]+rem;/);
	});
	it('--z-tabbar 高於既有 sticky 元件的 z-index: 1', () => {
		expect(Number(rootBlock.match(/--z-tabbar:\s*(\d+);/)?.[1])).toBeGreaterThan(1);
	});
	it('--scrim 深淺兩邊各自定義為帶透明度的色', () => {
		expect(rootBlock).toMatch(/--scrim:\s*rgba\([^)]*\);/);
		expect(darkBlock).toMatch(/--scrim:\s*rgba\([^)]*\);/);
	});
});

describe('頂部列 token', () => {
	it('--topbar-h 已定義，且不小於觸控高 --tap（2.75rem）', () => {
		expect(Number(rootBlock.match(/--topbar-h:\s*([\d.]+)rem;/)?.[1])).toBeGreaterThanOrEqual(2.75);
	});
	it('--z-topbar 高於頁內 sticky 元件的 z-index: 1', () => {
		expect(Number(rootBlock.match(/--z-topbar:\s*(\d+);/)?.[1])).toBeGreaterThan(1);
	});
	it('--shell-top 手機是頂部列加上方安全區', () => {
		expect(rootBlock).toContain('--shell-top: calc(var(--topbar-h) + var(--safe-top));');
	});
	it('--shell-top 在 64rem 以上歸零，桌面的 sticky 元件位置不變', () => {
		expect(css).toMatch(/@media \(min-width: 64rem\)\s*\{[^@]*:root\s*\{\s*--shell-top:\s*0px;/);
	});
});
