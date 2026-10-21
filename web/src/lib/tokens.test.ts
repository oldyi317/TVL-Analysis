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
