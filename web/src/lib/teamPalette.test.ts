import { describe, expect, it } from 'vitest';
import { contrastRatio, SURFACE, teamPalette } from '#lib/color.js';
import { teams } from '#lib/site.js';

describe('teamPalette', () => {
	it('十二隊的文字色在淺深兩種 surface 上皆達 4.5:1', () => {
		for (const t of teams) {
			const p = teamPalette(t.color);
			expect(p.fill).toBe(t.color);
			expect(contrastRatio(p.inkLight, '#ffffff')).toBeGreaterThanOrEqual(4.5);
			expect(contrastRatio(p.inkDark, '#1c1f23')).toBeGreaterThanOrEqual(4.5);
		}
	});

	it('每一隊的圖表色同一個值在淺深兩種 surface 上皆達 3:1', () => {
		for (const t of teams) {
			const { chart } = teamPalette(t.color);
			expect(contrastRatio(chart, SURFACE.light), t.name).toBeGreaterThanOrEqual(3);
			expect(contrastRatio(chart, SURFACE.dark), t.name).toBeGreaterThanOrEqual(3);
		}
	});

	it('圖表色只動明度，原本就達標的隊色原樣保留', () => {
		expect(teamPalette('#E72410').chart).toBe('#E72410');
		expect(teamPalette('#0B304B').chart).not.toBe('#0B304B');
	});
});
