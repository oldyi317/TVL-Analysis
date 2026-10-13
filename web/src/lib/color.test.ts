import { describe, expect, it } from 'vitest';
import { contrastRatio, inkOn, textColorOn } from './color.js';
import { teams } from './site.js';

describe('contrastRatio', () => {
	it('黑白對比為 21', () => {
		expect(contrastRatio('#000000', '#ffffff')).toBeCloseTo(21, 1);
	});
	it('同色對比為 1', () => {
		expect(contrastRatio('#377eb8', '#377eb8')).toBeCloseTo(1, 5);
	});
	it('#767676 對白色剛好達 AA（WCAG 範例值 4.54）', () => {
		expect(contrastRatio('#767676', '#ffffff')).toBeCloseTo(4.54, 1);
	});
});

describe('textColorOn', () => {
	it('每一隊隊標上的文字對隊色達 4.5:1', () => {
		for (const team of teams) {
			expect(contrastRatio(textColorOn(team.color), team.color), team.short_name).toBeGreaterThanOrEqual(4.5);
		}
	});
});

describe('inkOn', () => {
	const lightSurface = '#ffffff';
	const darkSurface = '#1c1f23';

	it('淺色模式：每一隊的文字色對白色表面達 4.5:1', () => {
		for (const team of teams) {
			expect(contrastRatio(inkOn(team.color, lightSurface), lightSurface), team.short_name).toBeGreaterThanOrEqual(4.5);
		}
	});
	it('深色模式：每一隊的文字色對深色表面達 4.5:1', () => {
		for (const team of teams) {
			expect(contrastRatio(inkOn(team.color, darkSurface), darkSurface), team.short_name).toBeGreaterThanOrEqual(4.5);
		}
	});
	it('已達標的顏色原樣回傳', () => {
		expect(inkOn('#1b9e77', lightSurface)).not.toBe('#1b9e77');
		expect(inkOn('#0000ff', lightSurface)).toBe('#0000ff');
	});
	it('保留色相：調整後的藍色仍是藍色', () => {
		const ink = inkOn('#377eb8', lightSurface);
		const [r, g, b] = [1, 3, 5].map((i) => parseInt(ink.slice(i, i + 2), 16));
		expect(b).toBeGreaterThan(r);
		expect(b).toBeGreaterThan(g);
	});
});
