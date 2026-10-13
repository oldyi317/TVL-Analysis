import { describe, expect, it } from 'vitest';
import { contrastRatio, teamPalette } from '#lib/color.js';
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
});
