import { teamPalette } from '#lib/color.js';
import type { Team } from '#lib/data/types.js';

export function teamStyle(team: Team): string {
	const p = teamPalette(team.color);
	return `--team-fill:${p.fill};--team-on-fill:${p.onFill};--team-ink-light:${p.inkLight};--team-ink-dark:${p.inkDark}`;
}
