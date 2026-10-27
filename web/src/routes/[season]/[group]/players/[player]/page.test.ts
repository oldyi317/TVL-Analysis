import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import PlayerPage from './+page.svelte';
import { findPlayerSeason, players, seasons } from '#lib/site.js';
import type { PageProps } from './$types';

const player = players.find((p) => p.seasons.length > 0)!;
const stats = player.seasons[0];
const found = findPlayerSeason(stats.cup_id, player.group, player.player_id)!;
const season = seasons.find((s) => s.cup_id === stats.cup_id)!;

function page(games: number, setsPlayed: number | null) {
	const props: PageProps = {
		params: { season: season.name, group: player.group, player: String(player.player_id) },
		form: null,
		data: {
			season, group: player.group, player, team: found.team,
			stats: { ...stats, position: 'MB', games, sets_played: setsPlayed },
			advanced: null, distribution: null, gameLog: []
		}
	};
	return render(PlayerPage, { props }).body;
}

describe('球員頁缺少上場局數', () => {
	it('顯示破折號與來源說明，移除每局攔網，兩場不標樣本少', () => {
		const html = page(2, null);
		expect(html).toMatch(/出賽局數<\/dt>\s*<dd[^>]*>—<\/dd>/);
		expect(html).toContain('官網 box score 不提供上場局數');
		expect(html).not.toContain('每局攔網');
		expect(html).not.toContain('樣本少');
		expect(page(1, null)).toContain('樣本少');
	});

	it('整季沒有上場紀錄時顯示明確文案', () => {
		expect(page(0, 0)).toContain('本季尚無上場紀錄');
	});
});
