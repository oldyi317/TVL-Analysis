import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import PlayerPage from './+page.svelte';
import { findPlayerSeason, GROUP_LABEL, playerDisplayName, players, seasons } from '#lib/site.js';
import type { PageProps } from './$types';

const player = players.find((p) => p.seasons.length > 0)!;
const stats = player.seasons[0];
const found = findPlayerSeason(stats.cup_id, player.group, player.player_id)!;
const season = seasons.find((s) => s.cup_id === stats.cup_id)!;

// SiteHead 從 $app/state 讀 page.url，SSR 下靠 __request__ context 供應
const context = new Map([
	['__request__', { page: { url: new URL('http://localhost/TVL-Analysis/22/men/players/1/') } }]
]);

function renderPage(games: number, setsPlayed: number | null) {
	const props: PageProps = {
		params: { season: season.name, group: player.group, player: String(player.player_id) },
		form: null,
		data: {
			season, group: player.group, player, team: found.team,
			stats: { ...stats, position: 'MB', games, sets_played: setsPlayed },
			advanced: null, distribution: null, gameLog: []
		}
	};
	return render(PlayerPage, { props, context });
}

const page = (games: number, setsPlayed: number | null) => renderPage(games, setsPlayed).body;

describe('球員頁 head', () => {
	it('title 維持原格式，並帶 Open Graph 與絕對網址', () => {
		const head = renderPage(2, 6).head;
		const title = `${playerDisplayName(player)}｜${season.name} ${GROUP_LABEL[player.group]} 球員｜TVL 數據`;
		expect(head).toContain(`<title>${title}</title>`);
		expect(head).toContain(`<meta property="og:title" content="${title}"`);
		expect(head).toContain(
			'<meta property="og:url" content="https://oldyi317.github.io/TVL-Analysis/22/men/players/1/"'
		);
	});
});

describe('球員頁缺少出賽局數', () => {
	it('顯示破折號與來源說明，移除每局攔網，兩場不標樣本少', () => {
		const html = page(2, null);
		expect(html).toMatch(/出賽局數<\/dt>\s*<dd[^>]*>—<\/dd>/);
		expect(html).toContain('官網單場數據不提供出賽局數');
		expect(html).not.toContain('每局攔網');
		expect(html).not.toContain('樣本少');
		expect(page(1, null)).toContain('樣本少');
	});

	it('整季沒有上場紀錄時顯示明確文案', () => {
		expect(page(0, 0)).toContain('本季尚無上場紀錄');
	});
});
