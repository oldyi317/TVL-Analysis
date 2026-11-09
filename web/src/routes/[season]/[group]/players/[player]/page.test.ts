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

const advancedEntry = (over: { below_threshold?: boolean; team_sets?: number; blocks_per_set?: number | null } = {}) =>
	({
		cup_id: stats.cup_id,
		group: player.group,
		player_id: player.player_id,
		position: 'MB',
		team_sets: over.team_sets ?? 36,
		below_threshold: over.below_threshold ?? false,
		ratios: { blocks_per_set: over.blocks_per_set === undefined ? 0.17 : over.blocks_per_set },
		pr: {}
	}) as unknown as PageProps['data']['advanced'];

function renderPage(games: number, advanced: PageProps['data']['advanced'] = null) {
	const props: PageProps = {
		params: { season: season.name, group: player.group, player: String(player.player_id) },
		form: null,
		data: {
			season, group: player.group, player, team: found.team,
			stats: { ...stats, position: 'MB', games, sets_played: null },
			advanced, distribution: null, gameLog: []
		}
	};
	return render(PlayerPage, { props, context });
}

const page = (games: number, advanced: PageProps['data']['advanced'] = null) => renderPage(games, advanced).body;

describe('球員頁 head', () => {
	it('title 維持原格式，並帶 Open Graph 與絕對網址', () => {
		const head = renderPage(2).head;
		const title = `${playerDisplayName(player)}｜${season.name} ${GROUP_LABEL[player.group]} 球員｜TVL 數據`;
		expect(head).toContain(`<title>${title}</title>`);
		expect(head).toContain(`<meta property="og:title" content="${title}"`);
		expect(head).toContain(
			'<meta property="og:url" content="https://oldyi317.github.io/TVL-Analysis/22/men/players/1/"'
		);
	});
});

describe('球員頁不顯示出賽局數', () => {
	it('沒有出賽局數的欄位與來源說明，也不再以它判斷樣本少', () => {
		const html = page(2, advancedEntry());
		expect(html).not.toContain('出賽局數');
		expect(html).not.toContain('樣本少');
	});

	it('每局攔網大標以球隊已打局數為分母', () => {
		const html = page(2, advancedEntry({ team_sets: 36, blocks_per_set: 0.17 }));
		expect(html).toContain('0.17');
		expect(html).toContain('球隊 36 局');
	});

	it('球隊已打局數為 0 時沒有每局攔網', () => {
		expect(page(2, advancedEntry({ team_sets: 0, blocks_per_set: null }))).not.toContain('每局攔網');
	});

	it('技術次數全部未達浮動門檻才標樣本少', () => {
		expect(page(2, advancedEntry({ below_threshold: true }))).toContain('樣本少');
	});

	it('整季沒有上場紀錄時顯示明確文案', () => {
		expect(page(0)).toContain('本季尚無上場紀錄');
	});
});
