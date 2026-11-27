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
	['__request__', { page: { url: new URL('http://localhost/22/men/players/1/') } }]
]);

const advancedEntry = (over: { below_threshold?: boolean; team_sets?: number; blocks_per_set?: number | null } = {}) =>
	({
		cup_id: stats.cup_id,
		group: player.group,
		player_id: player.player_id,
		position: 'MB',
		team_sets: over.team_sets ?? 36,
		below_threshold: over.below_threshold ?? false,
		eligible: [],
		ratios: {
			blocks_per_set: { numerator: 6, denominator: over.team_sets ?? 36, value: over.blocks_per_set === undefined ? 0.17 : over.blocks_per_set },
			attack_success_rate: { numerator: 5, denominator: 12, value: 41.67 }
		},
		pr: {}
	}) as unknown as PageProps['data']['advanced'];

function renderPage(games: number, advanced: PageProps['data']['advanced'] = null, gameLog: PageProps['data']['gameLog'] = []) {
	const props: PageProps = {
		params: { season: season.name, group: player.group, player: String(player.player_id) },
		form: null,
		data: {
			season, group: player.group, player, team: found.team,
			stats: { ...stats, position: 'MB', games, sets_played: null },
			advanced, distribution: null, gameLog
		}
	};
	return render(PlayerPage, { props, context });
}

const page = (games: number, advanced: PageProps['data']['advanced'] = null) => renderPage(games, advanced).body;

describe('球員頁逐場紀錄', () => {
	it('有紀錄就顯示欄列標頭、單場連結、季後賽與黃金決勝局標記', () => {
		const game: PageProps['data']['gameLog'][number] = {
			gameId: 310, date: '2026-03-21', stage: '總決賽', side: 'away', opponent: '對手隊',
			won: false, score: [2, 3], golden: true, playedGoldenSet: true,
			metrics: {
				attack_success_rate: [7, 22, 31.82], serve_point_rate: [1, 11, 9.09],
				blocks_per_set: [1, 5, 0.2], receive_excellent_rate: [6, 13, 46.15],
				dig_excellent_rate: [3, 5, 60], set_excellent_rate: [0, 0, null]
			},
			stats: {
				sets_played: null, total_points: 9, attack_points: 7, block_points: 1, serve_points: 1,
				attack_total: 22, serve_total: 11, receive_excellent: 6, receive_total: 13,
				dig_excellent: 3, dig_total: 5, set_excellent: 0, set_total: 0
			}
		};
		const regular = { ...game, gameId: 300, date: '2026-02-01', stage: '例行賽', playedGoldenSet: false };
		const html = renderPage(2, null, [regular, game]).body.replace(/<!--[\s\S]*?-->/g, '');
		expect(html).toContain('逐場紀錄');
		const table = html.match(/<table[^>]*>[\s\S]*?<\/table>/)![0];
		const headers = Array.from(table.matchAll(/<th scope="col"[^>]*>([^<]*)<\/th>/g), (m) => m[1]);
		expect(headers).toEqual([
			'日期', '對手', '勝敗', '得分', '攻擊', '攔網', '發球', '攻擊次數', '發球次數', '接發球', '防守', '舉球'
		]);
		expect(table).toContain('scope="row"');
		expect(table).toContain(`href="/${season.name}/${player.group}/matches/310/"`);
		expect(table).toContain(`href="/${season.name}/${player.group}/matches/300/"`);
		expect(table).not.toContain('例行賽');
		for (const text of ['客場', '對手隊', '敗', '2 : 3', '總決賽', '黃金決勝局']) expect(table).toContain(text);
		expect(table).toMatch(/>6／13<\/td>[\s\S]*?>3／5<\/td>[\s\S]*?>—<\/td>/);
		expect(table).not.toMatch(/出賽局數|合計|<tfoot/);
		expect(html).toContain('不含黃金決勝局');
	});

	it('沒有場次時連區塊標題與表下說明都不顯示', () => {
		const html = page(0);
		expect(html).not.toContain('逐場紀錄');
		expect(html).not.toContain('不含黃金決勝局');
	});
});

describe('球員頁 head', () => {
	it('title 維持原格式，並帶 Open Graph 與絕對網址', () => {
		const head = renderPage(2).head;
		const title = `${playerDisplayName(player)}｜${season.name} ${GROUP_LABEL[player.group]} 球員｜VStat 企排`;
		expect(head).toContain(`<title>${title}</title>`);
		expect(head).toContain(`<meta property="og:title" content="${title}"`);
		expect(head).toContain(
			'<meta property="og:url" content="https://vstat.tw/22/men/players/1/"'
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
