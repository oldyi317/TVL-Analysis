import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import type { MatchTopBoardView } from '#lib/boxScore.js';
import { seasons, seasonTeam, teams } from '#lib/site.js';
import MatchTopPerformers from './MatchTopPerformers.svelte';

const [homeRaw, awayRaw] = teams.filter((t) => t.group === 'men');
const cupId = seasons[0].cup_id;
const homeTeam = seasonTeam(homeRaw, cupId);
const awayTeam = seasonTeam(awayRaw, cupId);

const sampleBoards: MatchTopBoardView[] = [
	{
		key: 'total_points',
		label: '得分',
		rows: [
			{ rank: 1, player_id: 10, name: '王小明', linked: true, team: homeTeam, value: 24 },
			{ rank: 1, player_id: 20, name: '李大華', linked: true, team: awayTeam, value: 24 },
			{ rank: 3, player_id: 99, name: '球員 #99', linked: false, team: homeTeam, value: 15 }
		]
	},
	{
		key: 'block_points',
		label: '攔網得分',
		rows: [
			{ rank: 1, player_id: 20, name: '李大華', linked: true, team: awayTeam, value: 5 },
			{ rank: 2, player_id: 10, name: '王小明', linked: true, team: homeTeam, value: 3 },
			{ rank: 3, player_id: 30, name: '張小強', linked: true, team: awayTeam, value: 2 },
			{ rank: 3, player_id: 40, name: '陳小安', linked: true, team: homeTeam, value: 2 }
		]
	}
];

describe('MatchTopPerformers', () => {
	const html = (boards = sampleBoards) =>
		render(MatchTopPerformers, { props: { boards, season: '2026-27', group: 'men' } })
			.body.replace(/<!--[\s\S]*?-->/g, '');

	it('每個非空項目各出一張卡片與表格，具欄列標頭、名次、隊色圓標與簡稱、數值', () => {
		const body = html();
		expect(body).toContain('aria-label="得分單場前三名"');
		expect(body).toContain('aria-label="攔網得分單場前三名"');
		expect(body).not.toContain('發球得分');
		expect(body).toMatch(
			/<th scope="col"[^>]*>名次<\/th>\s*<th scope="col"[^>]*>球員<\/th>\s*<th scope="col"[^>]*>得分<\/th>/
		);
		expect(body).toMatch(/<th scope="row"[^>]*>[\s\S]*?王小明[\s\S]*?24<\/td>/);
		expect(body).toContain(homeTeam.short_name);
		expect(body).toContain(awayTeam.short_name);
		expect(body).toContain('class="badge sm');
		expect(body).toMatch(/class="rank-dot [^"]*top"/);
		// 第一名並列兩人時下一名是第三名（1、1、3）；攔網第三名並列兩人一起列出（1、2、3、3）
		const [, scoreSection, blockSection] = body.split('<section');
		const scoreRanks = [...scoreSection.matchAll(/rank-dot[^>]*>(\d+)</g)].map((m) => m[1]);
		const blockRanks = [...blockSection.matchAll(/rank-dot[^>]*>(\d+)</g)].map((m) => m[1]);
		expect(scoreRanks).toEqual(['1', '1', '3']);
		expect(blockRanks).toEqual(['1', '2', '3', '3']);
	});

	it('有姓名的球員連到球員頁，姓名空白列照備援顯示且不連到球員頁', () => {
		const body = html();
		expect(body).toContain('href="/2026-27/men/players/10/"');
		expect(body).toContain('href="/2026-27/men/players/20/"');
		expect(body).toContain('球員 #99');
		expect(body).not.toContain('/players/99/');
	});

	it('略過空的項目列，全空時不輸出表格', () => {
		const body = html([{ key: 'serve_points', label: '發球得分', rows: [] }]);
		expect(body).not.toContain('<table');
		expect(body).not.toContain('發球得分');
	});
});
