import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import LeadersPage from './+page.svelte';
import LeaderBoard from '#lib/components/LeaderBoard.svelte';
import type { LeaderBoardView, LeaderRowView } from '#lib/leaders.js';
import { seasons, seasonTeam, teams } from '#lib/site.js';
import { COUNT_STAT_KEYS, RATE_STAT_KEYS, STAT_LABELS } from '#lib/statLabels.js';
import type { PageProps } from './$types';

const season = seasons[0];
const team = seasonTeam(teams.find((t) => t.group === 'men')!, season.cup_id);

const context = new Map([
	['__request__', { page: { url: new URL('http://localhost/TVL-Analysis/22/men/leaders/') } }]
]);

function row(rank: number, id: number, position: string | null = 'OH'): LeaderRowView {
	return {
		rank, player_id: id, name: `球員${id}號`, team_id: team.team_id, team_short: team.short_name,
		position, value: 100 - id, numerator: null, denominator: null, title: rank === 1 ? '得分王' : null,
		team
	};
}

function renderPage(rows: LeaderRowView[], summary = { through_date: '2026-11-02', games: 6 }) {
	const boards: LeaderBoardView[] =
		rows.length === 0 ? [] : [{ key: 'total_points', label: '總得分', kind: 'count', threshold: null, rows }];
	const props: PageProps = {
		params: { season: season.name, group: 'men' },
		data: { season, group: 'men', summary: rows.length === 0 ? null : summary, boards }
	};
	return render(LeadersPage, { props, context }).body;
}

const ranks = (html: string) => [...html.matchAll(/class="rank-dot[^"]*"[^>]*>(\d+)</g)].map((m) => m[1]);

describe('排行榜頁有單場數據', () => {
	it('出現 h1、統計範圍、總得分卡，並列名次同號且跳號', () => {
		const html = renderPage([row(1, 1), row(1, 2), row(3, 3)]);
		expect(html).toMatch(/<h1[^>]*>[^<]*排行榜<\/h1>/);
		expect(html).toContain('統計至 11 月 2 日 · 共 6 場');
		expect(html).toMatch(/<h2[^>]*>總得分<\/h2>/);
		expect(html).toContain('名次');
		expect(ranks(html)).toEqual(['1', '1', '3']);
		expect(html).toMatch(/<a[^>]*href="\/TVL-Analysis\/[^"]+\/men\/players\/1\/"[^>]*>[\s\S]*?球員1號[\s\S]*?<\/a>/);
	});

	it('前 10 名截斷並附展開鈕；第 10 名並列全部顯示', () => {
		const twelve = Array.from({ length: 12 }, (_, i) => row(i + 1, i + 1));
		const html = renderPage(twelve);
		expect(html).toContain('球員10號');
		expect(html).not.toContain('球員11號');
		expect(html).toMatch(/<button[^>]*aria-expanded="false"[^>]*>[^<]*展開全部（12 人）/);

		const tied = [...twelve.slice(0, 9), row(10, 10), row(10, 11), row(10, 12), row(13, 13)];
		const tiedHtml = renderPage(tied);
		expect(tiedHtml).toContain('球員12號');
		expect(tiedHtml).not.toContain('球員13號');
	});

	it('缺位置時不顯示位置', () => {
		const html = renderPage([row(1, 1, null)]);
		expect(html).not.toContain('class="position"');
		expect(renderPage([row(1, 1, 'OH')])).toContain('主攻');
	});
});

describe('排行榜頁整組無單場數據', () => {
	it('只出現空狀態', () => {
		const html = renderPage([]);
		expect(html).toContain('本季尚無單場數據');
		expect(html).not.toContain('總得分');
		expect(html).not.toContain('統計至');
	});
});

it('七個計數榜、四王卡與並列球員連結；稱號只出現在四榜及四王卡', () => {
	const titles = ['得分王', '攻擊王', '攔網王', '發球王'];
	const boards: LeaderBoardView[] = COUNT_STAT_KEYS.map((key, i) => ({
		key, label: STAT_LABELS[key], kind: 'count', threshold: null,
		rows: [row(1, 1), row(1, 2), row(3, 3)].map((r) => ({ ...r, title: r.rank === 1 ? titles[i] ?? null : null }))
	}));
	const props: PageProps = { params: { season: season.name, group: 'men' }, data: {
		season, group: 'men', summary: { through_date: '2026-11-02', games: 6 }, boards
	}};
	const html = render(LeadersPage, { props, context }).body;
	expect(html).toContain('本季四王');
	for (const key of COUNT_STAT_KEYS) {
		expect(html).toMatch(new RegExp(`<h2[^>]*>${STAT_LABELS[key]}</h2>`));
		expect(html).toContain(`href="#b-${key}"`);
	}
	for (const title of titles) expect(html.split(`>${title}<`).length - 1).toBe(3);
	const champions = html.slice(html.indexOf('aria-label="本季四王"'), html.indexOf('</section>', html.indexOf('aria-label="本季四王"')));
	expect(champions).toContain('球員1號');
	expect(champions).toContain('球員2號');
	expect(champions).not.toContain('球員3號');
});

it('預設計數榜並提供兩個可操作的分段', () => {
	const html = renderPage([row(1, 1)]);
	expect(html).toMatch(/<button[^>]*aria-pressed="true"[^>]*>計數榜<\/button>/);
	expect(html).toMatch(/<button[^>]*aria-pressed="false"[^>]*>比率榜<\/button>/);
});

it('比率卡顯示標題、門檻、比率與分子分母，無人達門檻時顯示空狀態', () => {
	const board: LeaderBoardView = {
		key: 'attack_success_rate', label: '攻擊成功率', kind: 'rate', threshold: '門檻：出賽 5 局、攻擊 10 次',
		rows: [{ ...row(1, 1), value: 37.5, numerator: 45, denominator: 120, title: null }]
	};
	const html = render(LeaderBoard, { props: { board, season: season.name, group: 'men' }, context }).body;
	expect(html).toMatch(/<h2[^>]*>攻擊成功率<\/h2>/);
	expect(html).toContain('門檻：出賽 5 局、攻擊 10 次');
	expect(html).toContain('37.5%');
	expect(html).toContain('45／120');
	const empty = render(LeaderBoard, { props: { board: { ...board, rows: [] }, season: season.name, group: 'men' }, context }).body;
	expect(empty).toContain('尚無球員達門檻');
});

it('只傳入計數榜與六個比率榜時，不產生每局攔網卡與 chip', () => {
	const boards: LeaderBoardView[] = [
		{ key: 'total_points', label: '總得分', kind: 'count', threshold: null, rows: [row(1, 1)] },
		...RATE_STAT_KEYS.filter((key) => key !== 'blocks_per_set').map((key): LeaderBoardView => ({
			key, label: STAT_LABELS[key], kind: 'rate', threshold: '門檻：出賽 2 場', rows: []
		}))
	];
	const props: PageProps = { params: { season: season.name, group: 'men' }, data: { season, group: 'men', summary: null, boards } };
	const html = render(LeadersPage, { props, context }).body;
	expect(html).not.toContain('每局攔網');
	expect(html).not.toContain('#b-blocks_per_set');
});
