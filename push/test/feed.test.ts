import { describe, expect, it } from 'vitest';
import { hasResult, parseFeed } from '../src/feed';
import type { FeedMatch } from '../src/types';

const LINKS = { men: '/2026-27/men/matches/', women: '/2026-27/women/matches/', both: '/matches/' };

function match(overrides: Record<string, unknown> = {}) {
	return {
		group: 'men',
		game_id: 330,
		date: '2026-10-24',
		start_time: '15:00',
		home: '雲林美津濃',
		away: '屏東台電',
		home_sets_won: 3,
		away_sets_won: 1,
		home_win_prob: 0.3992871485506498,
		...overrides
	};
}

function feed(overrides: Record<string, unknown> = {}) {
	return { version: 1, generated_at: '2026-11-16T18:29:06+08:00', cup_id: 22, links: LINKS, matches: [match()], ...overrides };
}

function without(source: Record<string, unknown>, key: string) {
	const copy = { ...source };
	delete copy[key];
	return copy;
}

describe('parseFeed 合法的檔', () => {
	it('實際樣子的檔原樣讀出 cup_id、links、matches', () => {
		expect(parseFeed(feed())).toEqual({ cup_id: 22, links: LINKS, matches: [match()] });
	});

	it('matches 可以是空陣列', () => {
		expect(parseFeed(feed({ matches: [] }))).toEqual({ cup_id: 22, links: LINKS, matches: [] });
	});

	it('start_time、局數、勝率都可以是 null', () => {
		const loose = match({ start_time: null, home_sets_won: null, away_sets_won: null, home_win_prob: null });
		expect(parseFeed(feed({ matches: [loose] }))?.matches).toEqual([loose]);
	});

	it('局數 0、勝率 0 與 1 是合法值', () => {
		const edge = [match({ home_sets_won: 0, away_sets_won: 0, home_win_prob: 0 }), match({ game_id: 331, home_win_prob: 1 })];
		expect(parseFeed(feed({ matches: edge }))?.matches).toHaveLength(2);
	});

	it('女子組的場次合法', () => {
		expect(parseFeed(feed({ matches: [match({ group: 'women' })] }))?.matches[0].group).toBe('women');
	});

	it('多出來的未知欄位不影響', () => {
		const result = parseFeed(feed({ extra: 1, matches: [match({ venue: '天母' })] }));
		expect(result).not.toBeNull();
		expect(result?.cup_id).toBe(22);
	});

	it('links 帶子路徑與點、底線、連字號的片段合法', () => {
		const links = { men: '/a/b.c_d-e/', women: '/x/', both: '/' };
		expect(parseFeed(feed({ links }))?.links).toEqual(links);
	});
});

describe('parseFeed 整份不是物件', () => {
	it.each([null, undefined, [], 'text', 1, true])('%j 回 null', (data) => {
		expect(parseFeed(data)).toBeNull();
	});
});

describe('parseFeed 頂層欄位', () => {
	it('基準：合法的檔不是 null', () => {
		expect(parseFeed(feed())).not.toBeNull();
	});

	it.each([2, '1', 0, null, 1.5])('version 是 %j 回 null', (version) => {
		expect(parseFeed(feed({ version }))).toBeNull();
	});

	it.each(['version', 'cup_id', 'links', 'matches'])('缺 %s 回 null', (key) => {
		expect(parseFeed(without(feed(), key))).toBeNull();
	});

	it.each([22.5, '22', null])('cup_id 是 %j 回 null', (cup_id) => {
		expect(parseFeed(feed({ cup_id }))).toBeNull();
	});

	it.each([null, {}, 'matches', 5])('matches 是 %j（不是陣列）回 null', (matches) => {
		expect(parseFeed(feed({ matches }))).toBeNull();
	});

	it('matches 裡有一項不是物件回 null', () => {
		expect(parseFeed(feed({ matches: [match(), null] }))).toBeNull();
	});

	it('只要一場不合格，整份回 null', () => {
		expect(parseFeed(feed({ matches: [match(), match({ game_id: 'x' })] }))).toBeNull();
	});
});

describe('parseFeed links', () => {
	it.each(['//evil.com/', '/a b/', '/x#y/', '/x?y=1/', 'matches/', '/matches', '', '/a//b/', 'https://evil.com/', '/a/../b', '/é/', 5, null])(
		'men 是 %j 回 null',
		(men) => {
			expect(parseFeed(feed({ links: { ...LINKS, men } }))).toBeNull();
		}
	);

	it.each(['women', 'both'])('%s 不合格也回 null', (key) => {
		expect(parseFeed(feed({ links: { ...LINKS, [key]: '//evil.com/' } }))).toBeNull();
	});

	it.each(['men', 'women', 'both'])('缺 %s 回 null', (key) => {
		expect(parseFeed(feed({ links: without(LINKS, key) }))).toBeNull();
	});

	it.each([null, 'links', [], undefined])('links 是 %j 回 null', (links) => {
		expect(parseFeed(feed({ links }))).toBeNull();
	});
});

describe('parseFeed 每一場', () => {
	function rejects(overrides: Record<string, unknown>) {
		return parseFeed(feed({ matches: [match(overrides)] }));
	}

	it('基準：只有一場合法時不是 null', () => {
		expect(rejects({})).not.toBeNull();
	});

	it.each(['other', 'Men', '', null, 1])('group 是 %j 回 null', (group) => {
		expect(rejects({ group })).toBeNull();
	});

	it.each(['330', 1.5, null, NaN, Infinity])('game_id 是 %j 回 null', (game_id) => {
		expect(rejects({ game_id })).toBeNull();
	});

	it.each(['2026-11-8', '2026-1-08', '20261108', '2026-11-08T00:00', '', null, 20261108])('date 是 %j 回 null', (date) => {
		expect(rejects({ date })).toBeNull();
	});

	it.each(['9:00', '09:00:00', '0900', '', 900])('start_time 是 %j 回 null', (start_time) => {
		expect(rejects({ start_time })).toBeNull();
	});

	it.each(['home', 'away'])('%s 是空字串回 null', (key) => {
		expect(rejects({ [key]: '' })).toBeNull();
	});

	it.each(['home', 'away'])('%s 不是字串回 null', (key) => {
		expect(rejects({ [key]: 5 })).toBeNull();
		expect(rejects({ [key]: null })).toBeNull();
	});

	it.each(['home_sets_won', 'away_sets_won'])('%s 是 -1、1.5、字串回 null', (key) => {
		expect(rejects({ [key]: -1 })).toBeNull();
		expect(rejects({ [key]: 1.5 })).toBeNull();
		expect(rejects({ [key]: '3' })).toBeNull();
	});

	it.each([1.2, -0.1, '0.5', NaN])('home_win_prob 是 %j 回 null', (home_win_prob) => {
		expect(rejects({ home_win_prob })).toBeNull();
	});

	it.each(['group', 'game_id', 'date', 'start_time', 'home', 'away', 'home_sets_won', 'away_sets_won', 'home_win_prob'])('缺 %s 回 null', (key) => {
		expect(parseFeed(feed({ matches: [without(match(), key)] }))).toBeNull();
	});
});

describe('hasResult', () => {
	function game(home: number | null, away: number | null): FeedMatch {
		return { ...match(), group: 'men', home_sets_won: home, away_sets_won: away };
	}

	it('兩隊局數都有值是 true，包含 0', () => {
		expect(hasResult(game(3, 1))).toBe(true);
		expect(hasResult(game(0, 3))).toBe(true);
	});

	it('任一隊是 null 是 false', () => {
		expect(hasResult(game(null, 1))).toBe(false);
		expect(hasResult(game(3, null))).toBe(false);
	});

	it('兩隊都是 null 是 false', () => {
		expect(hasResult(game(null, null))).toBe(false);
	});
});
