import { env } from 'cloudflare:workers';
import { describe, expect, it } from 'vitest';
import { checkFeed, claimMatches, claimToday } from '../src/judge';
import type { Content, FeedMatch } from '../src/types';

const LINKS = { men: '/2026-27/men/matches/', women: '/2026-27/women/matches/', both: '/matches/' };

function match(overrides: Partial<FeedMatch> = {}): FeedMatch {
	return {
		group: 'men',
		game_id: 1,
		date: '2026-11-07',
		start_time: '13:00',
		home: '臺電',
		away: '雲林美津濃',
		home_sets_won: 3,
		away_sets_won: 1,
		home_win_prob: null,
		...overrides
	};
}

function feedOf(matches: FeedMatch[], extra: Record<string, unknown> = {}) {
	return { version: 1, cup_id: 22, generated_at: '2026-11-08T00:00:00Z', links: LINKS, matches, ...extra };
}

// 台北時間的某一分鐘，例如 at('2026-11-08', '10:00') 是 2026-11-08T02:00:00Z。
function at(date: string, time: string): Date {
	return new Date(`${date}T${time}:00+08:00`);
}

function fakeFetch(respond: () => Response | Promise<Response>) {
	const urls: string[] = [];
	const fetcher = (async (input: RequestInfo | URL) => {
		urls.push(String(input));
		return respond();
	}) as typeof fetch;
	return { fetcher, urls };
}

function serving(body: unknown) {
	return fakeFetch(() => new Response(JSON.stringify(body)));
}

async function run(body: unknown, when: Date, e: typeof env = env) {
	const fake = serving(body);
	await checkFeed(e, when, fake.fetcher);
	return fake;
}

// 第一次讀檔什麼都不通知，先用一份空檔讓它過去。
async function prime() {
	await run(feedOf([]), at('2026-11-08', '07:00'));
}

async function count(table: string): Promise<number> {
	const row = await env.DB.prepare(`SELECT COUNT(*) AS n FROM ${table}`).first<{ n: number }>();
	return row!.n;
}

interface Stored {
	kind: string;
	men: Content | null;
	women: Content | null;
	both: Content | null;
}

function parse(text: string | null): Content | null {
	return text === null ? null : JSON.parse(text);
}

async function stored(): Promise<Stored[]> {
	const { results } = await env.DB.prepare(
		'SELECT kind, content_men, content_women, content_both FROM notifications ORDER BY id'
	).all<{ kind: string; content_men: string | null; content_women: string | null; content_both: string | null }>();
	return results.map((row) => ({
		kind: row.kind,
		men: parse(row.content_men),
		women: parse(row.content_women),
		both: parse(row.content_both)
	}));
}

async function sentKeys(): Promise<string[]> {
	const { results } = await env.DB.prepare(
		'SELECT cup_id, match_group, game_id FROM sent_matches ORDER BY cup_id, match_group, game_id'
	).all<{ cup_id: number; match_group: string; game_id: number }>();
	return results.map((row) => `${row.cup_id}/${row.match_group}/${row.game_id}`);
}

describe('claimMatches', () => {
	const NOW = at('2026-11-08', '10:00');

	it('第一次回傳全部傳入的場次，同一批再認領回空陣列', async () => {
		const batch = [match({ game_id: 1 }), match({ game_id: 2 })];
		expect(await claimMatches(env.DB, 22, batch, NOW)).toEqual(batch);
		expect(await claimMatches(env.DB, 22, batch, NOW)).toEqual([]);
	});

	it('部分重疊只回新的', async () => {
		await claimMatches(env.DB, 22, [match({ game_id: 1 })], NOW);
		const fresh = match({ game_id: 2 });
		expect(await claimMatches(env.DB, 22, [match({ game_id: 1 }), fresh], NOW)).toEqual([fresh]);
	});

	it('game_id 相同但組別不同是兩場', async () => {
		await claimMatches(env.DB, 22, [match({ group: 'men', game_id: 5 })], NOW);
		const women = match({ group: 'women', game_id: 5 });
		expect(await claimMatches(env.DB, 22, [women], NOW)).toEqual([women]);
	});

	it('game_id 與組別相同但 cupId 不同是兩場', async () => {
		await claimMatches(env.DB, 22, [match({ game_id: 5 })], NOW);
		const same = match({ game_id: 5 });
		expect(await claimMatches(env.DB, 23, [same], NOW)).toEqual([same]);
		expect(await sentKeys()).toEqual(['22/men/5', '23/men/5']);
	});

	it('傳空陣列回空陣列', async () => {
		expect(await claimMatches(env.DB, 22, [], NOW)).toEqual([]);
	});
});

describe('claimToday', () => {
	const NOW = at('2026-11-08', '10:00');

	it('同一個日期第一次 true、第二次 false', async () => {
		expect(await claimToday(env.DB, '2026-11-08', NOW)).toBe(true);
		expect(await claimToday(env.DB, '2026-11-08', NOW)).toBe(false);
	});

	it('不同日期各自 true', async () => {
		expect(await claimToday(env.DB, '2026-11-08', NOW)).toBe(true);
		expect(await claimToday(env.DB, '2026-11-09', NOW)).toBe(true);
	});
});

describe('checkFeed 讀檔失敗不留痕跡', () => {
	const WHEN = at('2026-11-08', '10:00');
	const VALID = feedOf([match({ date: '2026-11-08' })]);
	const TABLES = ['first_feed_read', 'sent_matches', 'sent_today_dates', 'notifications'];

	const failures: Array<[string, () => Response | Promise<Response>]> = [
		['fetcher reject', () => Promise.reject(new Error('連不上'))],
		['狀態碼 404', () => new Response(JSON.stringify(VALID), { status: 404 })],
		['狀態碼 500', () => new Response(JSON.stringify(VALID), { status: 500 })],
		['本文不是 JSON', () => new Response('<html>不是 JSON</html>')],
		['version 是 2', () => new Response(JSON.stringify({ ...VALID, version: 2 }))],
		['links 格式不對', () => new Response(JSON.stringify({ ...VALID, links: { ...LINKS, men: '//evil.com/' } }))]
	];

	it.each(failures)('%s', async (_name, respond) => {
		const fake = fakeFetch(respond);
		await checkFeed(env, WHEN, fake.fetcher);
		expect(fake.urls).toEqual([env.FEED_URL]);
		for (const table of TABLES) expect(await count(table), table).toBe(0);
	});

	it('對照：同樣的時刻換成合法的檔就會寫出列', async () => {
		const fake = await run(feedOf([match({ date: '2026-11-08' })]), WHEN);
		expect(fake.urls).toEqual([env.FEED_URL]);
		expect(await count('first_feed_read')).toBe(1);
		expect(await count('sent_matches')).toBe(1);
	});
});

describe('checkFeed 第一次成功讀檔', () => {
	it('已有賽果的場次全記成已發，沒有賽果的不記，不通知也不建今日賽事', async () => {
		const matches = [
			match({ game_id: 1, date: '2026-11-08' }),
			match({ game_id: 2, date: '2026-11-01' }),
			match({ game_id: 3, date: '2026-10-01' }),
			match({ game_id: 4, date: '2026-11-08', home_sets_won: null, away_sets_won: null }),
			match({ group: 'women', game_id: 1, date: '2026-11-07' })
		];
		const when = at('2026-11-08', '10:00');
		expect(matches.some((m) => m.date === '2026-11-08')).toBe(true);
		const fake = await run(feedOf(matches), when);
		expect(fake.urls).toHaveLength(1);
		expect(await count('first_feed_read')).toBe(1);
		expect(await sentKeys()).toEqual(['22/men/1', '22/men/2', '22/men/3', '22/women/1']);
		expect(await count('notifications')).toBe(0);
		expect(await count('sent_today_dates')).toBe(0);
	});

	it('檔裡一場賽果都沒有時第一次讀檔照樣算數', async () => {
		await run(feedOf([match({ game_id: 1, home_sets_won: null, away_sets_won: null })]), at('2026-11-08', '12:00'));
		expect(await count('first_feed_read')).toBe(1);
		expect(await count('sent_matches')).toBe(0);
		const result = match({ game_id: 2, date: '2026-11-08' });
		await run(feedOf([result]), at('2026-11-08', '12:10'));
		const rows = await stored();
		expect(rows).toHaveLength(1);
		expect(rows[0].kind).toBe('results');
	});

	it('第一次之後再讀同一份檔沒有新通知', async () => {
		const body = feedOf([match({ game_id: 1, date: '2026-11-08' })]);
		await run(body, at('2026-11-08', '12:00'));
		const second = await run(body, at('2026-11-08', '12:10'));
		expect(second.urls).toHaveLength(1);
		expect(await count('first_feed_read')).toBe(1);
		expect(await sentKeys()).toEqual(['22/men/1']);
		expect(await count('notifications')).toBe(0);
	});
});

describe('checkFeed 賽果摘要', () => {
	const NOON = at('2026-11-08', '12:00');

	it('新賽果進站建立一則 results，三個版本逐字', async () => {
		await prime();
		const men = match({ group: 'men', game_id: 1, date: '2026-11-07', home: '臺電', away: '雲林美津濃', home_sets_won: 3, away_sets_won: 1 });
		const women = match({ group: 'women', game_id: 2, date: '2026-11-08', home: '台南', away: '高雄', home_sets_won: 0, away_sets_won: 3 });
		await run(feedOf([men, women]), NOON);
		expect(await stored()).toEqual([
			{
				kind: 'results',
				men: { title: '男子組賽果', body: '11/7 臺電 3–1 雲林美津濃', navigate: 'https://vstat.tw/2026-27/men/matches/#d-2026-11-07' },
				women: { title: '女子組賽果', body: '11/8 台南 0–3 高雄', navigate: 'https://vstat.tw/2026-27/women/matches/#d-2026-11-08' },
				both: {
					title: '最新賽果',
					body: '11/7 男子 臺電 3–1 雲林美津濃\n11/8 女子 台南 0–3 高雄',
					navigate: 'https://vstat.tw/matches/#d-2026-11-08'
				}
			}
		]);
	});

	it('只有男子組有新賽果時 content_women 是 NULL', async () => {
		await prime();
		await run(feedOf([match({ game_id: 329, date: '2026-10-24', home: '三福產業', away: '臺北鳳凰', home_sets_won: 0, away_sets_won: 3, start_time: '13:00' })]), at('2026-10-31', '12:00'));
		const row = await env.DB.prepare('SELECT content_men, content_women, content_both FROM notifications').first<Record<string, string | null>>();
		expect(row!.content_women).toBeNull();
		expect(JSON.parse(row!.content_men!)).toEqual({
			title: '男子組賽果',
			body: '10/24 三福產業 0–3 臺北鳳凰',
			navigate: 'https://vstat.tw/2026-27/men/matches/#d-2026-10-24'
		});
		expect(row!.content_both).toBe(row!.content_men);
	});
});

describe('checkFeed 7 天規則', () => {
	const NOON = at('2026-11-08', '12:00');

	it('差 7 天的進通知', async () => {
		await prime();
		await run(feedOf([match({ game_id: 1, date: '2026-11-01', home: '臺電', away: '雲林美津濃' })]), NOON);
		const rows = await stored();
		expect(rows).toHaveLength(1);
		expect(rows[0].both!.body).toBe('11/1 臺電 3–1 雲林美津濃');
	});

	it('差 8 天的進 sent_matches 但不進通知', async () => {
		await prime();
		const fake = await run(feedOf([match({ game_id: 1, date: '2026-10-31' })]), NOON);
		expect(fake.urls).toHaveLength(1);
		expect(await sentKeys()).toEqual(['22/men/1']);
		expect(await count('notifications')).toBe(0);
	});

	it('同一次讀檔兩種都有時，通知只含 7 天內的', async () => {
		await prime();
		await run(
			feedOf([
				match({ game_id: 1, date: '2026-10-31', home: '舊隊', away: '舊對手' }),
				match({ game_id: 2, date: '2026-11-01', home: '臺電', away: '雲林美津濃' })
			]),
			NOON
		);
		expect(await sentKeys()).toEqual(['22/men/1', '22/men/2']);
		const rows = await stored();
		expect(rows).toHaveLength(1);
		expect(rows[0].both!.body).toBe('11/1 臺電 3–1 雲林美津濃');
	});

	it('只有超過 7 天的時不建立通知', async () => {
		await prime();
		await run(feedOf([match({ game_id: 1, date: '2026-10-20' }), match({ game_id: 2, date: '2026-10-25' })]), NOON);
		expect(await sentKeys()).toEqual(['22/men/1', '22/men/2']);
		expect(await count('notifications')).toBe(0);
	});

	it('以台北日曆日算：UTC 日期與台北日期不同的時刻', async () => {
		await prime();
		const late = { ...env, SEND_WINDOW: '00:00-23:59' };
		const when = new Date('2026-11-07T23:50:00Z');
		await run(
			feedOf([
				match({ game_id: 1, date: '2026-10-31', home: '舊隊', away: '舊對手' }),
				match({ game_id: 2, date: '2026-11-01', home: '臺電', away: '雲林美津濃' })
			]),
			when,
			late
		);
		expect(await sentKeys()).toEqual(['22/men/1', '22/men/2']);
		const rows = await stored();
		expect(rows).toHaveLength(1);
		expect(rows[0].both!.body).toBe('11/1 臺電 3–1 雲林美津濃');
	});
});

describe('checkFeed 同一場不發第二次', () => {
	it('賽果事後更正再讀檔不建立新通知', async () => {
		await prime();
		await run(feedOf([match({ game_id: 1 })]), at('2026-11-08', '12:00'));
		expect(await count('notifications')).toBe(1);
		const fake = await run(feedOf([match({ game_id: 1, home_sets_won: 1, away_sets_won: 3 })]), at('2026-11-08', '12:10'));
		expect(fake.urls).toHaveLength(1);
		expect(await count('notifications')).toBe(1);
	});

	it('同一天第二批不同賽果進站建立第二則，內容只含新的場次', async () => {
		await prime();
		const first = match({ game_id: 1, home: '臺電', away: '雲林美津濃' });
		const second = match({ game_id: 2, home: '台南', away: '高雄', home_sets_won: 2, away_sets_won: 3 });
		await run(feedOf([first]), at('2026-11-08', '12:00'));
		await run(feedOf([first, second]), at('2026-11-08', '15:00'));
		const rows = await stored();
		expect(rows.map((row) => row.both!.body)).toEqual(['11/7 臺電 3–1 雲林美津濃', '11/7 台南 2–3 高雄']);
	});
});

describe('checkFeed 發送時段', () => {
	const NEW_RESULT = feedOf([match({ game_id: 1, date: '2026-11-07' })]);

	it('07:50 不建立也不認領，08:00 同一份檔就建立', async () => {
		await prime();
		const early = await run(NEW_RESULT, at('2026-11-08', '07:50'));
		expect(early.urls).toHaveLength(1);
		expect(await count('first_feed_read')).toBe(1);
		expect(await count('notifications')).toBe(0);
		expect(await count('sent_matches')).toBe(0);
		await run(NEW_RESULT, at('2026-11-08', '08:00'));
		expect(await count('notifications')).toBe(1);
	});

	it('22:00 建立', async () => {
		await prime();
		await run(NEW_RESULT, at('2026-11-08', '22:00'));
		expect(await count('notifications')).toBe(1);
	});

	it('22:10 不建立也不認領', async () => {
		await prime();
		const late = await run(NEW_RESULT, at('2026-11-08', '22:10'));
		expect(late.urls).toHaveLength(1);
		expect(await count('first_feed_read')).toBe(1);
		expect(await count('notifications')).toBe(0);
		expect(await count('sent_matches')).toBe(0);
	});

	it('SEND_WINDOW 解析不出來時不建立、不認領', async () => {
		await prime();
		const broken = { ...env, SEND_WINDOW: '全天' };
		await run(NEW_RESULT, at('2026-11-08', '12:00'), broken);
		expect(await count('notifications')).toBe(0);
		expect(await count('sent_matches')).toBe(0);
		await run(NEW_RESULT, at('2026-11-08', '12:00'));
		expect(await count('notifications')).toBe(1);
	});
});

describe('checkFeed 今日賽事', () => {
	const TODAY = '2026-11-08';
	const TODAYS = feedOf([
		match({ group: 'men', game_id: 10, date: TODAY, start_time: '14:00', home: '臺電', away: '雲林美津濃', home_sets_won: null, away_sets_won: null, home_win_prob: 0.62 }),
		match({ group: 'women', game_id: 11, date: TODAY, start_time: '16:00', home: '台南', away: '高雄', home_sets_won: null, away_sets_won: null, home_win_prob: null })
	]);

	it('10:00 建立一則 today，三個版本逐字', async () => {
		await prime();
		await run(TODAYS, at(TODAY, '10:00'));
		expect(await stored()).toEqual([
			{
				kind: 'today',
				men: { title: '今日男子組賽事', body: '14:00 臺電 62%–38% 雲林美津濃', navigate: 'https://vstat.tw/2026-27/men/matches/#d-2026-11-08' },
				women: { title: '今日女子組賽事', body: '16:00 台南 對 高雄', navigate: 'https://vstat.tw/2026-27/women/matches/#d-2026-11-08' },
				both: {
					title: '今日賽事',
					body: '14:00 男子 臺電 62%–38% 雲林美津濃\n16:00 女子 台南 對 高雄',
					navigate: 'https://vstat.tw/matches/#d-2026-11-08'
				}
			}
		]);
		expect(await count('sent_today_dates')).toBe(1);
	});

	it('同一天 10:10 再讀不建立第二則', async () => {
		await prime();
		await run(TODAYS, at(TODAY, '10:00'));
		const again = await run(TODAYS, at(TODAY, '10:10'));
		expect(again.urls).toHaveLength(1);
		expect(await count('notifications')).toBe(1);
	});

	it('09:50 不建立，11:00 建立', async () => {
		await prime();
		await run(TODAYS, at(TODAY, '09:50'));
		expect(await count('notifications')).toBe(0);
		expect(await count('sent_today_dates')).toBe(0);
		await run(TODAYS, at(TODAY, '11:00'));
		expect(await count('notifications')).toBe(1);
	});

	it('11:10 不建立', async () => {
		await prime();
		await run(TODAYS, at(TODAY, '11:10'));
		expect(await count('first_feed_read')).toBe(1);
		expect(await count('notifications')).toBe(0);
		expect(await count('sent_today_dates')).toBe(0);
	});

	it('10:00 讀檔失敗、10:10 成功建立', async () => {
		await prime();
		const failing = fakeFetch(() => new Response(JSON.stringify(TODAYS), { status: 500 }));
		await checkFeed(env, at(TODAY, '10:00'), failing.fetcher);
		expect(failing.urls).toHaveLength(1);
		expect(await count('notifications')).toBe(0);
		expect(await count('sent_today_dates')).toBe(0);
		await run(TODAYS, at(TODAY, '10:10'));
		expect(await count('notifications')).toBe(1);
	});

	it('檔裡沒有今天的賽事不建立也不認領日期，之後出現就建立', async () => {
		await prime();
		const tomorrow = feedOf([match({ game_id: 10, date: '2026-11-09', home_sets_won: null, away_sets_won: null })]);
		const first = await run(tomorrow, at(TODAY, '10:00'));
		expect(first.urls).toHaveLength(1);
		expect(await count('notifications')).toBe(0);
		expect(await count('sent_today_dates')).toBe(0);
		await run(TODAYS, at(TODAY, '10:20'));
		expect(await count('notifications')).toBe(1);
	});

	it('sent_today_dates 已經有今天（別的呼叫先認領）就不建立', async () => {
		await prime();
		expect(await claimToday(env.DB, TODAY, at(TODAY, '09:59'))).toBe(true);
		await run(TODAYS, at(TODAY, '10:00'));
		expect(await count('sent_today_dates')).toBe(1);
		expect(await count('notifications')).toBe(0);
	});

	it('TODAY_WINDOW 解析不出來時不建立今日賽事，賽果摘要照常', async () => {
		await prime();
		const broken = { ...env, TODAY_WINDOW: '十點' };
		const body = feedOf([...TODAYS.matches, match({ game_id: 1, date: '2026-11-07' })]);
		await run(body, at(TODAY, '10:00'), broken);
		expect((await stored()).map((row) => row.kind)).toEqual(['results']);
		expect(await count('sent_today_dates')).toBe(0);
	});

	it('兩種都要發時兩則都建立，今日賽事的 id 比賽果摘要小', async () => {
		await prime();
		const body = feedOf([...TODAYS.matches, match({ game_id: 1, date: '2026-11-07' })]);
		await run(body, at(TODAY, '10:00'));
		expect((await stored()).map((row) => row.kind)).toEqual(['today', 'results']);
	});
});

describe('checkFeed 並行', () => {
	it('兩個同時讀到同一批新賽果，每場恰好出現一次', async () => {
		await prime();
		const body = feedOf([
			match({ game_id: 1, home: '甲隊', away: '乙隊' }),
			match({ game_id: 2, home: '丙隊', away: '丁隊' }),
			match({ game_id: 3, home: '戊隊', away: '己隊' })
		]);
		const when = at('2026-11-08', '12:00');
		await Promise.all([run(body, when), run(body, when)]);
		const rows = await stored();
		expect(rows.length).toBeGreaterThan(0);
		const lines = rows.flatMap((row) => row.both!.body.split('\n')).sort();
		expect(lines).toEqual(['11/7 丙隊 3–1 丁隊', '11/7 戊隊 3–1 己隊', '11/7 甲隊 3–1 乙隊']);
	});

	it('兩個同時在 10:00 讀到今天的賽事，今日賽事只有一則', async () => {
		await prime();
		const body = feedOf([match({ game_id: 10, date: '2026-11-08', home_sets_won: null, away_sets_won: null })]);
		const when = at('2026-11-08', '10:00');
		await Promise.all([run(body, when), run(body, when)]);
		expect((await stored()).map((row) => row.kind)).toEqual(['today']);
	});
});
