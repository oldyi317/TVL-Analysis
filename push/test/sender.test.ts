import { env } from 'cloudflare:workers';
import { describe, expect, it } from 'vitest';
import { tick } from '../src/sender';
import type { Push } from '../src/types';

const LINKS = { men: '/2026-27/men/matches/', women: '/2026-27/women/matches/', both: '/matches/' };

function game(id: number, overrides: Record<string, unknown> = {}) {
	return {
		group: 'men',
		game_id: id,
		date: '2026-11-07',
		start_time: '13:00',
		home: `主隊${id}`,
		away: `客隊${id}`,
		home_sets_won: 3,
		away_sets_won: 1,
		home_win_prob: null,
		...overrides
	};
}

// 今天（2026-11-08）還沒打的一場。
function todayGame(id: number) {
	return game(id, { date: '2026-11-08', start_time: '14:00', home_sets_won: null, away_sets_won: null });
}

function feedOf(matches: unknown[]) {
	return { version: 1, cup_id: 22, generated_at: '2026-11-08T00:00:00Z', links: LINKS, matches };
}

// 台北時間的某一分鐘，例如 at('10:00') 是 2026-11-08T02:00:00Z。
function at(time: string): Date {
	return new Date(`2026-11-08T${time}:00+08:00`);
}

function fakeFetch(initial: unknown) {
	const state = { body: initial, urls: [] as string[] };
	const fetcher = (async (input: RequestInfo | URL) => {
		state.urls.push(String(input));
		return new Response(JSON.stringify(state.body));
	}) as typeof fetch;
	return { fetcher, state };
}

interface Call {
	endpoint: string;
	title: string;
	payload: unknown;
}

function fakePush() {
	const calls: Call[] = [];
	const push: Push = async (target, payload) => {
		const parsed = JSON.parse(payload);
		calls.push({ endpoint: target.endpoint, title: parsed.notification.title, payload: parsed });
		return 201;
	};
	return { push, calls };
}

function world(feed: unknown = feedOf([])) {
	const f = fakeFetch(feed);
	const p = fakePush();
	const run = (when: string, e: typeof env = env) => tick(e, at(when), { fetch: f.fetcher, push: p.push });
	return { run, feed: f.state, calls: p.calls };
}

async function addMany(count: number): Promise<string[]> {
	const endpoints = Array.from({ length: count }, (_, i) => `https://push.example/p${i + 1}`);
	await env.DB.batch(
		endpoints.map((e) =>
			env.DB.prepare(
				`INSERT INTO subscriptions (endpoint, p256dh, auth, want_results, want_today, group_pref, created_at, updated_at)
				VALUES (?, 'k', 'a', 1, 1, 'both', 't', 't')`
			).bind(e)
		)
	);
	return endpoints;
}

async function count(table: string): Promise<number> {
	const row = await env.DB.prepare(`SELECT COUNT(*) AS n FROM ${table}`).first<{ n: number }>();
	return row!.n;
}

async function finished(kind: string): Promise<boolean> {
	const row = await env.DB.prepare('SELECT finished_at FROM notifications WHERE kind = ?').bind(kind).first<{ finished_at: string | null }>();
	return row!.finished_at !== null;
}

function expectOnceEach(calls: Call[], title: string, endpoints: string[]) {
	const counts = new Map<string, number>();
	for (const call of calls.filter((c) => c.title === title)) counts.set(call.endpoint, (counts.get(call.endpoint) ?? 0) + 1);
	expect([...counts.keys()].sort()).toEqual([...endpoints].sort());
	expect([...counts.values()].every((n) => n === 1)).toBe(true);
}

describe('tick 何時讀檔', () => {
	it('只有台北時間分鐘數是 10 的倍數才讀檔', async () => {
		await addMany(1);
		const w = world();
		await w.run('10:01');
		await w.run('10:05');
		await w.run('10:09');
		expect(w.feed.urls).toEqual([]);
		await w.run('10:10');
		expect(w.feed.urls).toEqual(['https://vstat.tw/push/matches.json']);
	});

	it('不是讀檔的分鐘照樣送批次', async () => {
		await addMany(20);
		const w = world();
		await w.run('09:50');
		w.feed.body = feedOf([game(1)]);
		await w.run('10:00');
		expect(w.calls).toHaveLength(15);
		const reads = w.feed.urls.length;
		await w.run('10:11');
		expect(w.calls).toHaveLength(20);
		expect(w.feed.urls).toHaveLength(reads);
	});

	it('讀檔的那一分鐘讀完也送一批', async () => {
		await addMany(1);
		const w = world();
		await w.run('09:50');
		w.feed.body = feedOf([game(1)]);
		await w.run('10:00');
		expect(w.calls.map((c) => c.payload)).toEqual([
			{
				web_push: 8030,
				notification: {
					title: '男子組賽果',
					body: '11/7 主隊1 3–1 客隊1',
					navigate: 'https://vstat.tw/2026-27/men/matches/#d-2026-11-07'
				}
			}
		]);
	});
});

describe('tick 啟用與時段', () => {
	it('啟用當下不發，下一次讀檔只通知新的那一場', async () => {
		await addMany(1);
		const w = world(feedOf([game(1)]));
		await w.run('10:00');
		expect(w.feed.urls).toHaveLength(1);
		expect(w.calls).toEqual([]);
		expect(await count('notifications')).toBe(0);
		w.feed.body = feedOf([game(1), game(2)]);
		await w.run('10:10');
		expect(w.calls).toHaveLength(1);
		expect((w.calls[0].payload as { notification: { body: string } }).notification.body).toBe('11/7 主隊2 3–1 客隊2');
	});

	it('已開始的在發送時段外照樣送完', async () => {
		const endpoints = await addMany(40);
		const w = world();
		await w.run('21:50');
		w.feed.body = feedOf([game(1)]);
		const sizes: number[] = [];
		for (const time of ['22:00', '22:01', '22:02']) {
			const before = w.calls.length;
			await w.run(time);
			sizes.push(w.calls.length - before);
		}
		expect(sizes).toEqual([15, 15, 10]);
		expect(await finished('results')).toBe(false);
		await w.run('22:03');
		expect(await finished('results')).toBe(true);
		expectOnceEach(w.calls, '男子組賽果', endpoints);
	});
});

describe('tick 兩則與並行', () => {
	it('兩則相撞時今日賽事先送，送完賽果摘要從停下處接著送', async () => {
		const endpoints = await addMany(40);
		const all = { ...env, SEND_WINDOW: '00:00-23:59' };
		const w = world();
		await w.run('09:40', all);
		w.feed.body = feedOf([game(1)]);
		await w.run('09:50', all);
		expect(w.calls).toHaveLength(15);
		w.feed.body = feedOf([game(1), todayGame(2)]);
		const titles: string[][] = [];
		for (const time of ['10:00', '10:01', '10:02', '10:03', '10:04', '10:05', '10:06']) {
			const before = w.calls.length;
			await w.run(time, all);
			titles.push(w.calls.slice(before).map((c) => c.title));
			if (time === '10:03') expect(await finished('today')).toBe(true);
		}
		expect(titles.map((t) => [...new Set(t)])).toEqual([
			['今日男子組賽事'],
			['今日男子組賽事'],
			['今日男子組賽事'],
			[],
			['男子組賽果'],
			['男子組賽果'],
			[]
		]);
		expect(titles.map((t) => t.length)).toEqual([15, 15, 10, 0, 15, 10, 0]);
		expect(await finished('results')).toBe(true);
		expectOnceEach(w.calls, '今日男子組賽事', endpoints);
		expectOnceEach(w.calls, '男子組賽果', endpoints);
	});

	it('並行的定時呼叫不重發', async () => {
		const endpoints = await addMany(20);
		const w = world();
		await w.run('09:50');
		w.feed.body = feedOf([game(1), todayGame(2)]);
		await Promise.all([w.run('10:00'), w.run('10:00')]);
		for (const time of ['10:01', '10:02', '10:03', '10:04', '10:05', '10:06', '10:07', '10:08', '10:09']) await w.run(time);
		expect(await count('notifications')).toBe(2);
		expect(await finished('results')).toBe(true);
		expect(await finished('today')).toBe(true);
		expectOnceEach(w.calls, '男子組賽果', endpoints);
		expectOnceEach(w.calls, '今日男子組賽事', endpoints);
	});
});
