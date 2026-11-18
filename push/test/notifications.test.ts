import { env } from 'cloudflare:workers';
import { describe, expect, it } from 'vitest';
import { claimBatch, createNotification, sendNextBatch } from '../src/notifications';
import { upsertSubscription } from '../src/store';
import type { Content, Push, Versions } from '../src/types';

const NOW = new Date('2026-11-08T02:00:00Z');

function content(tag: string): Content {
	return { title: `${tag}標題`, body: `${tag}內文`, navigate: `https://vstat.tw/${tag}/#d-2026-11-08` };
}

const ALL: Versions = { men: content('男'), women: content('女'), both: content('合') };
const MEN_ONLY: Versions = { men: content('男'), women: null, both: content('男') };

interface Call {
	endpoint: string;
	payload: { web_push: number; notification: Content };
}

function fakePush(statuses: Record<string, number | 'reject'> = {}) {
	const calls: Call[] = [];
	const push: Push = async (target, payload) => {
		calls.push({ endpoint: target.endpoint, payload: JSON.parse(payload) });
		const status = statuses[target.endpoint] ?? 201;
		if (status === 'reject') throw new Error('連不上');
		return status;
	};
	return { push, calls };
}

async function addSub(endpoint: string, opts: { results?: number; today?: number; group?: string } = {}) {
	await env.DB.prepare(
		`INSERT INTO subscriptions (endpoint, p256dh, auth, want_results, want_today, group_pref, created_at, updated_at)
		VALUES (?, 'k', 'a', ?, ?, ?, 't', 't')`
	)
		.bind(endpoint, opts.results ?? 1, opts.today ?? 1, opts.group ?? 'both')
		.run();
}

async function addMany(count: number, prefix = 'p'): Promise<string[]> {
	const endpoints = Array.from({ length: count }, (_, i) => `https://push.example/${prefix}${i + 1}`);
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

async function notificationRows() {
	const { results } = await env.DB.prepare('SELECT * FROM notifications ORDER BY id').all<Record<string, unknown>>();
	return results;
}

async function endpointsLeft(): Promise<string[]> {
	const { results } = await env.DB.prepare('SELECT endpoint FROM subscriptions ORDER BY id').all<{ endpoint: string }>();
	return results.map((r) => r.endpoint);
}

function countBy(calls: Call[]): Map<string, number> {
	const counts = new Map<string, number>();
	for (const c of calls) counts.set(c.endpoint, (counts.get(c.endpoint) ?? 0) + 1);
	return counts;
}

// 連續送批次，回每批發出的筆數；某次一筆都沒發（送完或沒有待送）就停。
async function drain(push: Push, calls: Call[], maxRuns = 20): Promise<number[]> {
	const sizes: number[] = [];
	for (let i = 0; i < maxRuns; i++) {
		const before = calls.length;
		await sendNextBatch(env.DB, NOW, push);
		const size = calls.length - before;
		if (size === 0) break;
		sizes.push(size);
	}
	return sizes;
}

describe('createNotification', () => {
	it('寫出一列：種類、開始時間、三個版本、計數、進度與未完成', async () => {
		await addSub('https://push.example/a');
		await addSub('https://push.example/b', { group: 'men' });
		await createNotification(env.DB, 'results', { men: content('男'), women: null, both: content('合') }, NOW);
		const rows = await notificationRows();
		expect(rows).toHaveLength(1);
		expect(rows[0]).toMatchObject({
			kind: 'results',
			started_at: '2026-11-08T02:00:00.000Z',
			content_men: '{"title":"男標題","body":"男內文","navigate":"https://vstat.tw/男/#d-2026-11-08"}',
			content_women: null,
			content_both: '{"title":"合標題","body":"合內文","navigate":"https://vstat.tw/合/#d-2026-11-08"}',
			target_count: 2,
			subscription_count: 2,
			next_offset: 0,
			finished_at: null
		});
	});

	it('both 是 null 時不寫任何列', async () => {
		await addSub('https://push.example/a');
		await createNotification(env.DB, 'today', { men: null, women: null, both: null }, NOW);
		expect(await notificationRows()).toHaveLength(0);
	});

	it('target_count 依種類只算想收的', async () => {
		await addSub('https://push.example/a', { results: 1, today: 0 });
		await addSub('https://push.example/b', { results: 0, today: 1 });
		await addSub('https://push.example/c', { results: 1, today: 1 });
		await addSub('https://push.example/d', { results: 0, today: 0 });
		await createNotification(env.DB, 'results', ALL, NOW);
		await createNotification(env.DB, 'today', ALL, NOW);
		const rows = await notificationRows();
		expect(rows.map((r) => [r.kind, r.target_count, r.subscription_count])).toEqual([
			['results', 2, 4],
			['today', 2, 4]
		]);
	});

	it('只有男子組內容時，選女子組的不算、選兩組的算', async () => {
		await addSub('https://push.example/m', { group: 'men' });
		await addSub('https://push.example/w', { group: 'women' });
		await addSub('https://push.example/b', { group: 'both' });
		await createNotification(env.DB, 'results', MEN_ONLY, NOW);
		expect((await notificationRows())[0].target_count).toBe(2);
	});
});

describe('sendNextBatch', () => {
	it('沒有還沒送完的通知時不呼叫 push', async () => {
		await addMany(3);
		const { push, calls } = fakePush();
		await sendNextBatch(env.DB, NOW, push);
		expect(await endpointsLeft()).toHaveLength(3);
		expect(await notificationRows()).toHaveLength(0);
		expect(calls).toHaveLength(0);
	});

	it('每批 15 筆，送完寫上 finished_at，之後什麼都不做；每個推播位址恰好一次', async () => {
		const endpoints = await addMany(40);
		await createNotification(env.DB, 'results', ALL, NOW);
		const { push, calls } = fakePush();
		const sizes: number[] = [];
		for (let i = 0; i < 3; i++) {
			const before = calls.length;
			await sendNextBatch(env.DB, NOW, push);
			sizes.push(calls.length - before);
		}
		expect(sizes).toEqual([15, 15, 10]);
		expect((await notificationRows())[0].finished_at).toBeNull();
		await sendNextBatch(env.DB, NOW, push);
		expect((await notificationRows())[0].finished_at).toBe('2026-11-08T02:00:00.000Z');
		expect(calls).toHaveLength(40);
		await sendNextBatch(env.DB, NOW, push);
		expect(calls).toHaveLength(40);
		expect([...countBy(calls).keys()].sort()).toEqual([...endpoints].sort());
		expect([...countBy(calls).values()].every((n) => n === 1)).toBe(true);
	});

	it('依組別粒度發對應版本，沒有內容或不想收的不發', async () => {
		await addSub('https://push.example/m', { group: 'men' });
		await addSub('https://push.example/w', { group: 'women' });
		await addSub('https://push.example/b', { group: 'both' });
		await addSub('https://push.example/no-results', { results: 0, today: 1 });
		const { push, calls } = fakePush();
		await createNotification(env.DB, 'results', ALL, NOW);
		await drain(push, calls);
		const titles = Object.fromEntries(calls.map((c) => [c.endpoint, c.payload.notification.title]));
		expect(titles).toEqual({
			'https://push.example/m': '男標題',
			'https://push.example/w': '女標題',
			'https://push.example/b': '合標題'
		});
	});

	it('只有男子組內容時，選女子組的收不到，選兩組的收到男子組版', async () => {
		await addSub('https://push.example/m', { group: 'men' });
		await addSub('https://push.example/w', { group: 'women' });
		await addSub('https://push.example/b', { group: 'both' });
		const { push, calls } = fakePush();
		await createNotification(env.DB, 'today', MEN_ONLY, NOW);
		await drain(push, calls);
		expect(calls.map((c) => [c.endpoint, c.payload.notification])).toEqual([
			['https://push.example/m', content('男')],
			['https://push.example/b', content('男')]
		]);
	});

	it('賽果摘要只給 want_results，今日賽事只給 want_today', async () => {
		await addSub('https://push.example/r', { results: 1, today: 0 });
		await addSub('https://push.example/t', { results: 0, today: 1 });
		const first = fakePush();
		await createNotification(env.DB, 'results', ALL, NOW);
		await drain(first.push, first.calls);
		expect(first.calls.map((c) => c.endpoint)).toEqual(['https://push.example/r']);
		const second = fakePush();
		await createNotification(env.DB, 'today', ALL, NOW);
		await drain(second.push, second.calls);
		expect(second.calls.map((c) => c.endpoint)).toEqual(['https://push.example/t']);
	});

	it('payload 剛好是 web_push 8030 加 notification 三欄', async () => {
		await addSub('https://push.example/b');
		const seen: string[] = [];
		const push: Push = async (_target, payload) => {
			seen.push(payload);
			return 201;
		};
		await createNotification(env.DB, 'results', ALL, NOW);
		await sendNextBatch(env.DB, NOW, push);
		expect(seen).toHaveLength(1);
		expect(JSON.parse(seen[0])).toStrictEqual({
			web_push: 8030,
			notification: { title: '合標題', body: '合內文', navigate: 'https://vstat.tw/合/#d-2026-11-08' }
		});
	});

	it('404 與 410 刪訂閱並計入 deleted_count；201 計入成功；其他狀態與 reject 不刪不重試不計', async () => {
		const [ok, gone404, gone410, limited, broken, rejected] = ['ok', 'g404', 'g410', 'l429', 'e500', 'rej'].map(
			(n) => `https://push.example/${n}`
		);
		for (const e of [ok, gone404, gone410, limited, broken, rejected]) await addSub(e);
		const { push, calls } = fakePush({ [gone404]: 404, [gone410]: 410, [limited]: 429, [broken]: 500, [rejected]: 'reject' });
		await createNotification(env.DB, 'results', ALL, NOW);
		await drain(push, calls);
		await sendNextBatch(env.DB, NOW, push);
		expect(await endpointsLeft()).toEqual([ok, limited, broken, rejected]);
		expect([...countBy(calls).values()]).toEqual([1, 1, 1, 1, 1, 1]);
		expect(await notificationRows()).toMatchObject([{ success_count: 1, deleted_count: 2 }]);
	});

	it('認領沒成功就不送：並行的兩個呼叫沒有人收到兩次', async () => {
		await addMany(40);
		await createNotification(env.DB, 'results', ALL, NOW);
		const { push, calls } = fakePush();
		for (let i = 0; i < 4; i++) {
			await Promise.all([sendNextBatch(env.DB, NOW, push), sendNextBatch(env.DB, NOW, push)]);
		}
		expect(calls.length).toBeGreaterThan(0);
		expect([...countBy(calls).values()].every((n) => n === 1)).toBe(true);
	});

	it('送到一半訂閱增減，沒被刪的每筆恰好收到一次', async () => {
		const endpoints = await addMany(40);
		await createNotification(env.DB, 'results', ALL, NOW);
		const { push, calls } = fakePush();
		await sendNextBatch(env.DB, NOW, push);
		expect(calls).toHaveLength(15);
		await env.DB.prepare('DELETE FROM subscriptions WHERE endpoint = ?').bind('https://push.example/p20').run();
		await upsertSubscription(
			env.DB,
			{ endpoint: 'https://push.example/late', p256dh: 'B' + 'x'.repeat(86), auth: 'y'.repeat(22), wantResults: true, wantToday: false, group: 'both' },
			NOW
		);
		await drain(push, calls);
		await sendNextBatch(env.DB, NOW, push);
		const counts = countBy(calls);
		for (const e of endpoints.filter((e) => e !== 'https://push.example/p20')) expect(counts.get(e), e).toBe(1);
		expect(counts.has('https://push.example/p20')).toBe(false);
		expect(counts.get('https://push.example/late')).toBe(1);
		expect(calls).toHaveLength(40);
	});

	it('兩則相撞：今日賽事先送完，賽果摘要從停下處接著送', async () => {
		const endpoints = await addMany(40);
		await createNotification(env.DB, 'results', ALL, NOW);
		const { push, calls } = fakePush();
		await sendNextBatch(env.DB, NOW, push);
		await createNotification(env.DB, 'today', { men: null, women: null, both: content('今') }, NOW);
		const titles: string[] = [];
		for (let i = 0; i < 8; i++) {
			const before = calls.length;
			await sendNextBatch(env.DB, NOW, push);
			titles.push(calls.length === before ? '-' : calls[before].payload.notification.title);
		}
		expect(titles).toEqual(['今標題', '今標題', '今標題', '-', '合標題', '合標題', '-', '-']);
		const rows = await notificationRows();
		expect(rows.map((r) => [r.kind, r.finished_at !== null])).toEqual([
			['results', true],
			['today', true]
		]);
		for (const title of ['合標題', '今標題']) {
			const got = calls.filter((c) => c.payload.notification.title === title);
			expect(got).toHaveLength(40);
			expect(new Set(got.map((c) => c.endpoint))).toEqual(new Set(endpoints));
		}
	});

	it('送完那一列就是發送紀錄，數字與實際相符', async () => {
		const endpoints = await addMany(5);
		const { push, calls } = fakePush({ [endpoints[0]]: 410, [endpoints[1]]: 500 });
		await addSub('https://push.example/not-target', { results: 0 });
		await createNotification(env.DB, 'results', { men: content('男'), women: null, both: content('合') }, NOW);
		await drain(push, calls);
		await sendNextBatch(env.DB, NOW, push);
		const [row] = await notificationRows();
		expect(row).toMatchObject({
			kind: 'results',
			finished_at: '2026-11-08T02:00:00.000Z',
			target_count: 5,
			success_count: 3,
			deleted_count: 1,
			subscription_count: 6,
			content_men: JSON.stringify(content('男')),
			content_women: null,
			content_both: JSON.stringify(content('合'))
		});
	});
});

describe('跨批累計', () => {
	it('兩批各有成功與被刪的，紀錄的計數是兩批合計', async () => {
		const endpoints = await addMany(20);
		const gone = [2, 6, 15, 17].map((i) => endpoints[i]);
		const { push, calls } = fakePush({ [endpoints[2]]: 404, [endpoints[6]]: 410, [endpoints[15]]: 410, [endpoints[17]]: 404 });
		await createNotification(env.DB, 'results', ALL, NOW);
		expect(await drain(push, calls)).toEqual([15, 5]);
		await sendNextBatch(env.DB, NOW, push);
		expect(await notificationRows()).toMatchObject([{ success_count: 16, deleted_count: 4, finished_at: '2026-11-08T02:00:00.000Z' }]);
		expect(await endpointsLeft()).toEqual(endpoints.filter((e) => !gone.includes(e)));
	});
});

describe('claimBatch', () => {
	async function seed() {
		await env.DB.prepare(
			`INSERT INTO notifications (kind, started_at, next_offset, content_both) VALUES ('results', 't', 7, '{}')`
		).run();
		return (await notificationRows())[0].id as number;
	}

	it('from 與現在進度相同：回 true 且進度變成 to', async () => {
		const id = await seed();
		expect(await claimBatch(env.DB, id, 7, 22)).toBe(true);
		expect((await notificationRows())[0].next_offset).toBe(22);
	});

	it('from 與現在進度不同：回 false 且進度不變', async () => {
		const id = await seed();
		expect(await claimBatch(env.DB, id, 3, 18)).toBe(false);
		expect((await notificationRows())[0].next_offset).toBe(7);
	});
});
