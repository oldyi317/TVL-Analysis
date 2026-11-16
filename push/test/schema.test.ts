import { env } from 'cloudflare:workers';
import { describe, expect, it } from 'vitest';

async function count(table: string) {
	return (await env.DB.prepare(`SELECT COUNT(*) AS n FROM ${table}`).first<number>('n')) ?? 0;
}

describe('已發過的賽事以 cup_id、組別、場次編號為唯一鍵', () => {
	const claim = (cupId: number, group: string, gameId: number) =>
		env.DB.prepare('INSERT OR IGNORE INTO sent_matches (cup_id, match_group, game_id, sent_at) VALUES (?, ?, ?, ?)')
			.bind(cupId, group, gameId, '2026-11-16T02:00:00Z')
			.run();

	it('同一場寫第二次沒有新增', async () => {
		expect((await claim(22, 'men', 338)).meta.changes).toBe(1);
		expect((await claim(22, 'men', 338)).meta.changes).toBe(0);
		expect(await count('sent_matches')).toBe(1);
	});

	it('場次編號相同但組別或賽季不同是不同的賽事', async () => {
		await claim(22, 'men', 338);
		expect((await claim(22, 'women', 338)).meta.changes).toBe(1);
		expect((await claim(23, 'men', 338)).meta.changes).toBe(1);
		expect(await count('sent_matches')).toBe(3);
	});
});

describe('今日賽事以台北日期為唯一鍵', () => {
	const claim = (date: string) =>
		env.DB.prepare('INSERT OR IGNORE INTO sent_today_dates (taipei_date, sent_at) VALUES (?, ?)')
			.bind(date, '2026-11-16T02:00:00Z')
			.run();

	it('同一天寫第二次沒有新增', async () => {
		expect((await claim('2026-11-16')).meta.changes).toBe(1);
		expect((await claim('2026-11-16')).meta.changes).toBe(0);
		expect((await claim('2026-11-17')).meta.changes).toBe(1);
		expect(await count('sent_today_dates')).toBe(2);
	});
});

describe('讀檔成功的記號只會有一列', () => {
	it('第一次寫入改到一列，之後都沒改到', async () => {
		const mark = () =>
			env.DB.prepare('INSERT OR IGNORE INTO first_feed_read (id, read_at) VALUES (1, ?)').bind('2026-11-16T02:00:00Z').run();

		expect((await mark()).meta.changes).toBe(1);
		expect((await mark()).meta.changes).toBe(0);
		expect(await count('first_feed_read')).toBe(1);
	});
});

describe('批次進度以條件式寫入認領', () => {
	it('進度還是 N 才推到 N+15，各則分開記', async () => {
		const insert = (kind: string) =>
			env.DB.prepare("INSERT INTO notifications (kind, started_at, content_both) VALUES (?, 't', '{}') RETURNING id")
				.bind(kind)
				.first<number>('id');
		const results = await insert('results');
		const today = await insert('today');
		const advance = (id: number | null, from: number) =>
			env.DB.prepare('UPDATE notifications SET next_offset = next_offset + 15 WHERE id = ? AND next_offset = ?').bind(id, from).run();

		expect((await advance(results, 0)).meta.changes).toBe(1);
		expect((await advance(results, 0)).meta.changes).toBe(0);
		expect((await advance(today, 0)).meta.changes).toBe(1);
		expect((await advance(results, 15)).meta.changes).toBe(1);
	});
});
