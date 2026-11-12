const json = (body, status = 200) =>
	new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

const pick = (r) => ({
	results: r.results,
	changes: r.meta.changes,
	rows_written: r.meta.rows_written,
	changed_db: r.meta.changed_db,
	served_by_region: r.meta.served_by_region,
	served_by_primary: r.meta.served_by_primary
});

const SETUP = [
	'CREATE TABLE IF NOT EXISTS progress (id INTEGER PRIMARY KEY, next_offset INTEGER NOT NULL)',
	'CREATE TABLE IF NOT EXISTS seq (id INTEGER PRIMARY KEY, value INTEGER NOT NULL)',
	'CREATE TABLE IF NOT EXISTS sent (key TEXT PRIMARY KEY)',
	'CREATE TABLE IF NOT EXISTS cron_state (id INTEGER PRIMARY KEY, n INTEGER NOT NULL)',
	'CREATE TABLE IF NOT EXISTS cron_log (id INTEGER PRIMARY KEY AUTOINCREMENT, scheduled_time INTEGER, started_at INTEGER, read_n INTEGER, changes INTEGER, slept INTEGER, finished_at INTEGER)',
	'INSERT OR IGNORE INTO cron_state (id, n) VALUES (1, 0)'
];

const RESET = [
	'DELETE FROM progress',
	'INSERT INTO progress (id, next_offset) VALUES (1, 0)',
	'DELETE FROM seq',
	'INSERT INTO seq (id, value) VALUES (1, 0)',
	'DELETE FROM sent'
];

const CLAIM = 'UPDATE progress SET next_offset = next_offset + 15 WHERE id = 1 AND next_offset = ?';

const routes = {
	async '/init'(db) {
		await db.batch(SETUP.map((sql) => db.prepare(sql)));
		return { ok: true };
	},
	async '/reset'(db) {
		await db.batch(RESET.map((sql) => db.prepare(sql)));
		return { ok: true };
	},
	async '/seq'(db, q) {
		const i = Number(q.get('i'));
		const read = await db.prepare('SELECT value FROM seq WHERE id = 1').all();
		const write = await db.prepare('UPDATE seq SET value = ? WHERE id = 1').bind(i).run();
		return { read: read.results[0].value, wrote: i, read_meta: pick(read), changes: write.meta.changes };
	},
	async '/claim'(db, q) {
		return pick(await db.prepare(CLAIM).bind(Number(q.get('from'))).run());
	},
	async '/claim-returning'(db, q) {
		return pick(await db.prepare(`${CLAIM} RETURNING next_offset`).bind(Number(q.get('from'))).all());
	},
	async '/claim-batch'(db, q) {
		const out = await db.batch([
			db.prepare(CLAIM).bind(Number(q.get('from'))),
			db.prepare('SELECT next_offset FROM progress WHERE id = 1')
		]);
		return out.map(pick);
	},
	async '/insert'(db, q) {
		return pick(await db.prepare('INSERT OR IGNORE INTO sent (key) VALUES (?)').bind(q.get('key')).run());
	},
	async '/insert-returning'(db, q) {
		return pick(
			await db
				.prepare('INSERT INTO sent (key) VALUES (?) ON CONFLICT DO NOTHING RETURNING key')
				.bind(q.get('key'))
				.all()
		);
	},
	async '/insert-strict'(db, q) {
		return pick(await db.prepare('INSERT INTO sent (key) VALUES (?)').bind(q.get('key')).run());
	},
	async '/progress'(db) {
		return pick(await db.prepare('SELECT next_offset FROM progress WHERE id = 1').all());
	},
	async '/cron-log'(db) {
		return pick(await db.prepare('SELECT * FROM cron_log ORDER BY id').all());
	}
};

export default {
	async fetch(request, env) {
		const url = new URL(request.url);
		if (url.searchParams.get('t') !== env.TOKEN) return json({ error: 'forbidden' }, 403);
		const route = routes[url.pathname];
		if (!route) return json({ error: 'not found' }, 404);
		try {
			return json({ colo: request.cf?.colo, data: await route(env.DB, url.searchParams) });
		} catch (e) {
			return json({ colo: request.cf?.colo, error: String(e), cause: String(e.cause ?? '') }, 500);
		}
	},

	async scheduled(controller, env) {
		const db = env.DB;
		const startedAt = Date.now();
		const read = await db.prepare('SELECT n FROM cron_state WHERE id = 1').all();
		const n = read.results[0].n;
		const claim = await db.prepare('UPDATE cron_state SET n = n + 1 WHERE id = 1 AND n = ?').bind(n).run();
		const slept = n % 3 === 0 ? 1 : 0;
		const log = await db
			.prepare('INSERT INTO cron_log (scheduled_time, started_at, read_n, changes, slept) VALUES (?, ?, ?, ?, ?)')
			.bind(controller.scheduledTime, startedAt, n, claim.meta.changes, slept)
			.run();
		if (slept) await scheduler.wait(75_000);
		await db.prepare('UPDATE cron_log SET finished_at = ? WHERE id = ?').bind(Date.now(), log.meta.last_row_id).run();
	}
};
