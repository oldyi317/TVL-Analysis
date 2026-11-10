import { sendPush, encryptPayload, vapidJwt } from './webpush.js';

const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { 'content-type': 'application/json; charset=utf-8' } });
const TTL = { expirationTtl: 86400 };

async function subKey(endpoint) {
	const h = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(endpoint));
	return 'sub:' + [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function buildPayload(mode, id, sentAt, origin) {
	const path = `/matches?from=push&mode=${mode}&id=${id}&sentAt=${sentAt}`;
	const title = '推播測試';
	const body = `${mode} 測試推播`;
	if (mode === 'declarative') {
		// 只用 WebKit 文章的標準欄位；id、sentAt 放在 navigate 的 query。
		return { web_push: 8030, notification: { title, body, navigate: origin + path } };
	}
	return { id, sentAt, title, body, url: path };
}

// 實際發送並寫紀錄。trigger: now|delay20|cron
async function deliver(env, { id, endpoint, mode, when, trigger, origin, drift }) {
	const subRaw = await env.PROTO_KV.get(await subKey(endpoint));
	const rec = { id, mode, when, trigger, drift: drift ?? null, endpoint: new URL(endpoint).host };
	if (!subRaw) {
		rec.error = 'subscription not found';
	} else {
		rec.sentAt = Date.now();
		const t0 = Date.now();
		try {
			const payload = buildPayload(mode, id, rec.sentAt, origin);
			console.log('payload', JSON.stringify(payload));
			const res = await sendPush({
				subscription: JSON.parse(subRaw),
				payload,
				vapid: { publicKey: env.VAPID_PUBLIC, privateKey: env.VAPID_PRIVATE },
				subject: env.VAPID_SUB || origin,
			});
			rec.fetchMs = Date.now() - t0;
			rec.status = res.status;
			if (!res.ok) rec.responseText = (await res.text()).slice(0, 300);
		} catch (e) {
			rec.fetchMs = Date.now() - t0;
			rec.error = String(e && e.stack ? e.stack : e).slice(0, 300);
		}
	}
	console.log('deliver', JSON.stringify(rec));
	await env.PROTO_KV.put('log:' + id, JSON.stringify(rec), TTL);
	return rec;
}

async function listAll(kv, prefix) {
	const out = [];
	let cursor;
	do {
		const r = await kv.list({ prefix, cursor });
		out.push(...r.keys);
		cursor = r.list_complete ? undefined : r.cursor;
	} while (cursor);
	return out;
}

async function handleApi(request, env, ctx, url) {
	const p = url.pathname;
	if (p === '/api/config' && request.method === 'GET') return json({ vapidPublicKey: env.VAPID_PUBLIC });

	// 只量 CPU：加密 n 則（RFC 8291 範例金鑰）＋簽 JWT（once 或 each），不發送。CPU 時間看 wrangler tail。
	if (p === '/api/bench' && request.method === 'GET') {
		const n = Math.min(Number(url.searchParams.get('n')) || 1, 100);
		const each = url.searchParams.get('jwt') === 'each';
		const plaintext = new TextEncoder().encode(JSON.stringify(buildPayload('classic', 'bench', Date.now(), url.origin)));
		const sign = () => vapidJwt({ audience: 'https://web.push.apple.com', subject: url.origin, publicKey: env.VAPID_PUBLIC, privateKey: env.VAPID_PRIVATE });
		if (!each) await sign();
		for (let i = 0; i < n; i++) {
			await encryptPayload({ p256dh: 'BCVxsr7N_eNgVRqvHtD0zTZsEc6-VV-JvLexhqUzORcxaOzi6-AYWXvTBHm4bjyPjs7Vd8pZGH6SRpkNtoIAiw4', auth: 'BTBZMqHH6r4Tts7J_aSIgg', plaintext });
			if (each) await sign();
		}
		return json({ n, jwt: each ? 'each' : 'once' });
	}

	if (p === '/api/subscribe' && request.method === 'POST') {
		const sub = await request.json();
		if (!sub?.endpoint || !sub?.keys?.p256dh || !sub?.keys?.auth) return json({ error: 'bad subscription' }, 400);
		await env.PROTO_KV.put(await subKey(sub.endpoint), JSON.stringify(sub));
		return json({ ok: true, host: new URL(sub.endpoint).host });
	}

	if (p === '/api/send' && request.method === 'POST') {
		const { endpoint, mode, when } = await request.json();
		if (!['classic', 'declarative'].includes(mode) || !['now', 'delay20', 'cron'].includes(when)) return json({ error: 'bad mode/when' }, 400);
		if (!endpoint || !(await env.PROTO_KV.get(await subKey(endpoint)))) return json({ error: 'subscription not found' }, 404);
		const id = `${Date.now()}-${crypto.randomUUID().slice(0, 6)}`;
		const origin = url.origin;
		if (when === 'now') return json(await deliver(env, { id, endpoint, mode, when, trigger: 'now', origin }));
		if (when === 'delay20') {
			ctx.waitUntil(new Promise((r) => setTimeout(r, 20000)).then(() => deliver(env, { id, endpoint, mode, when, trigger: 'delay20', origin })));
			return json({ id, queued: 'delay20' });
		}
		await env.PROTO_KV.put('pending:' + id, JSON.stringify({ id, endpoint, mode, when, origin }), TTL);
		return json({ id, queued: 'cron' });
	}

	if (p === '/api/receipt' && request.method === 'POST') {
		const { id, event, at, ...extra } = await request.json();
		if (!id || !['push', 'click', 'opened'].includes(event)) return json({ error: 'bad receipt' }, 400);
		await env.PROTO_KV.put(`rc:${id}:${event}`, JSON.stringify({ ...extra, at, recvAt: Date.now() }), TTL);
		return json({ ok: true });
	}

	if (p === '/api/log' && request.method === 'GET') {
		const logKeys = (await listAll(env.PROTO_KV, 'log:')).map((k) => k.name).sort().slice(-30);
		const rcKeys = await listAll(env.PROTO_KV, 'rc:');
		const rcs = {};
		await Promise.all(
			rcKeys.map(async (k) => {
				const [, id, ev] = k.name.split(':');
				(rcs[id] ||= {})[ev] = JSON.parse(await env.PROTO_KV.get(k.name));
			}),
		);
		const logs = (await Promise.all(logKeys.map((k) => env.PROTO_KV.get(k)))).filter(Boolean).map((r) => JSON.parse(r));
		for (const r of logs) {
			r.receipts = rcs[r.id] || {};
			const push = r.receipts.push;
			// 前者用裝置時鐘（含時鐘誤差）；後者用伺服器收到回報的時間（含回報延遲）
			r.sendToSwMs = push && r.sentAt ? push.at - r.sentAt : null;
			r.sendToSwServerMs = push && r.sentAt ? push.recvAt - r.sentAt : null;
		}
		logs.reverse();
		const driftKeys = (await listAll(env.PROTO_KV, 'drift:')).map((k) => k.name).sort().slice(-20).reverse();
		const drifts = (await Promise.all(driftKeys.map((k) => env.PROTO_KV.get(k)))).filter(Boolean).map((r) => JSON.parse(r));
		return json({ logs, drifts });
	}

	return json({ error: 'not found' }, 404);
}

export default {
	async fetch(request, env, ctx) {
		const url = new URL(request.url);
		try {
			return await handleApi(request, env, ctx, url);
		} catch (e) {
			console.log('api error', String(e && e.stack ? e.stack : e));
			return json({ error: String(e && e.message ? e.message : e) }, 500);
		}
	},

	async scheduled(controller, env, ctx) {
		const started = Date.now();
		const drift = started - controller.scheduledTime;
		const minute = new Date(controller.scheduledTime).getUTCMinutes();
		console.log(`cron scheduledTime=${new Date(controller.scheduledTime).toISOString()} started=${new Date(started).toISOString()} driftMs=${drift}`);
		const pending = await listAll(env.PROTO_KV, 'pending:');
		for (const k of pending) {
			const raw = await env.PROTO_KV.get(k.name);
			await env.PROTO_KV.delete(k.name);
			if (!raw) continue;
			const job = JSON.parse(raw);
			await deliver(env, { ...job, trigger: 'cron', drift });
		}
		if (pending.length || minute % 10 === 0) {
			await env.PROTO_KV.put(`drift:${controller.scheduledTime}`, JSON.stringify({ scheduledTime: controller.scheduledTime, startedAt: started, driftMs: drift, sent: pending.length }), TTL);
		}
	},
};
