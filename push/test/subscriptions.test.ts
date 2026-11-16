import { env, exports } from 'cloudflare:workers';
import { describe, expect, it } from 'vitest';
import worker from '../src/index';

const ORIGIN = 'https://vstat.tw';
const ENDPOINT = 'https://fcm.googleapis.com/fcm/send/abc123';
const P256DH = 'B' + 'x'.repeat(86);
const AUTH = 'y'.repeat(22);

function subscription(overrides: Record<string, unknown> = {}) {
	return { endpoint: ENDPOINT, keys: { p256dh: P256DH, auth: AUTH }, results: true, today: false, group: 'both', ...overrides };
}

function post(path: string, body: unknown, headers: Record<string, string> = {}) {
	return exports.default.fetch(`https://push.vstat.tw${path}`, {
		method: 'POST',
		headers: { Origin: ORIGIN, 'Content-Type': 'application/json', ...headers },
		body: typeof body === 'string' ? body : JSON.stringify(body)
	});
}

async function rows() {
	const { results } = await env.DB.prepare(
		'SELECT endpoint, p256dh, auth, want_results, want_today, group_pref, created_at, updated_at FROM subscriptions ORDER BY id'
	).all();
	return results;
}

async function fillTo(count: number) {
	await env.DB.prepare(
		`WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < ?1)
		INSERT INTO subscriptions (endpoint, p256dh, auth, want_results, want_today, group_pref, created_at, updated_at)
		SELECT 'https://fcm.googleapis.com/fcm/send/filler' || i, ?2, ?3, 1, 1, 'both', 't', 't' FROM n`
	)
		.bind(count, P256DH, AUTH)
		.run();
}

describe('登記入口', () => {
	it('新的推播位址新增一筆，記下偏好', async () => {
		const res = await post('/subscribe', subscription({ results: true, today: false, group: 'women' }));

		expect(res.status).toBe(200);
		expect(await res.json()).toEqual({ ok: true });
		expect(await rows()).toMatchObject([
			{ endpoint: ENDPOINT, p256dh: P256DH, auth: AUTH, want_results: 1, want_today: 0, group_pref: 'women' }
		]);
	});

	it('已存在的推播位址覆寫偏好與金鑰，不新增第二筆', async () => {
		await post('/subscribe', subscription({ results: true, today: false, group: 'women' }));
		const [before] = await rows();
		const newAuth = 'z'.repeat(22);

		const res = await post('/subscribe', subscription({ results: false, today: true, group: 'men', keys: { p256dh: P256DH, auth: newAuth } }));

		expect(res.status).toBe(200);
		const after = await rows();
		expect(after).toHaveLength(1);
		expect(after[0]).toMatchObject({ auth: newAuth, want_results: 0, want_today: 1, group_pref: 'men', created_at: before.created_at });
	});

	it('瀏覽器 toJSON() 多帶的 expirationTime 與 base64url 填充都收', async () => {
		const res = await post('/subscribe', subscription({ expirationTime: null, keys: { p256dh: P256DH + '=', auth: AUTH + '==' } }));

		expect(res.status).toBe(200);
		expect(await rows()).toMatchObject([{ p256dh: P256DH, auth: AUTH }]);
	});
});

describe('同一個推播位址的不同寫法算同一筆', () => {
	const variant = 'https://FCM.googleapis.com:443/fcm/send/abc123#again';

	it('登記時存正規化後的推播位址，變體覆寫原本那筆', async () => {
		await post('/subscribe', subscription({ group: 'men' }));

		const res = await post('/subscribe', subscription({ endpoint: variant, group: 'women' }));

		expect(res.status).toBe(200);
		expect(await rows()).toMatchObject([{ endpoint: ENDPOINT, group_pref: 'women' }]);
	});

	it('退訂時用變體也刪得到', async () => {
		await post('/subscribe', subscription());

		await post('/unsubscribe', { endpoint: variant });

		expect(await rows()).toEqual([]);
	});
});

describe('退訂入口', () => {
	it('給推播位址就刪那一筆', async () => {
		await post('/subscribe', subscription());
		await post('/subscribe', subscription({ endpoint: 'https://updates.push.services.mozilla.com/wpush/v2/other' }));

		const res = await post('/unsubscribe', { endpoint: ENDPOINT });

		expect(res.status).toBe(200);
		expect(await res.json()).toEqual({ ok: true });
		expect((await rows()).map((r) => r.endpoint)).toEqual(['https://updates.push.services.mozilla.com/wpush/v2/other']);
	});

	it('推播位址不存在也回成功', async () => {
		const res = await post('/unsubscribe', { endpoint: ENDPOINT });

		expect(res.status).toBe(200);
		expect(await rows()).toEqual([]);
	});
});

describe('只收已知推播服務網域', () => {
	it.each([
		'https://web.push.apple.com/QGuQyavXutnMH6YPkK6yqIk',
		'https://fcm.googleapis.com/fcm/send/dSa1-xyz',
		'https://fcm.googleapis.com/wp/dSa1-xyz',
		'https://updates.push.services.mozilla.com/wpush/v2/gAAAAAB',
		'https://wns2-par02p.notify.windows.com/w/?token=BQYAAAD'
	])('收 %s', async (endpoint) => {
		const res = await post('/subscribe', subscription({ endpoint }));

		expect(res.status).toBe(200);
		expect(await rows()).toHaveLength(1);
	});

	it.each([
		['不是 https', 'http://fcm.googleapis.com/fcm/send/abc'],
		['不認得的網域', 'https://push.example.com/abc'],
		['只是前綴相同', 'https://fcm.googleapis.com.evil.example/abc'],
		['後綴沒有以點隔開', 'https://evilnotify.windows.com/w/?token=abc'],
		['Apple 網域掛在別人底下', 'https://push.apple.com.evil.example/abc'],
		['帶帳號密碼', 'https://user:pass@fcm.googleapis.com/fcm/send/abc'],
		['指定連接埠', 'https://fcm.googleapis.com:8443/fcm/send/abc'],
		['不是網址', 'not a url']
	])('拒絕%s，且不寫入', async (_, endpoint) => {
		const res = await post('/subscribe', subscription({ endpoint }));

		expect(res.status).toBe(400);
		expect(await res.json()).toEqual({ error: 'endpoint_not_allowed' });
		expect(await rows()).toEqual([]);
	});
});

describe('推播訂閱上限 500 筆', () => {
	it('滿了拒收新的推播位址，不寫入', async () => {
		await fillTo(500);

		const res = await post('/subscribe', subscription());

		expect(res.status).toBe(503);
		expect(await res.json()).toEqual({ error: 'subscription_limit_reached' });
		expect(await rows()).toHaveLength(500);
		expect((await rows()).some((r) => r.endpoint === ENDPOINT)).toBe(false);
	});

	it('滿了之後，已登記的推播位址重新登記仍成功', async () => {
		await fillTo(499);
		await post('/subscribe', subscription({ group: 'men' }));

		const res = await post('/subscribe', subscription({ group: 'women' }));

		expect(res.status).toBe(200);
		const all = await rows();
		expect(all).toHaveLength(500);
		expect(all.find((r) => r.endpoint === ENDPOINT)).toMatchObject({ group_pref: 'women' });
	});

	it('第 500 筆還收', async () => {
		await fillTo(499);

		const res = await post('/subscribe', subscription());

		expect(res.status).toBe(200);
		expect(await rows()).toHaveLength(500);
	});
});

describe('只接受允許的網站 origin', () => {
	it('origin 不在允許清單就拒絕，不寫入', async () => {
		const res = await post('/subscribe', subscription(), { Origin: 'https://evil.example' });

		expect(res.status).toBe(403);
		expect(res.headers.get('Access-Control-Allow-Origin')).toBeNull();
		expect(await rows()).toEqual([]);
	});

	it('沒帶 Origin 一律拒絕，不寫入', async () => {
		const res = await exports.default.fetch('https://push.vstat.tw/subscribe', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(subscription())
		});

		expect(res.status).toBe(403);
		expect(await rows()).toEqual([]);
	});

	it('退訂入口同樣擋 origin', async () => {
		await post('/subscribe', subscription());

		const res = await post('/unsubscribe', { endpoint: ENDPOINT }, { Origin: 'https://evil.example' });

		expect(res.status).toBe(403);
		expect(await rows()).toHaveLength(1);
	});

	it('允許的 origin 取自設定', async () => {
		const preview = 'https://preview.example';
		const request = new Request<unknown, IncomingRequestCfProperties>('https://push.vstat.tw/subscribe', {
			method: 'POST',
			headers: { Origin: preview, 'Content-Type': 'application/json' },
			body: JSON.stringify(subscription())
		});

		const res = await worker.fetch(request, { ...env, ALLOWED_ORIGIN: preview });

		expect(res.status).toBe(200);
		expect(res.headers.get('Access-Control-Allow-Origin')).toBe(preview);
		expect(await rows()).toHaveLength(1);
	});

	it('成功的回應帶 CORS 標頭', async () => {
		const res = await post('/subscribe', subscription());

		expect(res.headers.get('Access-Control-Allow-Origin')).toBe(ORIGIN);
		expect(res.headers.get('Vary')).toBe('Origin');
	});

	it.each(['/subscribe', '/unsubscribe'])('%s 對允許的 origin 回應預檢', async (path) => {
		const res = await exports.default.fetch(`https://push.vstat.tw${path}`, {
			method: 'OPTIONS',
			headers: { Origin: ORIGIN, 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type' }
		});

		expect(res.status).toBe(204);
		expect(res.headers.get('Access-Control-Allow-Origin')).toBe(ORIGIN);
		expect(res.headers.get('Access-Control-Allow-Methods')).toBe('POST');
		expect(res.headers.get('Access-Control-Allow-Headers')).toBe('Content-Type');
	});

	it('不允許的 origin 預檢被拒，不帶 CORS 標頭', async () => {
		const res = await exports.default.fetch('https://push.vstat.tw/subscribe', {
			method: 'OPTIONS',
			headers: { Origin: 'https://evil.example', 'Access-Control-Request-Method': 'POST' }
		});

		expect(res.status).toBe(403);
		expect(res.headers.get('Access-Control-Allow-Origin')).toBeNull();
	});
});

describe('請求內容不合就拒絕且不寫入', () => {
	it.each([
		['不是 JSON', '{endpoint'],
		['是陣列', [subscription()]],
		['少了 keys', { ...subscription(), keys: undefined }],
		['p256dh 長度不對', subscription({ keys: { p256dh: P256DH.slice(1), auth: AUTH } })],
		['auth 有非 base64url 字元', subscription({ keys: { p256dh: P256DH, auth: 'y'.repeat(21) + '+' } })],
		['推播位址不是字串', subscription({ endpoint: 42 })],
		['推播位址超過 2048 字', subscription({ endpoint: 'https://fcm.googleapis.com/fcm/send/' + 'a'.repeat(2048) })],
		['偏好不是布林', subscription({ results: 'yes' })],
		['兩種都不收', subscription({ results: false, today: false })],
		['組別不認得', subscription({ group: 'mixed' })]
	])('登記：%s → 400', async (_, body) => {
		const res = await post('/subscribe', body);

		expect(res.status).toBe(400);
		expect(await res.json()).toEqual({ error: 'invalid_body' });
		expect(await rows()).toEqual([]);
	});

	it('請求超過 4096 bytes → 413，不寫入', async () => {
		const res = await post('/subscribe', subscription({ padding: 'p'.repeat(5000) }));

		expect(res.status).toBe(413);
		expect(await res.json()).toEqual({ error: 'body_too_large' });
		expect(await rows()).toEqual([]);
	});

	it.each([
		['不是 JSON', 'nope'],
		['少了推播位址', {}],
		['推播位址超長', { endpoint: 'https://fcm.googleapis.com/' + 'a'.repeat(2048) }]
	])('退訂：%s → 400，不刪', async (_, body) => {
		await post('/subscribe', subscription());

		const res = await post('/unsubscribe', body);

		expect(res.status).toBe(400);
		expect(await rows()).toHaveLength(1);
	});
});

describe('其他路徑與方法', () => {
	it('不認得的路徑回 404', async () => {
		const res = await post('/nope', subscription());

		expect(res.status).toBe(404);
	});

	it('入口不收 GET', async () => {
		const res = await exports.default.fetch('https://push.vstat.tw/subscribe', { headers: { Origin: ORIGIN } });

		expect(res.status).toBe(405);
		expect(res.headers.get('Allow')).toBe('POST, OPTIONS');
	});
});
