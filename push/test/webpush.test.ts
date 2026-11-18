import { describe, expect, it } from 'vitest';
import { createPush, encryptPayload } from '../src/webpush';
import { decrypt, fromB64u, generateReceiver, parseHeader, toB64u, type Receiver } from './decrypt';

const enc = new TextEncoder();
const dec = new TextDecoder();

// RFC 8291 附錄 A 的測試向量（base64url 原文，去掉原文排版的空白）
const RFC = {
	plaintext: 'When I grow up, I want to be a watermelon',
	asPublic: 'BP4z9KsN6nGRTbVYI_c7VJSPQTBtkgcy27mlmlMoZIIgDll6e3vCYLocInmYWAmS6TlzAC8wEqKK6PBru3jl7A8',
	uaPublic: 'BCVxsr7N_eNgVRqvHtD0zTZsEc6-VV-JvLexhqUzORcxaOzi6-AYWXvTBHm4bjyPjs7Vd8pZGH6SRpkNtoIAiw4',
	uaPrivate: 'q1dXpw3UpT5VOmu_cf_v6ih07Aems3njxI-JWgLcM94',
	salt: 'DGv6ra1nlYgDCS1FRnbzlw',
	auth: 'BTBZMqHH6r4Tts7J_aSIgg',
	header: 'DGv6ra1nlYgDCS1FRnbzlwAAEABBBP4z9KsN6nGRTbVYI_c7VJSPQTBtkgcy27mlmlMoZIIgDll6e3vCYLocInmYWAmS6TlzAC8wEqKK6PBru3jl7A8',
	ciphertext: '8pfeW0KbunFT06SuDKoJH9Ql87S1QUrdirN6GcG7sFz1y1sqLgVi1VhjVkHsUoEsbI_0LpXMuGvnzQ'
};

function concat(a: Uint8Array, b: Uint8Array): Uint8Array {
	const out = new Uint8Array(a.length + b.length);
	out.set(a);
	out.set(b, a.length);
	return out;
}

describe('測試端解密器', () => {
	it('解得開 RFC 8291 附錄 A 的密文，得到 RFC 寫的明文', async () => {
		const body = concat(fromB64u(RFC.header), fromB64u(RFC.ciphertext));
		const receiver: Receiver = { publicKey: RFC.uaPublic, privateKey: RFC.uaPrivate, auth: RFC.auth };
		expect(dec.decode(await decrypt(receiver, body))).toBe(RFC.plaintext);
	});
});

describe('encryptPayload', () => {
	it('加密後用接收端的私鑰解得回原文', async () => {
		const receiver = await generateReceiver();
		const body = await encryptPayload({ p256dh: receiver.publicKey, auth: receiver.auth }, enc.encode('賽果摘要：台電 3:1 長榮'));
		expect(dec.decode(await decrypt(receiver, body))).toBe('賽果摘要：台電 3:1 長榮');
	});

	it('本文開頭是 16 bytes salt、record size 4096、65 bytes 的暫時公鑰', async () => {
		const receiver = await generateReceiver();
		const body = await encryptPayload({ p256dh: receiver.publicKey, auth: receiver.auth }, enc.encode('x'));
		const header = parseHeader(body);
		expect(header.salt.length).toBe(16);
		expect(header.recordSize).toBe(4096);
		expect(header.keyId.length).toBe(65);
		expect(header.keyId[0]).toBe(0x04);
		// 單一 record：16 + 4 + 1 + 65 的標頭，加上 1 byte 明文、1 byte 分隔位元組、16 bytes GCM tag
		expect(body.length).toBe(86 + 1 + 1 + 16);
	});

	it('同一段明文加密兩次，salt、暫時公鑰與密文都不同', async () => {
		const receiver = await generateReceiver();
		const target = { p256dh: receiver.publicKey, auth: receiver.auth };
		const a = await encryptPayload(target, enc.encode('同一段明文'));
		const b = await encryptPayload(target, enc.encode('同一段明文'));
		expect(toB64u(a)).not.toBe(toB64u(b));
		expect(toB64u(parseHeader(a).salt)).not.toBe(toB64u(parseHeader(b).salt));
		expect(toB64u(parseHeader(a).keyId)).not.toBe(toB64u(parseHeader(b).keyId));
	});
});

interface Call {
	url: string;
	init: RequestInit;
	headers: Headers;
	body: Uint8Array;
}

function fakeFetcher(status: number | Error = 201) {
	const calls: Call[] = [];
	const fetcher = (async (input: RequestInfo | URL, init?: RequestInit) => {
		calls.push({ url: String(input), init: init!, headers: new Headers(init!.headers), body: init!.body as Uint8Array });
		if (status instanceof Error) throw status;
		return new Response(null, { status });
	}) as typeof fetch;
	return { calls, fetcher };
}

async function vapidKeys() {
	const pair = (await crypto.subtle.generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, true, ['sign', 'verify'])) as CryptoKeyPair;
	const jwk = (await crypto.subtle.exportKey('jwk', pair.privateKey)) as JsonWebKey;
	const raw = new Uint8Array((await crypto.subtle.exportKey('raw', pair.publicKey)) as ArrayBuffer);
	return { publicKey: toB64u(raw), privateKey: jwk.d!, verifyKey: pair.publicKey };
}

async function target(endpoint: string, receiver: Receiver) {
	return { endpoint, p256dh: receiver.publicKey, auth: receiver.auth };
}

const NOW = new Date('2026-11-08T02:00:00Z');
const ENDPOINT = 'https://fcm.googleapis.com/fcm/send/abc123';

function jwtOf(call: Call): string {
	const auth = call.headers.get('Authorization')!;
	return auth.match(/^vapid t=([^,]+), k=/)![1];
}

describe('createPush', () => {
	it('POST 到推播位址，標頭照規格，不送 Urgency 與 Topic', async () => {
		const vapid = await vapidKeys();
		const receiver = await generateReceiver();
		const { calls, fetcher } = fakeFetcher();
		const push = createPush(vapid, NOW, fetcher);
		await push(await target(ENDPOINT, receiver), '{"title":"t"}');
		expect(calls.length).toBe(1);
		const call = calls[0];
		expect(call.url).toBe(ENDPOINT);
		expect(call.init.method).toBe('POST');
		expect(call.headers.get('Authorization')).toMatch(new RegExp(`^vapid t=[\\w-]+\\.[\\w-]+\\.[\\w-]+, k=${vapid.publicKey}$`));
		expect(call.headers.get('Content-Encoding')).toBe('aes128gcm');
		expect(call.headers.get('Content-Type')).toBe('application/octet-stream');
		expect(call.headers.get('TTL')).toBe('43200');
		expect(call.headers.has('Urgency')).toBe(false);
		expect(call.headers.has('Topic')).toBe(false);
	});

	it('本文解得回 payload', async () => {
		const vapid = await vapidKeys();
		const receiver = await generateReceiver();
		const { calls, fetcher } = fakeFetcher();
		await createPush(vapid, NOW, fetcher)(await target(ENDPOINT, receiver), '{"title":"賽果摘要"}');
		expect(dec.decode(await decrypt(receiver, calls[0].body))).toBe('{"title":"賽果摘要"}');
	});

	it('JWT 用 VAPID 公鑰驗章通過，header 與 claims 正確', async () => {
		const vapid = await vapidKeys();
		const receiver = await generateReceiver();
		const { calls, fetcher } = fakeFetcher();
		await createPush(vapid, NOW, fetcher)(await target(ENDPOINT, receiver), 'x');
		const [head, claims, sig] = jwtOf(calls[0]).split('.');
		const ok = await crypto.subtle.verify({ name: 'ECDSA', hash: 'SHA-256' }, vapid.verifyKey, fromB64u(sig), enc.encode(`${head}.${claims}`));
		expect(ok).toBe(true);
		expect(JSON.parse(dec.decode(fromB64u(head)))).toEqual({ typ: 'JWT', alg: 'ES256' });
		expect(JSON.parse(dec.decode(fromB64u(claims)))).toEqual({
			aud: 'https://fcm.googleapis.com',
			sub: 'https://vstat.tw',
			// 2026-11-08T14:00:00Z：NOW 的 12 小時後
			exp: 1794146400
		});
	});

	it('同一個 origin 的兩個推播位址拿到同一張 JWT，不同 origin 的 aud 不同', async () => {
		const vapid = await vapidKeys();
		const receiver = await generateReceiver();
		const { calls, fetcher } = fakeFetcher();
		const push = createPush(vapid, NOW, fetcher);
		await push(await target('https://fcm.googleapis.com/fcm/send/one', receiver), 'x');
		await push(await target('https://fcm.googleapis.com/fcm/send/two', receiver), 'x');
		await push(await target('https://updates.push.services.mozilla.com/wpush/v2/abc', receiver), 'x');
		expect(calls.length).toBe(3);
		expect(jwtOf(calls[0])).toBe(jwtOf(calls[1]));
		expect(jwtOf(calls[2])).not.toBe(jwtOf(calls[0]));
		const aud = (call: Call) => JSON.parse(dec.decode(fromB64u(jwtOf(call).split('.')[1]))).aud;
		expect(aud(calls[0])).toBe('https://fcm.googleapis.com');
		expect(aud(calls[2])).toBe('https://updates.push.services.mozilla.com');
	});

	it.each([201, 404, 410, 429])('狀態碼 %i 原樣回傳', async (status) => {
		const vapid = await vapidKeys();
		const receiver = await generateReceiver();
		const { calls, fetcher } = fakeFetcher(status);
		const result = await createPush(vapid, NOW, fetcher)(await target(ENDPOINT, receiver), 'x');
		expect(calls.length).toBe(1);
		expect(result).toBe(status);
	});

	it('fetcher 丟錯時 reject', async () => {
		const vapid = await vapidKeys();
		const receiver = await generateReceiver();
		const { calls, fetcher } = fakeFetcher(new TypeError('network down'));
		await expect(createPush(vapid, NOW, fetcher)(await target(ENDPOINT, receiver), 'x')).rejects.toThrow('network down');
		expect(calls.length).toBe(1);
	});

	it('把 AbortSignal 傳給 fetcher', async () => {
		const vapid = await vapidKeys();
		const receiver = await generateReceiver();
		const { calls, fetcher } = fakeFetcher();
		await createPush(vapid, NOW, fetcher)(await target(ENDPOINT, receiver), 'x');
		expect(calls[0].init.signal).toBeInstanceOf(AbortSignal);
		expect(calls[0].init.signal!.aborted).toBe(false);
	});
});
