// 手寫 Web Push：VAPID (RFC 8292) + aes128gcm (RFC 8291 / 8188)，只用 WebCrypto。
const enc = new TextEncoder();

export function b64uEncode(bytes) {
	let s = '';
	for (const b of bytes) s += String.fromCharCode(b);
	return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export function b64uDecode(str) {
	const b64 = str.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (str.length % 4)) % 4);
	return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
}

function concat(...parts) {
	const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
	let o = 0;
	for (const p of parts) {
		out.set(p, o);
		o += p.length;
	}
	return out;
}

// raw 未壓縮公鑰（65 bytes）+ 可選私鑰 d（32 bytes）→ JWK
function toJwk(pubRaw, d) {
	const jwk = { kty: 'EC', crv: 'P-256', x: b64uEncode(pubRaw.slice(1, 33)), y: b64uEncode(pubRaw.slice(33, 65)) };
	if (d) jwk.d = b64uEncode(d);
	return jwk;
}

async function hkdf(salt, ikm, info, bits) {
	const key = await crypto.subtle.importKey('raw', ikm, 'HKDF', false, ['deriveBits']);
	return new Uint8Array(await crypto.subtle.deriveBits({ name: 'HKDF', hash: 'SHA-256', salt, info }, key, bits));
}

/**
 * 回傳完整 aes128gcm 請求本文（header + 密文）。
 * opts.salt、opts.senderKey = { publicRaw, d }（皆 Uint8Array）只為測試注入；不給就隨機產生。
 */
export async function encryptPayload({ p256dh, auth, plaintext, salt, senderKey }) {
	const uaPub = b64uDecode(p256dh);
	const authSecret = b64uDecode(auth);
	if (!salt) salt = crypto.getRandomValues(new Uint8Array(16));

	let asPriv, asPubRaw;
	if (senderKey) {
		asPubRaw = senderKey.publicRaw;
		asPriv = await crypto.subtle.importKey('jwk', toJwk(asPubRaw, senderKey.d), { name: 'ECDH', namedCurve: 'P-256' }, false, ['deriveBits']);
	} else {
		const kp = await crypto.subtle.generateKey({ name: 'ECDH', namedCurve: 'P-256' }, true, ['deriveBits']);
		asPriv = kp.privateKey;
		asPubRaw = new Uint8Array(await crypto.subtle.exportKey('raw', kp.publicKey));
	}
	const uaKey = await crypto.subtle.importKey('raw', uaPub, { name: 'ECDH', namedCurve: 'P-256' }, false, []);
	const ecdh = new Uint8Array(await crypto.subtle.deriveBits({ name: 'ECDH', public: uaKey }, asPriv, 256));

	const keyInfo = concat(enc.encode('WebPush: info\0'), uaPub, asPubRaw);
	const ikm = await hkdf(authSecret, ecdh, keyInfo, 256);
	const cek = await hkdf(salt, ikm, enc.encode('Content-Encoding: aes128gcm\0'), 128);
	const nonce = await hkdf(salt, ikm, enc.encode('Content-Encoding: nonce\0'), 96);

	const data = concat(plaintext, new Uint8Array([2])); // 單一最後一筆 record 的分隔位元組
	const aes = await crypto.subtle.importKey('raw', cek, 'AES-GCM', false, ['encrypt']);
	const ct = new Uint8Array(await crypto.subtle.encrypt({ name: 'AES-GCM', iv: nonce }, aes, data));

	const rs = new Uint8Array([0, 0, 0x10, 0]); // 4096
	return concat(salt, rs, new Uint8Array([asPubRaw.length]), asPubRaw, ct);
}

/** VAPID JWT（ES256）。exp 預設 12 小時（上限 24h）。 */
export async function vapidJwt({ audience, subject, publicKey, privateKey, expSeconds = 12 * 3600, now = Date.now() }) {
	const head = b64uEncode(enc.encode(JSON.stringify({ typ: 'JWT', alg: 'ES256' })));
	const claims = b64uEncode(enc.encode(JSON.stringify({ aud: audience, exp: Math.floor(now / 1000) + expSeconds, sub: subject })));
	const key = await crypto.subtle.importKey('jwk', toJwk(b64uDecode(publicKey), b64uDecode(privateKey)), { name: 'ECDSA', namedCurve: 'P-256' }, false, ['sign']);
	const sig = new Uint8Array(await crypto.subtle.sign({ name: 'ECDSA', hash: 'SHA-256' }, key, enc.encode(`${head}.${claims}`)));
	return `${head}.${claims}.${b64uEncode(sig)}`;
}

/** 送出一則推播，回傳 fetch Response（呼叫端自己量時間）。 */
export async function sendPush({ subscription, payload, vapid, subject, ttl = 60 }) {
	const body = await encryptPayload({
		p256dh: subscription.keys.p256dh,
		auth: subscription.keys.auth,
		plaintext: enc.encode(typeof payload === 'string' ? payload : JSON.stringify(payload)),
	});
	const jwt = await vapidJwt({ audience: new URL(subscription.endpoint).origin, subject, publicKey: vapid.publicKey, privateKey: vapid.privateKey });
	return fetch(subscription.endpoint, {
		method: 'POST',
		headers: {
			Authorization: `vapid t=${jwt}, k=${vapid.publicKey}`,
			'Content-Encoding': 'aes128gcm',
			'Content-Type': 'application/octet-stream',
			TTL: String(ttl),
			Urgency: 'high',
		},
		body,
	});
}
