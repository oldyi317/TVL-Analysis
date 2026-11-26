import type { Push, PushTarget } from './types';

const SUBJECT = 'https://vstat.tw';
const JWT_LIFETIME_SECONDS = 12 * 3600;
const TTL_SECONDS = 43200;
const TIMEOUT_MS = 10_000;
const RECORD_SIZE = 4096;

const encoder = new TextEncoder();

function toBase64Url(bytes: Uint8Array): string {
	let binary = '';
	for (const b of bytes) binary += String.fromCharCode(b);
	return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export function fromBase64Url(text: string): Uint8Array {
	const base64 = text.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (text.length % 4)) % 4);
	return Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
}

function concat(...parts: Uint8Array[]): Uint8Array {
	const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
	let offset = 0;
	for (const part of parts) {
		out.set(part, offset);
		offset += part.length;
	}
	return out;
}

async function hkdf(salt: Uint8Array, ikm: Uint8Array, info: Uint8Array, bytes: number): Promise<Uint8Array> {
	const key = await crypto.subtle.importKey('raw', ikm, 'HKDF', false, ['deriveBits']);
	return new Uint8Array(await crypto.subtle.deriveBits({ name: 'HKDF', hash: 'SHA-256', salt, info }, key, bytes * 8));
}

// 未壓縮公鑰（65 bytes）轉 JWK；WebCrypto 的 raw 匯入不收私鑰，所以私鑰要走 JWK。
function toJwk(publicKey: Uint8Array, d: string): JsonWebKey {
	return { kty: 'EC', crv: 'P-256', x: toBase64Url(publicKey.slice(1, 33)), y: toBase64Url(publicKey.slice(33, 65)), d };
}

// RFC 8291 aes128gcm：單一 record，明文後接 0x02 分隔位元組。
export async function encryptPayload(target: Pick<PushTarget, 'p256dh' | 'auth'>, plaintext: Uint8Array): Promise<Uint8Array> {
	const uaPublic = fromBase64Url(target.p256dh);
	const salt = crypto.getRandomValues(new Uint8Array(16));
	const ephemeral = (await crypto.subtle.generateKey({ name: 'ECDH', namedCurve: 'P-256' }, false, ['deriveBits'])) as CryptoKeyPair;
	const asPublic = new Uint8Array((await crypto.subtle.exportKey('raw', ephemeral.publicKey)) as ArrayBuffer);
	const uaKey = await crypto.subtle.importKey('raw', uaPublic, { name: 'ECDH', namedCurve: 'P-256' }, false, []);
	// workers-types 把標準的 public 屬性寫成 $public，直接寫字面物件會報型別錯；執行時 workerd 吃的是 public
	const algorithm = { name: 'ECDH', public: uaKey };
	const secret = new Uint8Array(await crypto.subtle.deriveBits(algorithm, ephemeral.privateKey, 256));

	const keyInfo = concat(encoder.encode('WebPush: info\0'), uaPublic, asPublic);
	const ikm = await hkdf(fromBase64Url(target.auth), secret, keyInfo, 32);
	const cek = await hkdf(salt, ikm, encoder.encode('Content-Encoding: aes128gcm\0'), 16);
	const nonce = await hkdf(salt, ikm, encoder.encode('Content-Encoding: nonce\0'), 12);

	const aesKey = await crypto.subtle.importKey('raw', cek, 'AES-GCM', false, ['encrypt']);
	const record = await crypto.subtle.encrypt({ name: 'AES-GCM', iv: nonce }, aesKey, concat(plaintext, new Uint8Array([2])));

	const recordSize = new Uint8Array(4);
	new DataView(recordSize.buffer).setUint32(0, RECORD_SIZE);
	return concat(salt, recordSize, new Uint8Array([asPublic.length]), asPublic, new Uint8Array(record));
}

export function createPush(
	vapid: { publicKey: string; privateKey: string },
	now: Date,
	fetcher: typeof fetch = (input, init) => fetch(input, init)
): Push {
	const exp = Math.floor(now.getTime() / 1000) + JWT_LIFETIME_SECONDS;
	const header = toBase64Url(encoder.encode(JSON.stringify({ typ: 'JWT', alg: 'ES256' })));
	// 簽章金鑰只匯入一次；同一個 audience 的 JWT 只簽一次（每次定時呼叫只有 10 ms CPU）
	let signingKey: Promise<CryptoKey> | undefined;
	const tokens = new Map<string, Promise<string>>();

	async function sign(aud: string): Promise<string> {
		signingKey ??= crypto.subtle.importKey('jwk', toJwk(fromBase64Url(vapid.publicKey), vapid.privateKey), { name: 'ECDSA', namedCurve: 'P-256' }, false, ['sign']);
		const claims = toBase64Url(encoder.encode(JSON.stringify({ aud, exp, sub: SUBJECT })));
		const signature = await crypto.subtle.sign({ name: 'ECDSA', hash: 'SHA-256' }, await signingKey, encoder.encode(`${header}.${claims}`));
		return `${header}.${claims}.${toBase64Url(new Uint8Array(signature))}`;
	}

	function tokenFor(aud: string): Promise<string> {
		let token = tokens.get(aud);
		if (!token) {
			token = sign(aud);
			tokens.set(aud, token);
		}
		return token;
	}

	return async (target, payload) => {
		const [body, jwt] = await Promise.all([encryptPayload(target, encoder.encode(payload)), tokenFor(new URL(target.endpoint).origin)]);
		const response = await fetcher(target.endpoint, {
			method: 'POST',
			headers: {
				Authorization: `vapid t=${jwt}, k=${vapid.publicKey}`,
				'Content-Encoding': 'aes128gcm',
				'Content-Type': 'application/octet-stream',
				TTL: String(TTL_SECONDS)
			},
			body,
			signal: AbortSignal.timeout(TIMEOUT_MS)
		});
		await response.body?.cancel();
		return response.status;
	};
}
