// 測試端的 RFC 8291 解密器，寫法獨立於 src/webpush.ts；先用 RFC 附錄的測試向量驗過才拿來解實作的輸出。
const enc = new TextEncoder();

export function fromB64u(str: string): Uint8Array {
	const b64 = str.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (str.length % 4)) % 4);
	return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
}

export function toB64u(bytes: Uint8Array): string {
	return btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function concat(...parts: Uint8Array[]): Uint8Array {
	const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
	let offset = 0;
	for (const p of parts) {
		out.set(p, offset);
		offset += p.length;
	}
	return out;
}

async function hkdf(salt: Uint8Array, ikm: Uint8Array, info: Uint8Array, length: number): Promise<Uint8Array> {
	const key = await crypto.subtle.importKey('raw', ikm, 'HKDF', false, ['deriveBits']);
	return new Uint8Array(await crypto.subtle.deriveBits({ name: 'HKDF', hash: 'SHA-256', salt, info }, key, length * 8));
}

export interface Receiver {
	publicKey: string;
	privateKey: string;
	auth: string;
}

export async function generateReceiver(): Promise<Receiver> {
	const pair = (await crypto.subtle.generateKey({ name: 'ECDH', namedCurve: 'P-256' }, true, ['deriveBits'])) as CryptoKeyPair;
	const jwk = (await crypto.subtle.exportKey('jwk', pair.privateKey)) as JsonWebKey;
	const raw = new Uint8Array((await crypto.subtle.exportKey('raw', pair.publicKey)) as ArrayBuffer);
	return { publicKey: toB64u(raw), privateKey: jwk.d!, auth: toB64u(crypto.getRandomValues(new Uint8Array(16))) };
}

export interface Header {
	salt: Uint8Array;
	recordSize: number;
	keyId: Uint8Array;
}

export function parseHeader(body: Uint8Array): Header {
	const idLength = body[20];
	return {
		salt: body.slice(0, 16),
		recordSize: new DataView(body.buffer, body.byteOffset + 16, 4).getUint32(0),
		keyId: body.slice(21, 21 + idLength)
	};
}

export async function decrypt(receiver: Receiver, body: Uint8Array): Promise<Uint8Array> {
	const { salt, keyId } = parseHeader(body);
	const ciphertext = body.slice(21 + keyId.length);
	const uaPublic = fromB64u(receiver.publicKey);
	const jwk = { kty: 'EC', crv: 'P-256', x: toB64u(uaPublic.slice(1, 33)), y: toB64u(uaPublic.slice(33, 65)), d: receiver.privateKey };
	const uaPrivate = await crypto.subtle.importKey('jwk', jwk, { name: 'ECDH', namedCurve: 'P-256' }, false, ['deriveBits']);
	const asPublic = await crypto.subtle.importKey('raw', keyId, { name: 'ECDH', namedCurve: 'P-256' }, false, []);
	const algorithm = { name: 'ECDH', public: asPublic }; // workers-types 寫成 $public，見 src/webpush.ts
	const ecdh = new Uint8Array(await crypto.subtle.deriveBits(algorithm, uaPrivate, 256));
	const ikm = await hkdf(fromB64u(receiver.auth), ecdh, concat(enc.encode('WebPush: info\0'), uaPublic, keyId), 32);
	const cek = await hkdf(salt, ikm, enc.encode('Content-Encoding: aes128gcm\0'), 16);
	const nonce = await hkdf(salt, ikm, enc.encode('Content-Encoding: nonce\0'), 12);
	const key = await crypto.subtle.importKey('raw', cek, 'AES-GCM', false, ['decrypt']);
	const padded = new Uint8Array(await crypto.subtle.decrypt({ name: 'AES-GCM', iv: nonce }, key, ciphertext));
	let end = padded.length;
	while (end > 0 && padded[end - 1] === 0) end--;
	if (padded[end - 1] !== 2) throw new Error('最後一筆 record 的分隔位元組不是 0x02');
	return padded.slice(0, end - 1);
}
