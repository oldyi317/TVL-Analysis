import assert from 'node:assert/strict';
import { generateKeyPairSync, createPublicKey, verify } from 'node:crypto';
import { encryptPayload, vapidJwt, b64uDecode, b64uEncode } from '../src/webpush.js';

// ---- 1. RFC 8291 第 5 節 / 附錄 A 已知答案 ----
const rfc = {
	auth: 'BTBZMqHH6r4Tts7J_aSIgg',
	uaPublic: 'BCVxsr7N_eNgVRqvHtD0zTZsEc6-VV-JvLexhqUzORcxaOzi6-AYWXvTBHm4bjyPjs7Vd8pZGH6SRpkNtoIAiw4',
	asPrivate: 'yfWPiYE-n46HLnH0KqZOF1fJJU3MYrct3AELtAQ-oRw',
	asPublic: 'BP4z9KsN6nGRTbVYI_c7VJSPQTBtkgcy27mlmlMoZIIgDll6e3vCYLocInmYWAmS6TlzAC8wEqKK6PBru3jl7A8',
	salt: 'DGv6ra1nlYgDCS1FRnbzlw',
	body:
		'DGv6ra1nlYgDCS1FRnbzlwAAEABBBP4z9KsN6nGRTbVYI_c7VJSPQTBtkgcy27ml' +
		'mlMoZIIgDll6e3vCYLocInmYWAmS6TlzAC8wEqKK6PBru3jl7A_yl95bQpu6cVPT' +
		'pK4Mqgkf1CXztLVBSt2Ks3oZwbuwXPXLWyouBWLVWGNWQexSgSxsj_Qulcy4a-fN',
};
const out = await encryptPayload({
	p256dh: rfc.uaPublic,
	auth: rfc.auth,
	plaintext: new TextEncoder().encode('When I grow up, I want to be a watermelon'),
	salt: b64uDecode(rfc.salt),
	senderKey: { publicRaw: b64uDecode(rfc.asPublic), d: b64uDecode(rfc.asPrivate) },
});
assert.deepEqual(Buffer.from(out), Buffer.from(b64uDecode(rfc.body)));
console.log(`OK RFC 8291 已知答案：${out.length} bytes 逐位元組相同`);

// ---- 2. VAPID JWT ----
const { privateKey, publicKey } = generateKeyPairSync('ec', { namedCurve: 'P-256' });
const jwk = privateKey.export({ format: 'jwk' });
const pubRaw = Buffer.concat([Buffer.from([4]), Buffer.from(jwk.x, 'base64url'), Buffer.from(jwk.y, 'base64url')]);
const pub = pubRaw.toString('base64url');
const now = Date.now();
const jwt = await vapidJwt({ audience: 'https://push.example.net', subject: 'https://tvl-push-proto.example.workers.dev', publicKey: pub, privateKey: jwk.d, now });
const [h, c, s] = jwt.split('.');
const ok = verify('sha256', Buffer.from(`${h}.${c}`), { key: createPublicKey({ key: { kty: 'EC', crv: 'P-256', x: jwk.x, y: jwk.y }, format: 'jwk' }), dsaEncoding: 'ieee-p1363' }, Buffer.from(s, 'base64url'));
assert.ok(ok, '簽章驗證失敗');
const claims = JSON.parse(Buffer.from(c, 'base64url'));
assert.equal(claims.aud, 'https://push.example.net');
assert.equal(claims.sub, 'https://tvl-push-proto.example.workers.dev');
assert.ok(claims.exp > now / 1000 && claims.exp - now / 1000 <= 86400);
assert.equal(JSON.parse(Buffer.from(h, 'base64url')).alg, 'ES256');
console.log('OK VAPID JWT：驗簽成功，aud/sub/exp 正確', claims);

// ---- 3. 準備計時用的隨機接收端金鑰 ----
const rk = generateKeyPairSync('ec', { namedCurve: 'P-256' });
const rj = rk.privateKey.export({ format: 'jwk' });
const rpub = b64uEncode(Buffer.concat([Buffer.from([4]), Buffer.from(rj.x, 'base64url'), Buffer.from(rj.y, 'base64url')]));
const payload = new TextEncoder().encode(JSON.stringify({ id: 'x', sentAt: Date.now(), title: '推播測試', body: 'b'.repeat(80), url: '/matches.html?from=push&id=x' }));
console.log('payload bytes:', payload.length);

// ---- 4. 粗略計時 ----
const times = [];
for (let i = 0; i < 200; i++) {
	const t0 = performance.now();
	await vapidJwt({ audience: 'https://push.example.net', subject: 'https://x.example', publicKey: pub, privateKey: jwk.d });
	await encryptPayload({ p256dh: rpub, auth: b64uEncode(crypto.getRandomValues(new Uint8Array(16))), plaintext: payload });
	times.push(performance.now() - t0);
}
const avg = times.reduce((a, b) => a + b, 0) / times.length;
console.log(`計時（Node，200 次 簽 JWT＋加密 ${payload.length} bytes）：平均 ${avg.toFixed(3)} ms，最大 ${Math.max(...times).toFixed(3)} ms`);
console.log('註：只是量級參考，不等於 Workers 的 CPU 計費時間。');
