// 產生 VAPID 金鑰對，印出可直接貼進 .dev.vars 與 wrangler.jsonc vars 的內容。
import { generateKeyPairSync } from 'node:crypto';

const { privateKey } = generateKeyPairSync('ec', { namedCurve: 'P-256' });
const jwk = privateKey.export({ format: 'jwk' });
const b = (s) => Buffer.from(s, 'base64url');
const pub = Buffer.concat([Buffer.from([4]), b(jwk.x), b(jwk.y)]).toString('base64url');
console.log(`VAPID_PUBLIC=${pub}`);
console.log(`VAPID_PRIVATE=${jwk.d}`);
