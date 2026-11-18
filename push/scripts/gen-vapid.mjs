// 產生一組 VAPID 金鑰對（P-256）。需要 Node 22 以上，只用內建模組。
import { generateKeyPairSync } from 'node:crypto';

const { publicKey, privateKey } = generateKeyPairSync('ec', { namedCurve: 'P-256' });
const jwk = privateKey.export({ format: 'jwk' });
const raw = Buffer.concat([Buffer.from([0x04]), Buffer.from(jwk.x, 'base64url'), Buffer.from(jwk.y, 'base64url')]);

console.log(`VAPID_PUBLIC_KEY=${raw.toString('base64url')}`);
console.log(`VAPID_PRIVATE_KEY=${jwk.d}`);

console.error(`
公鑰：填進 push/wrangler.jsonc 對應環境的 VAPID_PUBLIC_KEY。
私鑰：用 wrangler secret put VAPID_PRIVATE_KEY 存（預覽環境加 --env preview），不要放進 repo。
私鑰要自己備份；掉了等於所有推播訂閱作廢。`);
