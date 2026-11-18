import type { Env } from './index';
import { checkFeed } from './judge';
import { sendNextBatch } from './notifications';
import { taipeiTime } from './time';
import type { Push } from './types';

const FEED_INTERVAL_MINUTES = 10;

export interface Deps {
	fetch: typeof fetch;
	push: Push;
}

// 不包 try/catch：讀檔失敗 checkFeed 自己會結束；真的丟錯（例如 D1）就讓這次呼叫失敗，批次下一分鐘照送。
// 讀檔先於送批次：讀檔的那一分鐘讀到的新通知，同一次就送出第一批。
export async function tick(env: Env, now: Date, deps: Deps): Promise<void> {
	if (taipeiTime(now).minutes % FEED_INTERVAL_MINUTES === 0) await checkFeed(env, now, deps.fetch);
	await sendNextBatch(env.DB, now, deps.push);
}
