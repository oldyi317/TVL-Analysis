import { applyD1Migrations } from 'cloudflare:test';
import { env } from 'cloudflare:workers';
import { beforeEach } from 'vitest';

await applyD1Migrations(env.DB, env.TEST_MIGRATIONS);

// 儲存不會在測試之間自動清空，每個測試從空表開始。
beforeEach(async () => {
	await env.DB.batch(
		['subscriptions', 'sent_matches', 'sent_today_dates', 'notifications', 'first_feed_read'].map((table) =>
			env.DB.prepare(`DELETE FROM ${table}`)
		)
	);
});
