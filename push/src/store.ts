import type { Subscription } from './validate';

const SUBSCRIPTION_LIMIT = 500;

// 兩條在同一個 batch（一個交易）裡跑：已有就覆寫偏好，沒有且未滿 500 筆才新增。
// 上限判斷寫在 INSERT 的條件裡，不先讀筆數再寫，並行登記也不會超過。
export async function upsertSubscription(db: D1Database, sub: Subscription, now: Date): Promise<boolean> {
	const values = [
		sub.endpoint,
		sub.p256dh,
		sub.auth,
		sub.wantResults ? 1 : 0,
		sub.wantToday ? 1 : 0,
		sub.group,
		now.toISOString()
	];
	const [updated, inserted] = await db.batch([
		db
			.prepare(
				`UPDATE subscriptions
				SET p256dh = ?2, auth = ?3, want_results = ?4, want_today = ?5, group_pref = ?6, updated_at = ?7
				WHERE endpoint = ?1`
			)
			.bind(...values),
		db
			.prepare(
				`INSERT INTO subscriptions (endpoint, p256dh, auth, want_results, want_today, group_pref, created_at, updated_at)
				SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7
				WHERE NOT EXISTS (SELECT 1 FROM subscriptions WHERE endpoint = ?1)
				AND (SELECT COUNT(*) FROM subscriptions) < ?8`
			)
			.bind(...values, SUBSCRIPTION_LIMIT)
	]);
	return updated.meta.changes + inserted.meta.changes > 0;
}

export async function deleteSubscription(db: D1Database, endpoint: string): Promise<void> {
	await db.prepare('DELETE FROM subscriptions WHERE endpoint = ?').bind(endpoint).run();
}
