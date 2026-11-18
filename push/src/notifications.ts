import type { Content, GroupPref, Kind, Push, Versions } from './types';

const BATCH_SIZE = 15;

interface NotificationRow {
	id: number;
	kind: Kind;
	next_offset: number;
	content_men: string | null;
	content_women: string | null;
	content_both: string | null;
}

interface Recipient {
	id: number;
	endpoint: string;
	p256dh: string;
	auth: string;
	group_pref: GroupPref;
}

// 誰該收一則：想收這一種，而且所選組別有內容。建立時算 target_count 與分批選人共用。
// 內插的只有程式內的常數（欄名與固定的組別字面值），沒有外部輸入。
function recipientCondition(kind: Kind, hasMen: boolean, hasWomen: boolean): string {
	const column = kind === 'results' ? 'want_results' : 'want_today';
	const prefs = ["'both'", ...(hasMen ? ["'men'"] : []), ...(hasWomen ? ["'women'"] : [])];
	return `${column} = 1 AND group_pref IN (${prefs.join(', ')})`;
}

function serialize(content: Content | null): string | null {
	return content && JSON.stringify(content);
}

export async function createNotification(db: D1Database, kind: Kind, versions: Versions, now: Date): Promise<void> {
	if (!versions.both) return;
	const condition = recipientCondition(kind, versions.men !== null, versions.women !== null);
	await db
		.prepare(
			`INSERT INTO notifications (kind, started_at, content_men, content_women, content_both, target_count, subscription_count)
			VALUES (?, ?, ?, ?, ?,
				(SELECT COUNT(*) FROM subscriptions WHERE ${condition}),
				(SELECT COUNT(*) FROM subscriptions))`
		)
		.bind(
			kind,
			now.toISOString(),
			serialize(versions.men),
			serialize(versions.women),
			serialize(versions.both)
		)
		.run();
}

// 進度是「已認領到的最後一筆訂閱 id」。只有進度還是 from 的那一個呼叫改得到，回 true。
export async function claimBatch(db: D1Database, id: number, from: number, to: number): Promise<boolean> {
	const result = await db
		.prepare('UPDATE notifications SET next_offset = ? WHERE id = ? AND next_offset = ?')
		.bind(to, id, from)
		.run();
	return result.meta.changes === 1;
}

function contentFor(row: NotificationRow, group: GroupPref): string | null {
	if (group === 'men') return row.content_men;
	if (group === 'women') return row.content_women;
	return row.content_both;
}

// 一次呼叫只送一批。重複的保護只靠 claimBatch 的條件式寫入，前面讀到的資料可以是舊的。
export async function sendNextBatch(db: D1Database, now: Date, push: Push): Promise<void> {
	const notification = await db
		.prepare(
			`SELECT id, kind, next_offset, content_men, content_women, content_both
			FROM notifications WHERE finished_at IS NULL
			ORDER BY (kind = 'today') DESC, id LIMIT 1`
		)
		.first<NotificationRow>();
	if (!notification) return;

	const condition = recipientCondition(notification.kind, notification.content_men !== null, notification.content_women !== null);
	const { results: batch } = await db
		.prepare(
			`SELECT id, endpoint, p256dh, auth, group_pref FROM subscriptions
			WHERE id > ? AND ${condition} ORDER BY id LIMIT ?`
		)
		.bind(notification.next_offset, BATCH_SIZE)
		.all<Recipient>();

	if (batch.length === 0) {
		await db
			.prepare('UPDATE notifications SET finished_at = ? WHERE id = ? AND finished_at IS NULL')
			.bind(now.toISOString(), notification.id)
			.run();
		return;
	}

	if (!(await claimBatch(db, notification.id, notification.next_offset, batch[batch.length - 1].id))) return;

	const statuses = await Promise.allSettled(
		batch.map((sub) => {
			const payload = JSON.stringify({
				web_push: 8030,
				notification: JSON.parse(contentFor(notification, sub.group_pref)!)
			});
			return push({ endpoint: sub.endpoint, p256dh: sub.p256dh, auth: sub.auth }, payload);
		})
	);

	let success = 0;
	const gone: Recipient[] = [];
	statuses.forEach((outcome, i) => {
		if (outcome.status !== 'fulfilled') return;
		if (outcome.value >= 200 && outcome.value < 300) success++;
		else if (outcome.value === 404 || outcome.value === 410) gone.push(batch[i]);
	});

	await db.batch([
		...gone.map((sub) => db.prepare('DELETE FROM subscriptions WHERE id = ?').bind(sub.id)),
		db
			.prepare('UPDATE notifications SET success_count = success_count + ?, deleted_count = deleted_count + ? WHERE id = ?')
			.bind(success, gone.length, notification.id)
	]);
}
