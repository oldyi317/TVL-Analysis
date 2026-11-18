import { hasResult, parseFeed } from './feed';
import type { Env } from './index';
import { createNotification } from './notifications';
import { resultsVersions, todayVersions } from './text';
import { daysBetween, inWindow, parseWindow, taipeiTime } from './time';
import type { Feed, FeedMatch } from './types';

const RESULTS_DAYS = 7;
const FEED_TIMEOUT_MS = 10_000;

function keys(matches: FeedMatch[]): string {
	return JSON.stringify(matches.map((match) => ({ group: match.group, game_id: match.game_id })));
}

// 一條語句認領整批：INSERT OR IGNORE 只對真的寫進去的列回 RETURNING，所以回傳的就是這次認領到的。
// 並行時同一場只會有一個呼叫拿到，不依賴先讀後寫。
export async function claimMatches(db: D1Database, cupId: number, matches: FeedMatch[], now: Date): Promise<FeedMatch[]> {
	if (matches.length === 0) return [];
	const { results } = await db
		.prepare(
			`INSERT OR IGNORE INTO sent_matches (cup_id, match_group, game_id, sent_at)
			SELECT ?, json_extract(value, '$.group'), json_extract(value, '$.game_id'), ? FROM json_each(?)
			RETURNING match_group, game_id`
		)
		.bind(cupId, now.toISOString(), keys(matches))
		.all<{ match_group: string; game_id: number }>();
	const claimed = new Set(results.map((row) => `${row.match_group}/${row.game_id}`));
	return matches.filter((match) => claimed.has(`${match.group}/${match.game_id}`));
}

export async function claimToday(db: D1Database, date: string, now: Date): Promise<boolean> {
	const result = await db
		.prepare('INSERT OR IGNORE INTO sent_today_dates (taipei_date, sent_at) VALUES (?, ?)')
		.bind(date, now.toISOString())
		.run();
	return result.meta.changes === 1;
}

// 連不上、逾時、狀態碼不是 2xx、不是 JSON、格式不符都回 null，呼叫端什麼都不寫。
async function readFeed(env: Env, fetcher: typeof fetch): Promise<Feed | null> {
	try {
		const response = await fetcher(env.FEED_URL, { signal: AbortSignal.timeout(FEED_TIMEOUT_MS) });
		if (!response.ok) return null;
		return parseFeed(await response.json());
	} catch {
		return null;
	}
}

// 同一個 batch：還沒有記號才把已有賽果的場次全記成已發，再寫記號。
// 記號那條的 changes 為 1 的呼叫就是第一次；兩條在同一個交易裡，不會只寫一半。
async function isFirstRead(db: D1Database, feed: Feed, now: Date): Promise<boolean> {
	const sentAt = now.toISOString();
	const [, mark] = await db.batch([
		db
			.prepare(
				`INSERT OR IGNORE INTO sent_matches (cup_id, match_group, game_id, sent_at)
				SELECT ?, json_extract(value, '$.group'), json_extract(value, '$.game_id'), ? FROM json_each(?)
				WHERE NOT EXISTS (SELECT 1 FROM first_feed_read)`
			)
			.bind(feed.cup_id, sentAt, keys(feed.matches.filter(hasResult))),
		db.prepare('INSERT OR IGNORE INTO first_feed_read (id, read_at) VALUES (1, ?)').bind(sentAt)
	]);
	return mark.meta.changes === 1;
}

export async function checkFeed(env: Env, now: Date, fetcher: typeof fetch): Promise<void> {
	const feed = await readFeed(env, fetcher);
	if (!feed) return;
	if (await isFirstRead(env.DB, feed, now)) return;

	const sendWindow = parseWindow(env.SEND_WINDOW);
	const taipei = taipeiTime(now);
	if (!sendWindow || !inWindow(sendWindow, taipei.minutes)) return;

	// 今日賽事先建立，賽果摘要後建立。
	const todayWindow = parseWindow(env.TODAY_WINDOW);
	if (todayWindow && inWindow(todayWindow, taipei.minutes)) {
		const todays = feed.matches.filter((match) => match.date === taipei.date);
		if (todays.length > 0 && (await claimToday(env.DB, taipei.date, now))) {
			await createNotification(env.DB, 'today', todayVersions(todays, feed.links, env.SITE_ROOT, taipei.date), now);
		}
	}

	// 超過 7 天的也在這裡一起記成已發，只是不進通知。
	const claimed = await claimMatches(env.DB, feed.cup_id, feed.matches.filter(hasResult), now);
	const recent = claimed.filter((match) => daysBetween(match.date, taipei.date) <= RESULTS_DAYS);
	if (recent.length === 0) return;
	await createNotification(env.DB, 'results', resultsVersions(recent, feed.links, env.SITE_ROOT), now);
}
