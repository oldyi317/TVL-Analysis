import type { Feed, FeedLinks, FeedMatch } from './types';

// #243 的 schema 對網址欄偏鬆（放得過 //evil.com/），這裡自己收緊。
const LINK_PATTERN = /^\/(?:[A-Za-z0-9._-]+\/)*$/;
const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;
const TIME_PATTERN = /^\d{2}:\d{2}$/;

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isLink(value: unknown): value is string {
	return typeof value === 'string' && LINK_PATTERN.test(value);
}

function isNonEmptyString(value: unknown): value is string {
	return typeof value === 'string' && value !== '';
}

function isSets(value: unknown): value is number | null {
	return value === null || (Number.isInteger(value) && (value as number) >= 0);
}

function isProbability(value: unknown): value is number | null {
	return value === null || (typeof value === 'number' && value >= 0 && value <= 1);
}

function parseLinks(value: unknown): FeedLinks | null {
	if (!isRecord(value)) return null;
	const { men, women, both } = value;
	if (!isLink(men) || !isLink(women) || !isLink(both)) return null;
	return { men, women, both };
}

function parseMatch(value: unknown): FeedMatch | null {
	if (!isRecord(value)) return null;
	const { group, game_id, date, start_time, home, away, home_sets_won, away_sets_won, home_win_prob } = value;
	if (group !== 'men' && group !== 'women') return null;
	if (!Number.isInteger(game_id)) return null;
	if (typeof date !== 'string' || !DATE_PATTERN.test(date)) return null;
	if (start_time !== null && (typeof start_time !== 'string' || !TIME_PATTERN.test(start_time))) return null;
	if (!isNonEmptyString(home) || !isNonEmptyString(away)) return null;
	if (!isSets(home_sets_won) || !isSets(away_sets_won)) return null;
	if (!isProbability(home_win_prob)) return null;
	return { group, game_id: game_id as number, date, start_time, home, away, home_sets_won, away_sets_won, home_win_prob };
}

// 任何一項不符整份回 null：呼叫端把它當成讀檔失敗。generated_at 每次建置都會變，不檢查也不用。
export function parseFeed(data: unknown): Feed | null {
	if (!isRecord(data)) return null;
	if (data.version !== 1 || !Number.isInteger(data.cup_id)) return null;
	const links = parseLinks(data.links);
	if (!links || !Array.isArray(data.matches)) return null;
	const matches: FeedMatch[] = [];
	for (const item of data.matches) {
		const parsed = parseMatch(item);
		if (!parsed) return null;
		matches.push(parsed);
	}
	return { cup_id: data.cup_id as number, links, matches };
}

export function hasResult(match: FeedMatch): boolean {
	return match.home_sets_won !== null && match.away_sets_won !== null;
}
