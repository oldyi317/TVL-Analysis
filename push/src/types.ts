export type Group = 'men' | 'women';
export type GroupPref = Group | 'both';
export type Kind = 'results' | 'today';

export interface FeedMatch {
	group: Group;
	game_id: number;
	date: string;
	start_time: string | null;
	home: string;
	away: string;
	home_sets_won: number | null;
	away_sets_won: number | null;
	home_win_prob: number | null;
}

export type FeedLinks = Record<GroupPref, string>;

export interface Feed {
	cup_id: number;
	links: FeedLinks;
	matches: FeedMatch[];
}

export interface Content {
	title: string;
	body: string;
	navigate: string;
}

// 一則通知的三個版本；該組別沒有內容時是 null。
export type Versions = Record<GroupPref, Content | null>;

export interface PushTarget {
	endpoint: string;
	p256dh: string;
	auth: string;
}

// 回推播服務的 HTTP 狀態碼；連不上或逾時就 reject。
export type Push = (target: PushTarget, payload: string) => Promise<number>;
