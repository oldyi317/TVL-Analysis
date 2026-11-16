-- 推播訂閱：推播位址本身就是憑證，唯一。id 只增不重用，給分批排序用。
CREATE TABLE subscriptions (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	endpoint TEXT NOT NULL UNIQUE,
	p256dh TEXT NOT NULL,
	auth TEXT NOT NULL,
	want_results INTEGER NOT NULL CHECK (want_results IN (0, 1)),
	want_today INTEGER NOT NULL CHECK (want_today IN (0, 1)),
	group_pref TEXT NOT NULL CHECK (group_pref IN ('men', 'women', 'both')),
	created_at TEXT NOT NULL,
	updated_at TEXT NOT NULL
);

-- 已發過的賽事：場次編號男女各自獨立，識別要三欄合起來。寫入時用 INSERT OR IGNORE 認領。
CREATE TABLE sent_matches (
	cup_id INTEGER NOT NULL,
	match_group TEXT NOT NULL CHECK (match_group IN ('men', 'women')),
	game_id INTEGER NOT NULL,
	sent_at TEXT NOT NULL,
	PRIMARY KEY (cup_id, match_group, game_id)
);

-- 今日賽事已發過的台北日期（YYYY-MM-DD）。
CREATE TABLE sent_today_dates (
	taipei_date TEXT PRIMARY KEY,
	sent_at TEXT NOT NULL
);

-- 每一則通知一筆：批次進度、開始送時定下來的三個版本內容，送完後就是發送紀錄。
-- 進度各則分開記，暫停一則（next_offset 停著）再接著送不需要額外狀態。
-- content_* 是 JSON（title、body、navigate），該版本沒有內容時為 NULL。
CREATE TABLE notifications (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	kind TEXT NOT NULL CHECK (kind IN ('results', 'today')),
	started_at TEXT NOT NULL,
	finished_at TEXT,
	next_offset INTEGER NOT NULL DEFAULT 0,
	target_count INTEGER NOT NULL DEFAULT 0,
	success_count INTEGER NOT NULL DEFAULT 0,
	deleted_count INTEGER NOT NULL DEFAULT 0,
	subscription_count INTEGER NOT NULL DEFAULT 0,
	content_men TEXT,
	content_women TEXT,
	content_both TEXT
);

-- 發送服務第一次成功讀推播賽事檔的記號：有這一列代表讀過。
-- 用 INSERT OR IGNORE 寫，changes 為 1 的那次呼叫就是第一次，負責把已有賽果的場次記成已發。
CREATE TABLE first_feed_read (
	id INTEGER PRIMARY KEY CHECK (id = 1),
	read_at TEXT NOT NULL
);
