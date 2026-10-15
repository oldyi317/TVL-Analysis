-- TVL 資料庫 Schema（可重複執行，冪等：僅 CREATE TABLE IF NOT EXISTS，不清空既有資料）
-- players = 球員身分層（跨週不變的屬性）；roster_registrations = 球員週次登錄層
-- （球員×週次×隊伍×背號×位置，因企業排球每週登錄名單可能不同）
-- 注意：男女組的 team_id 可能重複，因此 teams 使用複合主鍵 (team_id, gender)

CREATE TABLE IF NOT EXISTS teams (
    team_id   INTEGER NOT NULL,
    team_name TEXT    NOT NULL,
    gender    TEXT    NOT NULL CHECK (gender IN ('M', 'F')),
    short_name    TEXT,
    official_name TEXT,
    color         TEXT,
    is_active     INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    PRIMARY KEY (team_id, gender)
);
-- 過渡期 short_name 與 team_name 同值（既有程式仍讀 team_name）；球隊身分種子見 sql/team_seed.sql

-- 隊名別名 → 球隊。source 值域：short / official / match / legacy / ext_stats
CREATE TABLE IF NOT EXISTS team_aliases (
    alias   TEXT PRIMARY KEY,
    team_id INTEGER NOT NULL,
    gender  TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    source  TEXT NOT NULL,
    FOREIGN KEY (team_id, gender) REFERENCES teams (team_id, gender)
);
CREATE INDEX IF NOT EXISTS idx_team_aliases_team ON team_aliases (team_id, gender);

CREATE TABLE IF NOT EXISTS players (
    player_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT,
    gender     TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    dob        DATE,
    height_cm  REAL,
    weight_kg  REAL
);

CREATE TABLE IF NOT EXISTS roster_registrations (
    registration_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id        INTEGER NOT NULL,
    team_id          INTEGER NOT NULL,
    gender           TEXT    NOT NULL CHECK (gender IN ('M', 'F')),
    cup_id           INTEGER NOT NULL,
    week_label       TEXT    NOT NULL,
    week_start_date  DATE,
    jersey_number    INTEGER,
    position         TEXT,
    source           TEXT    NOT NULL CHECK (source IN ('match_page', 'backfill')),
    FOREIGN KEY (player_id) REFERENCES players (player_id),
    FOREIGN KEY (team_id, gender) REFERENCES teams (team_id, gender),
    UNIQUE (player_id, team_id, gender, cup_id, week_label)
);

CREATE TABLE IF NOT EXISTS player_match_stats (
    stat_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    registration_id   INTEGER NOT NULL,
    match_date        DATE,
    opponent          TEXT,
    sets_played       INTEGER,
    attack_total      INTEGER,
    attack_points     INTEGER,
    block_points      INTEGER,
    serve_total       INTEGER,
    serve_points      INTEGER,
    receive_total     INTEGER,
    receive_excellent INTEGER,
    dig_total         INTEGER,
    dig_excellent     INTEGER,
    set_total         INTEGER,
    set_excellent     INTEGER,
    total_points      INTEGER,
    is_golden_set     INTEGER NOT NULL DEFAULT 0 CHECK (is_golden_set IN (0, 1)),
    FOREIGN KEY (registration_id) REFERENCES roster_registrations (registration_id)
);

-- 賽季起訖採「七月到隔年六月」不重疊視窗，讓季前季後的賽程也落在正確賽季
CREATE TABLE IF NOT EXISTS seasons (
    cup_id     INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    start_date DATE NOT NULL,
    end_date   DATE NOT NULL
);

INSERT OR IGNORE INTO seasons (cup_id, name, start_date, end_date) VALUES
    (20, '2024-25', '2024-07-01', '2025-06-30'),
    (21, '2025-26', '2025-07-01', '2026-06-30'),
    (22, '2026-27', '2026-07-01', '2027-06-30');

CREATE TABLE IF NOT EXISTS matches (
    match_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         INTEGER NOT NULL,
    gender          TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    cup_id          INTEGER NOT NULL REFERENCES seasons (cup_id),
    match_date      DATE NOT NULL,
    start_time      TEXT,
    venue           TEXT,
    round_name      TEXT,
    game_label      TEXT,
    is_golden_set   INTEGER NOT NULL DEFAULT 0,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    home_team_id    INTEGER,  -- 複合主鍵 teams(team_id, gender) 無法用 ALTER 加 FK；FK 留到 contract 步重建表時加
    away_team_id    INTEGER,
    home_set1       INTEGER,
    home_set2       INTEGER,
    home_set3       INTEGER,
    home_set4       INTEGER,
    home_set5       INTEGER,
    home_total      INTEGER,
    away_set1       INTEGER,
    away_set2       INTEGER,
    away_set3       INTEGER,
    away_set4       INTEGER,
    away_set5       INTEGER,
    away_total      INTEGER,
    home_sets_won   INTEGER,
    away_sets_won   INTEGER,
    UNIQUE (game_id, gender)
);

-- 賽前勝率（ADR 0003）：每日對未開打賽事重算 upsert；開打後凍結不再改寫
CREATE TABLE IF NOT EXISTS match_predictions (
    match_id       INTEGER NOT NULL REFERENCES matches (match_id),
    model_version  TEXT    NOT NULL,
    home_win_prob  REAL    NOT NULL,
    computed_at    TEXT    NOT NULL,
    UNIQUE (match_id, model_version)
);

-- 效能索引
CREATE INDEX IF NOT EXISTS idx_pms_registration_id ON player_match_stats(registration_id);
CREATE INDEX IF NOT EXISTS idx_pms_match_date       ON player_match_stats(match_date);
CREATE INDEX IF NOT EXISTS idx_roster_player        ON roster_registrations(player_id);
CREATE INDEX IF NOT EXISTS idx_roster_team_gender   ON roster_registrations(team_id, gender);
CREATE INDEX IF NOT EXISTS idx_roster_week          ON roster_registrations(week_label);
CREATE INDEX IF NOT EXISTS idx_matches_date         ON matches(match_date);
