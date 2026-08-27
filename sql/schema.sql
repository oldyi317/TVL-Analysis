-- TVL 資料庫 Schema（可重複執行，冪等：僅 CREATE TABLE IF NOT EXISTS，不清空既有資料）
-- players = 球員身分層（跨週不變的屬性）；roster_registrations = 球員週次登錄層
-- （球員×週次×隊伍×背號×位置，因企業排球每週登錄名單可能不同）
-- 注意：男女組的 team_id 可能重複，因此 teams 使用複合主鍵 (team_id, gender)

CREATE TABLE IF NOT EXISTS teams (
    team_id   INTEGER NOT NULL,
    team_name TEXT    NOT NULL,
    gender    TEXT    NOT NULL CHECK (gender IN ('M', 'F')),
    PRIMARY KEY (team_id, gender)
);

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

CREATE TABLE IF NOT EXISTS matches (
    match_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         INTEGER NOT NULL,
    gender          TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    match_date      DATE NOT NULL,
    venue           TEXT,
    round_name      TEXT,
    game_label      TEXT,
    is_golden_set   INTEGER NOT NULL DEFAULT 0,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
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

-- 效能索引
CREATE INDEX IF NOT EXISTS idx_pms_registration_id ON player_match_stats(registration_id);
CREATE INDEX IF NOT EXISTS idx_pms_match_date       ON player_match_stats(match_date);
CREATE INDEX IF NOT EXISTS idx_roster_player        ON roster_registrations(player_id);
CREATE INDEX IF NOT EXISTS idx_roster_team_gender   ON roster_registrations(team_id, gender);
CREATE INDEX IF NOT EXISTS idx_roster_week          ON roster_registrations(week_label);
CREATE INDEX IF NOT EXISTS idx_matches_date         ON matches(match_date);
