-- TVL 資料庫 Schema（可重複執行）
-- 注意：男女組的 team_id 可能重複，因此 teams 使用複合主鍵 (team_id, gender)

-- 依 FK 順序先刪除子表，再刪除父表
DROP TABLE IF EXISTS player_match_stats;
DROP TABLE IF EXISTS players;
DROP TABLE IF EXISTS teams;

CREATE TABLE teams (
    team_id   INTEGER NOT NULL,
    team_name TEXT    NOT NULL,
    gender    TEXT    NOT NULL CHECK (gender IN ('M', 'F')),
    PRIMARY KEY (team_id, gender)
);

CREATE TABLE players (
    player_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id       INTEGER NOT NULL,
    gender        TEXT    NOT NULL,
    jersey_number INTEGER,
    name          TEXT,
    position      TEXT,
    dob           DATE,
    height_cm     REAL,
    weight_kg     REAL,
    FOREIGN KEY (team_id, gender) REFERENCES teams (team_id, gender)
);

CREATE TABLE player_match_stats (
    stat_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id         INTEGER NOT NULL,
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
    FOREIGN KEY (player_id) REFERENCES players (player_id)
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
CREATE INDEX IF NOT EXISTS idx_pms_player_id  ON player_match_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_pms_match_date ON player_match_stats(match_date);
CREATE INDEX IF NOT EXISTS idx_players_team_gender ON players(team_id, gender);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
