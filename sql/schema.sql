-- TVL 資料庫 Schema（可重複執行）
-- 注意：男女組的 team_id 可能重複，因此 teams 使用複合主鍵 (team_id, gender)

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

CREATE TABLE IF NOT EXISTS player_match_stats (
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
    FOREIGN KEY (player_id) REFERENCES players (player_id)
);
