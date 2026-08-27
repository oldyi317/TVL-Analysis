import sqlite3
from pathlib import Path

import pytest

from src.etl.migrate_add_cup_id import run_migration

# 遷移前的 v2 schema（無 cup_id）——固定快照，不隨 schema.sql 演進
SCHEMA_V2_OLD = """
CREATE TABLE teams (
    team_id INTEGER NOT NULL, team_name TEXT NOT NULL,
    gender TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    PRIMARY KEY (team_id, gender)
);
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
    gender TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    dob DATE, height_cm REAL, weight_kg REAL
);
CREATE TABLE roster_registrations (
    registration_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL, team_id INTEGER NOT NULL,
    gender TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    week_label TEXT NOT NULL, week_start_date DATE,
    jersey_number INTEGER, position TEXT,
    source TEXT NOT NULL CHECK (source IN ('match_page', 'backfill')),
    FOREIGN KEY (player_id) REFERENCES players (player_id),
    FOREIGN KEY (team_id, gender) REFERENCES teams (team_id, gender),
    UNIQUE (player_id, team_id, gender, week_label)
);
CREATE TABLE player_match_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    registration_id INTEGER NOT NULL, match_date DATE, opponent TEXT,
    sets_played INTEGER, attack_total INTEGER, attack_points INTEGER,
    block_points INTEGER, serve_total INTEGER, serve_points INTEGER,
    receive_total INTEGER, receive_excellent INTEGER, dig_total INTEGER,
    dig_excellent INTEGER, set_total INTEGER, set_excellent INTEGER,
    total_points INTEGER, is_golden_set INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (registration_id) REFERENCES roster_registrations (registration_id)
);
CREATE TABLE matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL,
    gender TEXT NOT NULL, match_date DATE NOT NULL, round_name TEXT,
    home_team TEXT NOT NULL, away_team TEXT NOT NULL,
    UNIQUE (game_id, gender)
);
CREATE INDEX idx_roster_player      ON roster_registrations(player_id);
CREATE INDEX idx_roster_team_gender ON roster_registrations(team_id, gender);
CREATE INDEX idx_roster_week        ON roster_registrations(week_label);
"""


def _seed_old_db(tmp_db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_V2_OLD)
    conn.execute("INSERT INTO teams VALUES (5, '新北中纖', 'F')")
    conn.execute("INSERT INTO players (name, gender) VALUES ('張瓈文', 'F')")
    conn.execute(
        "INSERT INTO roster_registrations "
        "(registration_id, player_id, team_id, gender, week_label, week_start_date, jersey_number, position, source) "
        "VALUES (77, 1, 5, 'F', '例行賽 Week 1', '2025-11-01', 2, 'OP', 'match_page')"
    )
    conn.execute(
        "INSERT INTO player_match_stats (registration_id, match_date, opponent, total_points) "
        "VALUES (77, '2025-11-01', '義力營造', 20)"
    )
    conn.commit()
    return conn


def test_migration_adds_cup_id_and_preserves_registration_id(tmp_db_path):
    conn = _seed_old_db(tmp_db_path)
    try:
        result = run_migration(conn, cup_id=21)

        row = conn.execute(
            "SELECT registration_id, cup_id, week_label, jersey_number FROM roster_registrations"
        ).fetchone()
        assert row == (77, 21, "例行賽 Week 1", 2), "registration_id 必須原值保留、cup_id 全補 21"
        assert result["registrations_migrated"] == 1

        fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        assert fk_errors == [], "遷移後不得有外鍵孤兒"

        stat = conn.execute("SELECT registration_id FROM player_match_stats").fetchone()
        assert stat == (77,)
    finally:
        conn.close()


def test_migration_recreates_indexes(tmp_db_path):
    conn = _seed_old_db(tmp_db_path)
    try:
        run_migration(conn, cup_id=21)
        idx = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='roster_registrations'"
                " AND name LIKE 'idx_%'"
            )
        }
        assert {"idx_roster_player", "idx_roster_team_gender", "idx_roster_week"} <= idx
    finally:
        conn.close()


def test_migration_refuses_rerun(tmp_db_path):
    conn = _seed_old_db(tmp_db_path)
    try:
        run_migration(conn, cup_id=21)
        with pytest.raises(RuntimeError, match="已有 cup_id"):
            run_migration(conn, cup_id=21)
    finally:
        conn.close()
