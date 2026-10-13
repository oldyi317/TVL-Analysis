import sqlite3
from pathlib import Path

import pytest

from src.etl.migrate_add_team_identity import run_migration

# 遷移前的 teams 等表（無新欄位、無 team_aliases）——固定快照，不隨 schema.sql 演進
SCHEMA_OLD = """
CREATE TABLE teams (
    team_id   INTEGER NOT NULL,
    team_name TEXT    NOT NULL,
    gender    TEXT    NOT NULL CHECK (gender IN ('M', 'F')),
    PRIMARY KEY (team_id, gender)
);
CREATE TABLE players (
    player_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT,
    gender     TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    dob        DATE, height_cm REAL, weight_kg REAL
);
CREATE TABLE roster_registrations (
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
CREATE TABLE seasons (
    cup_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
    start_date DATE NOT NULL, end_date DATE NOT NULL
);
"""

OLD_TEAMS = [
    (3, "臺北鯨華", "F"), (4, "高雄台電", "F"), (5, "新北中纖", "F"), (7, "義力營造", "F"),
    (1, "屏東台電", "M"), (2, "雲林美津濃", "M"), (4, "臺北國北獅", "M"),
    (5, "桃園臺產", "M"), (7, "獅子王", "M"),
]


def _seed_old_db(tmp_db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript(SCHEMA_OLD)
    conn.executemany("INSERT INTO teams (team_id, team_name, gender) VALUES (?,?,?)", OLD_TEAMS)
    conn.execute("INSERT INTO players (name, gender) VALUES ('測試員', 'F')")
    conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, cup_id, week_label, source) "
        "VALUES (1, 3, 'F', 21, 'W1', 'backfill')"
    )
    conn.commit()
    return conn


def _count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_migration_adds_columns_and_seeds(tmp_db_path):
    conn = _seed_old_db(tmp_db_path)
    try:
        result = run_migration(conn)

        cols = {r[1] for r in conn.execute("PRAGMA table_info(teams)")}
        assert {"short_name", "official_name", "color", "is_active"} <= cols
        assert _count(conn, "teams") == 12
        assert _count(conn, "team_aliases") == 32
        assert result == {"teams_before": 9, "teams_after": 12, "aliases": 32}

        legacy = conn.execute("SELECT team_id FROM teams WHERE is_active = 0 ORDER BY team_id").fetchall()
        assert legacy == [(8,), (9,), (10,)]
        assert conn.execute("SELECT COUNT(*) FROM teams WHERE is_active = 1").fetchone()[0] == 9

        names = set(conn.execute("SELECT team_id, team_name, gender FROM teams").fetchall())
        assert set(OLD_TEAMS) <= names
        assert conn.execute("SELECT COUNT(*) FROM teams WHERE short_name IS NULL").fetchone()[0] == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_roster_registrations_still_join_teams(tmp_db_path):
    conn = _seed_old_db(tmp_db_path)
    try:
        run_migration(conn)
        row = conn.execute(
            "SELECT t.short_name FROM roster_registrations r "
            "JOIN teams t ON t.team_id = r.team_id AND t.gender = r.gender"
        ).fetchone()
        assert row == ("臺北鯨華",)
    finally:
        conn.close()


def test_migration_refuses_rerun(tmp_db_path):
    conn = _seed_old_db(tmp_db_path)
    try:
        run_migration(conn)
        with pytest.raises(RuntimeError, match="short_name"):
            run_migration(conn)
        assert _count(conn, "teams") == 12
        assert _count(conn, "team_aliases") == 32
    finally:
        conn.close()


def test_migration_rolls_back_on_verification_failure(tmp_db_path):
    conn = _seed_old_db(tmp_db_path)
    conn.execute("INSERT INTO teams (team_id, team_name, gender) VALUES (99, '多出來的隊', 'M')")
    conn.commit()
    try:
        with pytest.raises(RuntimeError, match="已回滾"):
            run_migration(conn)

        cols = {r[1] for r in conn.execute("PRAGMA table_info(teams)")}
        assert "short_name" not in cols
        assert _count(conn, "teams") == 10
        assert _count(conn, "team_aliases") == 0
    finally:
        conn.close()
