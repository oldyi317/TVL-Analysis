import sqlite3
from pathlib import Path

import pytest

from src.etl.migrate_add_match_team_id import run_migration

SCHEMA_OLD = """
CREATE TABLE teams (
    team_id   INTEGER NOT NULL,
    team_name TEXT    NOT NULL,
    gender    TEXT    NOT NULL CHECK (gender IN ('M', 'F')),
    PRIMARY KEY (team_id, gender)
);
CREATE TABLE team_aliases (
    alias    TEXT    PRIMARY KEY,
    team_id  INTEGER NOT NULL,
    gender   TEXT    NOT NULL CHECK (gender IN ('M', 'F')),
    source   TEXT    NOT NULL,
    FOREIGN KEY (team_id, gender) REFERENCES teams (team_id, gender)
);
CREATE TABLE seasons (
    cup_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
    start_date DATE NOT NULL, end_date DATE NOT NULL
);
CREATE TABLE matches (
    match_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id       INTEGER NOT NULL,
    gender        TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    cup_id        INTEGER NOT NULL REFERENCES seasons (cup_id),
    match_date    DATE NOT NULL,
    home_team     TEXT NOT NULL,
    away_team     TEXT NOT NULL,
    UNIQUE (game_id, gender)
);
"""


def _seed_old_db(tmp_db_path: Path, matches: list[tuple]) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript(SCHEMA_OLD)
    conn.executemany("INSERT INTO teams VALUES (?,?,?)", [
        (5, "新北中纖", "F"), (7, "義力營造", "F"), (1, "屏東台電", "M"),
    ])
    conn.executemany("INSERT INTO team_aliases VALUES (?,?,?,?)", [
        ("新北中纖", 5, "F", "short"), ("義力營造", 7, "F", "short"),
        ("屏東台電", 1, "M", "short"), ("屏東台電男排", 1, "M", "match"),
        ("5", 5, "F", "ext_stats"),
    ])
    conn.execute("INSERT INTO seasons VALUES (21, '2025-26', '2025-07-01', '2026-06-30')")
    conn.executemany(
        "INSERT INTO matches (game_id, gender, cup_id, match_date, home_team, away_team) "
        "VALUES (?,?,21,'2025-12-01',?,?)",
        matches,
    )
    conn.commit()
    return conn


def _cols(conn):
    return {r[1] for r in conn.execute("PRAGMA table_info(matches)")}


def test_backfill_success(tmp_db_path):
    conn = _seed_old_db(tmp_db_path, [(1, "F", "新北中纖", "義力營造"), (2, "M", "屏東台電男排", "屏東台電")])
    try:
        result = run_migration(conn)
        assert {"home_team_id", "away_team_id"} <= _cols(conn)
        rows = conn.execute(
            "SELECT game_id, home_team, away_team, home_team_id, away_team_id FROM matches ORDER BY game_id"
        ).fetchall()
        assert rows == [(1, "新北中纖", "義力營造", 5, 7), (2, "屏東台電男排", "屏東台電", 1, 1)]
        assert result == {"matches": 2, "home_filled": 2, "away_filled": 2}
    finally:
        conn.close()


def test_unresolved_team_aborts_and_rolls_back(tmp_db_path):
    conn = _seed_old_db(tmp_db_path, [(1, "F", "新北中纖", "義力營造"), (2, "F", "不存在隊", "義力營造")])
    try:
        with pytest.raises(RuntimeError, match="不存在隊"):
            run_migration(conn)
        assert "home_team_id" not in _cols(conn)
    finally:
        conn.close()


def test_gender_mismatch_aborts_and_lists_name(tmp_db_path):
    conn = _seed_old_db(tmp_db_path, [(1, "F", "屏東台電", "義力營造")])
    try:
        with pytest.raises(RuntimeError, match="屏東台電"):
            run_migration(conn)
        assert "home_team_id" not in _cols(conn)
    finally:
        conn.close()


def test_ext_stats_alias_is_not_used_for_backfill(tmp_db_path):
    conn = _seed_old_db(tmp_db_path, [(1, "F", "5", "義力營造")])
    try:
        with pytest.raises(RuntimeError, match="5（F）"):
            run_migration(conn)
        assert "home_team_id" not in _cols(conn)
    finally:
        conn.close()


def test_rerun_raises(tmp_db_path):
    conn = _seed_old_db(tmp_db_path, [(1, "F", "新北中纖", "義力營造")])
    try:
        run_migration(conn)
        with pytest.raises(RuntimeError, match="home_team_id"):
            run_migration(conn)
    finally:
        conn.close()
