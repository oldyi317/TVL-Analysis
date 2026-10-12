import sqlite3
from pathlib import Path

import pytest

from src.etl.migrate_add_season import run_migration

# 遷移前的 matches（無 cup_id／start_time）——固定快照，不隨 schema.sql 演進
SCHEMA_OLD = """
CREATE TABLE matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL,
    gender TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    match_date DATE NOT NULL, venue TEXT, round_name TEXT, game_label TEXT,
    is_golden_set INTEGER NOT NULL DEFAULT 0,
    home_team TEXT NOT NULL, away_team TEXT NOT NULL,
    home_set1 INTEGER, home_set2 INTEGER, home_set3 INTEGER, home_set4 INTEGER, home_set5 INTEGER,
    home_total INTEGER,
    away_set1 INTEGER, away_set2 INTEGER, away_set3 INTEGER, away_set4 INTEGER, away_set5 INTEGER,
    away_total INTEGER,
    home_sets_won INTEGER, away_sets_won INTEGER,
    UNIQUE (game_id, gender)
);
CREATE INDEX idx_matches_date ON matches(match_date);
"""

ROWS = [
    (1, "F", "2024-12-01", 0, "新北中纖", "義力營造", 25, 20, 75, 70, 3, 1),
    (2, "M", "2025-03-15", 0, "臺北鯨華", "高雄台電", 20, 25, 60, 75, 0, 3),
    (3, "F", "2026-01-10", 1, "新北中纖", "屏東台電", 15, 12, 15, 12, 1, 0),
]


def _seed_old_db(tmp_db_path: Path, extra_date: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript(SCHEMA_OLD)
    for game_id, gender, date, golden, home, away, h1, a1, ht, at, hw, aw in ROWS:
        conn.execute(
            "INSERT INTO matches (game_id, gender, match_date, is_golden_set, home_team, away_team, "
            "home_set1, away_set1, home_total, away_total, home_sets_won, away_sets_won) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (game_id, gender, date, golden, home, away, h1, a1, ht, at, hw, aw),
        )
    if extra_date:
        conn.execute(
            "INSERT INTO matches (game_id, gender, match_date, home_team, away_team) "
            "VALUES (99, 'M', ?, 'A', 'B')",
            (extra_date,),
        )
    conn.commit()
    return conn


def _tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_migration_assigns_cup_id_and_preserves_data(tmp_db_path):
    conn = _seed_old_db(tmp_db_path)
    try:
        result = run_migration(conn)

        assert conn.execute("SELECT cup_id FROM seasons ORDER BY cup_id").fetchall() == [(20,), (21,)]
        assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 3
        rows = conn.execute(
            """SELECT m.game_id, m.cup_id, m.match_date, s.start_date, s.end_date
               FROM matches m JOIN seasons s ON s.cup_id = m.cup_id ORDER BY m.game_id"""
        ).fetchall()
        assert [r[1] for r in rows] == [20, 20, 21]
        assert all(r[3] <= r[2] <= r[4] for r in rows)

        golden = conn.execute(
            "SELECT is_golden_set, home_set1, away_set1, home_total, away_total, home_sets_won, away_sets_won "
            "FROM matches WHERE game_id = 3"
        ).fetchone()
        assert golden == (1, 15, 12, 15, 12, 1, 0)

        assert conn.execute("SELECT COUNT(*) FROM matches WHERE start_time IS NOT NULL").fetchone()[0] == 0
        assert "matches_old" not in _tables(conn)
        assert result["matches_migrated"] == 3
        assert result["matches_per_cup"] == {20: 2, 21: 1}
    finally:
        conn.close()


def test_migrated_matches_accepts_unplayed_match(tmp_db_path):
    conn = _seed_old_db(tmp_db_path)
    try:
        run_migration(conn)
        conn.execute(
            "INSERT INTO matches (game_id, gender, cup_id, match_date, home_team, away_team, start_time) "
            "VALUES (500, 'F', 21, '2026-02-01', 'A', 'B', '19:00')"
        )
        row = conn.execute(
            "SELECT start_time, home_set1, home_total, home_sets_won FROM matches WHERE game_id = 500"
        ).fetchone()
        assert row == ("19:00", None, None, None)
    finally:
        conn.close()


def test_migration_refuses_rerun(tmp_db_path):
    conn = _seed_old_db(tmp_db_path)
    try:
        run_migration(conn)
        with pytest.raises(RuntimeError):
            run_migration(conn)
    finally:
        conn.close()


def test_migration_rolls_back_when_season_missing(tmp_db_path):
    conn = _seed_old_db(tmp_db_path, extra_date="2023-01-01")
    try:
        with pytest.raises(RuntimeError, match="2023-01-01"):
            run_migration(conn)

        cols = {r[1] for r in conn.execute("PRAGMA table_info(matches)")}
        assert "cup_id" not in cols
        assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 4
        assert "matches_old" not in _tables(conn)
        assert "seasons" not in _tables(conn)
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'idx_matches_date'"
        ).fetchone()[0] == 1
    finally:
        conn.close()
