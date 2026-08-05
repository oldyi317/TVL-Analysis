import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import text

from src.etl.migrate_to_postgres import migrate

OLD_SCHEMA = """
CREATE TABLE teams (
    team_id INTEGER NOT NULL, team_name TEXT NOT NULL, gender TEXT NOT NULL,
    PRIMARY KEY (team_id, gender)
);
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT, team_id INTEGER NOT NULL, gender TEXT NOT NULL,
    jersey_number INTEGER, name TEXT, position TEXT, dob DATE, height_cm REAL, weight_kg REAL
);
CREATE TABLE player_match_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT, player_id INTEGER NOT NULL,
    match_date DATE, opponent TEXT, sets_played INTEGER,
    attack_total INTEGER, attack_points INTEGER, block_points INTEGER,
    serve_total INTEGER, serve_points INTEGER, receive_total INTEGER, receive_excellent INTEGER,
    dig_total INTEGER, dig_excellent INTEGER, set_total INTEGER, set_excellent INTEGER,
    total_points INTEGER, is_golden_set INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL, gender TEXT NOT NULL,
    match_date DATE NOT NULL, venue TEXT, round_name TEXT, game_label TEXT,
    is_golden_set INTEGER NOT NULL DEFAULT 0,
    home_team TEXT NOT NULL, away_team TEXT NOT NULL,
    home_set1 INTEGER, home_set2 INTEGER, home_set3 INTEGER, home_set4 INTEGER, home_set5 INTEGER, home_total INTEGER,
    away_set1 INTEGER, away_set2 INTEGER, away_set3 INTEGER, away_set4 INTEGER, away_set5 INTEGER, away_total INTEGER,
    home_sets_won INTEGER, away_sets_won INTEGER,
    UNIQUE (game_id, gender)
);
"""


def _build_old_source_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    conn.execute("INSERT INTO teams VALUES (1, '屏東台電', 'M')")
    conn.execute(
        "INSERT INTO players (team_id, gender, jersey_number, name, position, dob, height_cm, weight_kg) "
        "VALUES (1, 'M', 4, '李元', 'OH', '2000-01-01', 190.0, 80.0)"
    )
    pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO player_match_stats "
        "(player_id, match_date, opponent, sets_played, attack_total, attack_points, block_points, "
        "serve_total, serve_points, receive_total, receive_excellent, dig_total, dig_excellent, "
        "set_total, set_excellent, total_points, is_golden_set) "
        "VALUES (?, '2026-01-05', '雲林美津濃', 3, 10, 5, 1, 5, 1, 5, 3, 5, 2, 0, 0, 7, 0)",
        (pid,),
    )
    conn.execute(
        "INSERT INTO matches "
        "(game_id, gender, match_date, venue, round_name, game_label, is_golden_set, "
        "home_team, away_team, home_set1, home_set2, home_set3, home_set4, home_set5, home_total, "
        "away_set1, away_set2, away_set3, away_set4, away_set5, away_total, home_sets_won, away_sets_won) "
        "VALUES (301, 'M', '2026-01-05', '台南', '例行賽 Week 5', 'Game 301', 0, "
        "'屏東台電', '雲林美津濃', 25, 25, 25, NULL, NULL, 75, 20, 18, 22, NULL, NULL, 60, 3, 0)"
    )
    conn.commit()
    conn.close()


def test_migrate_copies_all_tables_and_tags_season(tmp_path, sqlite_engine):
    source_path = tmp_path / "old_source.db"
    _build_old_source_db(source_path)

    counts = migrate(source_path, season="2025-26")
    assert counts == {"teams": 1, "players": 1, "player_match_stats": 1, "matches": 1}

    with sqlite_engine.begin() as conn:
        season_val = conn.execute(text("SELECT DISTINCT season FROM players")).scalar_one()
        assert season_val == "2025-26"


def test_migrate_is_idempotent_on_rerun(tmp_path, sqlite_engine):
    source_path = tmp_path / "old_source.db"
    _build_old_source_db(source_path)

    migrate(source_path, season="2025-26")
    counts2 = migrate(source_path, season="2025-26")
    assert counts2 == {"teams": 1, "players": 1, "player_match_stats": 1, "matches": 1}

    with sqlite_engine.begin() as conn:
        n_players = conn.execute(text("SELECT COUNT(*) FROM players")).scalar_one()
    assert n_players == 1


def test_migrate_handles_null_arbiter_columns(tmp_path, sqlite_engine):
    """Test that NULL values in former arbiter columns (name, match_date, opponent) don't break rerun idempotency."""
    source_path = tmp_path / "old_source_with_nulls.db"
    conn = sqlite3.connect(source_path)
    conn.executescript(OLD_SCHEMA)
    conn.execute("INSERT INTO teams VALUES (2, '新竹台元', 'F')")
    # Insert player with NULL name
    conn.execute(
        "INSERT INTO players (player_id, team_id, gender, jersey_number, name, position, dob, height_cm, weight_kg) "
        "VALUES (100, 2, 'F', 10, NULL, 'S', '1998-05-15', 175.0, 65.0)"
    )
    # Insert stats with NULL match_date and NULL opponent
    conn.execute(
        "INSERT INTO player_match_stats "
        "(stat_id, player_id, match_date, opponent, sets_played, attack_total, attack_points, block_points, "
        "serve_total, serve_points, receive_total, receive_excellent, dig_total, dig_excellent, "
        "set_total, set_excellent, total_points, is_golden_set) "
        "VALUES (200, 100, NULL, NULL, 2, 8, 4, 0, 4, 1, 4, 2, 4, 1, 0, 0, 5, 0)"
    )
    conn.commit()
    conn.close()

    # First migration
    counts1 = migrate(source_path, season="2025-26")
    assert counts1 == {"teams": 1, "players": 1, "player_match_stats": 1, "matches": 0}

    # Second migration (rerun) should succeed without IntegrityError and return same counts
    counts2 = migrate(source_path, season="2025-26")
    assert counts2 == {"teams": 1, "players": 1, "player_match_stats": 1, "matches": 0}

    # Verify no duplicates
    with sqlite_engine.begin() as c:
        n_players = c.execute(text("SELECT COUNT(*) FROM players WHERE player_id = 100")).scalar_one()
        assert n_players == 1
        n_stats = c.execute(text("SELECT COUNT(*) FROM player_match_stats WHERE stat_id = 200")).scalar_one()
        assert n_stats == 1


def test_migrate_raises_when_target_equals_source(tmp_path, monkeypatch):
    """目標 DATABASE_URL 指向與來源相同的 sqlite 檔案時應拒絕執行，避免讀寫同一檔案。"""
    import src.utils.db_config as db_config

    source_path = tmp_path / "same_file.db"
    _build_old_source_db(source_path)

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{source_path}")
    db_config.reset_engine()
    try:
        with pytest.raises(RuntimeError, match="目標不可與來源相同"):
            migrate(source_path, season="2025-26")
    finally:
        db_config.reset_engine()
        monkeypatch.delenv("DATABASE_URL", raising=False)
