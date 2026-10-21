import sqlite3
from pathlib import Path

import pytest

from src.etl.match_crawler import init_matches_table, resolve_cup_id, upsert_match

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"

MATCH = {
    "game_id": 301, "gender": "F", "match_date": "2025-12-01", "venue": None,
    "round_name": "例行賽 Week 1", "game_label": "Game 1", "is_golden_set": 0,
    "home_team": "新北中纖", "away_team": "義力營造",
    "home_set1": 25, "home_set2": 25, "home_set3": 25, "home_set4": None, "home_set5": None,
    "home_total": 75, "away_set1": 20, "away_set2": 20, "away_set3": 20,
    "away_set4": None, "away_set5": None, "away_total": 60,
    "home_sets_won": 3, "away_sets_won": 0,
}


@pytest.fixture
def conn(tmp_db_path):
    connection = sqlite3.connect(tmp_db_path)
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    yield connection
    connection.close()


def _cup_id(conn):
    return conn.execute("SELECT cup_id FROM matches WHERE game_id = 301 AND gender = 'F'").fetchone()[0]


def test_upsert_insert_sets_cup_id(conn):
    assert upsert_match(conn, dict(MATCH)) is True
    assert _cup_id(conn) == 21


def test_upsert_update_refreshes_cup_id(conn):
    upsert_match(conn, dict(MATCH))
    assert upsert_match(conn, {**MATCH, "match_date": "2025-01-10"}) is False
    assert _cup_id(conn) == 20


def test_resolve_cup_id_unknown_season_raises(conn):
    with pytest.raises(RuntimeError, match="seasons"):
        resolve_cup_id(conn, "2023-01-01")


def test_init_refuses_unmigrated_matches_table(tmp_db_path):
    connection = sqlite3.connect(tmp_db_path)
    connection.execute(
        "CREATE TABLE matches (match_id INTEGER PRIMARY KEY, game_id INTEGER NOT NULL, "
        "gender TEXT NOT NULL, match_date DATE NOT NULL, home_team TEXT, away_team TEXT)"
    )
    try:
        with pytest.raises(RuntimeError, match="migrate_add_season"):
            init_matches_table(connection)
    finally:
        connection.close()


def test_resolve_cup_id_covers_2026_27_season(conn):
    """2026-27 新季（官網標題「企業22年」，cup_id 即屆數）開幕日 2026-10-24 須解析到 22。"""
    assert resolve_cup_id(conn, "2026-10-24") == 22
