import logging
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from src.etl.match_crawler import init_matches_table, scrape_match_page, upsert_match
from src.utils.teams import apply_team_seed

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "match_game_sample.html"

MATCH = {
    "game_id": 301, "gender": "F", "match_date": "2025-12-01", "venue": None,
    "round_name": "例行賽 Week 1", "game_label": "Game 1", "is_golden_set": 0,
    "home_team": "新北中纖", "away_team": "義力營造",
    "home_set1": 25, "home_set2": 25, "home_set3": 25, "home_set4": None, "home_set5": None,
    "home_total": 75, "away_set1": 20, "away_set2": 20, "away_set3": 20,
    "away_set4": None, "away_set5": None, "away_total": 60,
    "home_sets_won": 3, "away_sets_won": 0,
}


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text
        self.encoding = "utf-8"

    def raise_for_status(self):
        pass


@pytest.fixture
def conn(tmp_db_path):
    connection = sqlite3.connect(tmp_db_path)
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    apply_team_seed(connection)
    yield connection
    connection.close()


def _row(conn, game_id=301, gender="F"):
    return conn.execute(
        "SELECT home_team, away_team, home_team_id, away_team_id FROM matches WHERE game_id = ? AND gender = ?",
        (game_id, gender),
    ).fetchone()


def _team_id(conn, name, gender):
    return conn.execute(
        "SELECT team_id FROM team_aliases WHERE alias = ? AND gender = ?", (name, gender)
    ).fetchone()[0]


def test_insert_resolves_both_team_ids(conn):
    assert upsert_match(conn, dict(MATCH)) is True
    home, away, home_id, away_id = _row(conn)
    assert (home, away) == ("新北中纖", "義力營造")
    assert home_id == _team_id(conn, "新北中纖", "F")
    assert away_id == _team_id(conn, "義力營造", "F")


def test_insert_resolves_match_source_alias(conn):
    upsert_match(conn, {**MATCH, "gender": "M", "home_team": "屏東台電男排", "away_team": "雲林美津濃"})
    home, _, home_id, away_id = _row(conn, gender="M")
    assert home == "屏東台電男排"
    assert home_id == _team_id(conn, "屏東台電男排", "M")
    assert away_id == _team_id(conn, "雲林美津濃", "M")


def test_unknown_team_writes_null_and_warns(conn, caplog):
    with caplog.at_level(logging.WARNING):
        upsert_match(conn, {**MATCH, "away_team": "不存在隊"})
    home, away, home_id, away_id = _row(conn)
    assert away == "不存在隊"
    assert home_id is not None
    assert away_id is None
    assert "301" in caplog.text and "不存在隊" in caplog.text


def test_gender_mismatch_writes_null_and_warns(conn, caplog):
    with caplog.at_level(logging.WARNING):
        upsert_match(conn, {**MATCH, "home_team": "雲林美津濃"})
    _, _, home_id, away_id = _row(conn)
    assert home_id is None
    assert away_id is not None
    assert "雲林美津濃" in caplog.text


def test_update_refreshes_team_ids(conn):
    upsert_match(conn, {**MATCH, "home_team": "不存在隊"})
    assert _row(conn)[2] is None
    assert upsert_match(conn, dict(MATCH)) is False
    assert _row(conn)[2] == _team_id(conn, "新北中纖", "F")


def test_scrape_fixture_then_upsert_carries_team_ids(conn):
    html = FIXTURE.read_text(encoding="utf-8")
    with patch("src.etl.match_crawler.requests.get", return_value=_FakeResponse(html)):
        match = scrape_match_page("game", 301)
    assert match is not None
    upsert_match(conn, match)
    home, away, home_id, away_id = _row(conn, gender="M")
    assert (home, away) == ("屏東台電男排", "雲林美津濃")
    assert home_id == _team_id(conn, "屏東台電男排", "M")
    assert away_id == _team_id(conn, "雲林美津濃", "M")


def test_init_refuses_matches_without_team_id(tmp_db_path):
    connection = sqlite3.connect(tmp_db_path)
    connection.execute(
        "CREATE TABLE matches (match_id INTEGER PRIMARY KEY, game_id INTEGER NOT NULL, "
        "gender TEXT NOT NULL, cup_id INTEGER, match_date DATE NOT NULL, home_team TEXT, away_team TEXT)"
    )
    try:
        with pytest.raises(RuntimeError, match="schema 過舊（缺 home_team_id）"):
            init_matches_table(connection)
    finally:
        connection.close()
