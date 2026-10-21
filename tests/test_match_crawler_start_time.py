import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from src.etl.match_crawler import scrape_match_page, upsert_match
from src.utils.teams import apply_team_seed

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


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


def _scrape(fixture_name):
    html = (FIXTURES / fixture_name).read_text(encoding="utf-8")
    with patch("src.etl.match_crawler.requests.get", return_value=_FakeResponse(html)):
        return scrape_match_page("game", 301)


def _db(conn):
    return conn.execute(
        "SELECT start_time, venue, home_sets_won, away_sets_won FROM matches WHERE game_id = 301 AND gender = 'M'"
    ).fetchone()


def test_played_fixture_parses_and_stores_start_time(conn):
    match = _scrape("match_game_sample.html")
    assert match["start_time"] == "16:30"
    upsert_match(conn, match)
    assert _db(conn) == ("16:30", "屏東體育館", 3, 0)


def test_unplayed_fixture_sets_won_is_null(conn):
    match = _scrape("match_game_unplayed_sample.html")
    assert match["home_sets_won"] is None
    assert match["away_sets_won"] is None
    upsert_match(conn, match)
    start_time, _, home_won, away_won = _db(conn)
    assert start_time == "16:30"
    assert home_won is None
    assert away_won is None


def test_update_overwrites_zero_sets_won_with_null(conn):
    match = _scrape("match_game_unplayed_sample.html")
    upsert_match(conn, {**match, "home_sets_won": 0, "away_sets_won": 0})
    assert _db(conn)[2:] == (0, 0)
    assert upsert_match(conn, match) is False
    assert _db(conn)[2:] == (None, None)


def test_partial_sets_still_counted(conn):
    html = (FIXTURES / "match_game_sample.html").read_text(encoding="utf-8")
    html = html.replace("<td>25</td><td>25</td><td>25</td>", "<td>25</td><td></td><td></td>")
    html = html.replace("<td>20</td><td>20</td><td>20</td>", "<td>20</td><td></td><td></td>")
    with patch("src.etl.match_crawler.requests.get", return_value=_FakeResponse(html)):
        partial = scrape_match_page("game", 301)
    assert (partial["home_sets_won"], partial["away_sets_won"]) == (1, 0)


def test_update_keeps_existing_start_time_and_venue(conn):
    match = _scrape("match_game_sample.html")
    upsert_match(conn, match)
    assert upsert_match(conn, {**match, "start_time": None, "venue": None}) is False
    assert _db(conn)[:2] == ("16:30", "屏東體育館")


def test_update_fills_null_start_time_and_venue(conn):
    match = _scrape("match_game_sample.html")
    upsert_match(conn, {**match, "start_time": None, "venue": None})
    assert _db(conn)[:2] == (None, None)
    upsert_match(conn, match)
    assert _db(conn)[:2] == ("16:30", "屏東體育館")
