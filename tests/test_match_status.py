import sqlite3

import pytest

from src.utils.match_status import HAS_RESULT_WHERE, UNPLAYED_WHERE, has_result
from tests.conftest import SCHEMA_SQL


@pytest.mark.parametrize(
    "home, away, expected",
    [(3, 1, True), (None, 1, False), (3, None, False), (None, None, False), (float("nan"), 1.0, False)],
)
def test_has_result(home, away, expected):
    assert has_result(home, away) is expected


@pytest.fixture
def conn():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
    rows = [
        (1, None, None, None, None),
        (2, 25, 20, None, None),
        (3, 25, 20, 3, 1),
    ]
    for game_id, h1, a1, hsw, asw in rows:
        conn.execute(
            "INSERT INTO matches (game_id, gender, cup_id, match_date, home_team, away_team, "
            "home_set1, away_set1, home_sets_won, away_sets_won) VALUES (?, 'M', 22, '2026-01-01', 'A', 'B', ?, ?, ?, ?)",
            (game_id, h1, a1, hsw, asw),
        )
    return conn


def _game_ids(conn, where):
    return [g for (g,) in conn.execute(f"SELECT game_id FROM matches WHERE {where} ORDER BY game_id")]


def test_unplayed_where_counts_in_progress_as_played(conn):
    assert _game_ids(conn, UNPLAYED_WHERE) == [1]


def test_has_result_where_excludes_in_progress(conn):
    assert _game_ids(conn, HAS_RESULT_WHERE) == [3]
