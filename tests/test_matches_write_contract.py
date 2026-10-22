import pytest

from src.etl.match_crawler import upsert_match
from src.etl.schedule_crawler import upsert_schedule_match
from tests.conftest import connect_seeded

CUP_ID = 21

SCHEDULE = {
    "game_id": 301, "gender": "F", "match_date": "2025-12-01", "start_time": "18:30",
    "venue": "列表頁場館", "round_name": "例行賽 Week 1", "game_label": "Game 1", "is_golden_set": 0,
    "home_team": "新北中纖", "away_team": "義力營造",
}

MATCH = {
    "game_id": 301, "gender": "F", "match_date": "2025-12-01", "start_time": "10:00",
    "venue": "單場頁場館", "round_name": "例行賽 Week 1", "game_label": "Game 1", "is_golden_set": 0,
    "home_team": "新北中纖", "away_team": "義力營造",
    "home_set1": 25, "home_set2": 25, "home_set3": 25, "home_set4": None, "home_set5": None,
    "home_total": 75, "away_set1": 20, "away_set2": 20, "away_set3": 20,
    "away_set4": None, "away_set5": None, "away_total": 60,
    "home_sets_won": 3, "away_sets_won": 0,
}

SCORES = "home_set1, home_set2, home_set3, away_set1, away_set2, away_set3, home_total, away_total, home_sets_won, away_sets_won"


@pytest.fixture
def conn(tmp_db_path):
    connection = connect_seeded(tmp_db_path)
    yield connection
    connection.close()


def _select(conn, columns):
    return conn.execute(f"SELECT {columns} FROM matches WHERE game_id = 301 AND gender = 'F'").fetchone()


def test_match_with_null_round_name_keeps_schedule_values(conn):
    # 列表頁先寫入未開打賽事，單場頁解析不到 round_name／game_label 時，保留既有值並補上比分
    upsert_schedule_match(conn, dict(SCHEDULE), CUP_ID)
    upsert_match(conn, dict(MATCH, round_name=None, game_label=None))
    assert _select(conn, "round_name, game_label") == ("例行賽 Week 1", "Game 1")
    assert _select(conn, SCORES) == (25, 25, 25, 20, 20, 20, 75, 60, 3, 0)


def test_schedule_with_null_round_name_keeps_match_values(conn):
    # 單場頁先寫入含比分的賽事，列表頁缺 round_name／game_label 時，保留既有值且比分不變
    upsert_match(conn, dict(MATCH))
    upsert_schedule_match(conn, dict(SCHEDULE, round_name=None, game_label=None), CUP_ID)
    assert _select(conn, "round_name, game_label") == ("例行賽 Week 1", "Game 1")
    assert _select(conn, SCORES) == (25, 25, 25, 20, 20, 20, 75, 60, 3, 0)


def test_match_with_new_round_name_replaces_old_value(conn):
    # 單場頁解析到非 NULL 的新 round_name／game_label 時，新值取代舊值
    upsert_match(conn, dict(MATCH))
    upsert_match(conn, dict(MATCH, round_name="例行賽 Week 2", game_label="Game 2"))
    assert _select(conn, "round_name, game_label") == ("例行賽 Week 2", "Game 2")


@pytest.mark.parametrize("schedule_first", [True, False])
def test_start_time_and_venue_schedule_wins_match_fills_gaps(conn, schedule_first):
    # 任一寫入順序下，start_time／venue 都以列表頁新值為準；單場頁只在既有值為 NULL 時補缺
    if schedule_first:
        upsert_schedule_match(conn, dict(SCHEDULE), CUP_ID)
        upsert_match(conn, dict(MATCH))
    else:
        upsert_match(conn, dict(MATCH))
        upsert_schedule_match(conn, dict(SCHEDULE), CUP_ID)
    assert _select(conn, "start_time, venue") == ("18:30", "列表頁場館")
    # 既有值為 NULL 時，單場頁補缺
    conn.execute("UPDATE matches SET start_time = NULL, venue = NULL")
    upsert_match(conn, dict(MATCH))
    assert _select(conn, "start_time, venue") == ("10:00", "單場頁場館")
