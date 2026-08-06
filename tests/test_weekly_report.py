from sqlalchemy import text

from src.etl.db_loader import insert_players, insert_teams
from src.etl.stats_crawler import upsert_stats
from src.etl.weekly_report import gather_weekly_data, get_match_weeks
import pandas as pd


def _seed(engine) -> int:
    df = pd.DataFrame([{
        "team_id": 1, "team_name": "屏東台電", "gender": "M",
        "jersey_number": 4, "name": "李元", "position": "OH",
        "dob": None, "height_cm": None, "weight_kg": None,
    }])
    insert_teams(engine, df)
    insert_players(engine, df, season="2025-26")
    with engine.begin() as conn:
        pid = conn.execute(
            text("SELECT player_id FROM players WHERE name = '李元'")
        ).scalar_one()

    row = dict(
        match_date="2026-01-05", opponent="雲林美津濃", sets_played=3,
        attack_total=10, attack_points=5, block_points=1,
        serve_total=5, serve_points=1, receive_total=5, receive_excellent=3,
        dig_total=5, dig_excellent=2, set_total=0, set_excellent=0,
        total_points=7, is_golden_set=0,
    )
    upsert_stats(engine, pid, [row], "2025-26")
    return pid


def test_get_match_weeks_returns_week_ranges(sqlite_engine):
    _seed(sqlite_engine)
    weeks = get_match_weeks("2025-26")
    assert weeks == [("2026-01-05", "2026-01-05")]


def test_gather_weekly_data_filters_by_date_range(sqlite_engine):
    _seed(sqlite_engine)
    result = gather_weekly_data("2026-01-01", "2026-01-10", "2025-26")
    assert result["period"] == "2026-01-01 ~ 2026-01-10"
    assert len(result["matches"]) == 1
    assert result["matches"][0]["team_name"] == "屏東台電"
    assert result["matches"][0]["opponent"] == "雲林美津濃"


def test_gather_weekly_data_filters_by_gender(sqlite_engine):
    _seed(sqlite_engine)
    result_f = gather_weekly_data("2026-01-01", "2026-01-10", "2025-26", gender_filter="F")
    assert result_f["matches"] == []

    result_m = gather_weekly_data("2026-01-01", "2026-01-10", "2025-26", gender_filter="M")
    assert len(result_m["matches"]) == 1
