"""
賽季過濾迴歸測試：確保換季後，聯盟聚合與週報彙整不會把不同賽季的同名球員混在一起
（對應計畫一「已知風險 #6」：跨賽季後聯盟 PR 頁同一人出現兩筆）。
"""

import pandas as pd
from sqlalchemy import text

from src.etl.db_loader import insert_players, insert_teams
from src.etl.stats_crawler import upsert_stats
from src.etl.weekly_report import gather_weekly_data, get_match_weeks


def _seed_two_seasons(engine) -> None:
    df = pd.DataFrame([{
        "team_id": 1, "team_name": "屏東台電", "gender": "M",
        "jersey_number": 4, "name": "李元", "position": "OH",
        "dob": None, "height_cm": None, "weight_kg": None,
    }])
    insert_teams(engine, df)

    for season, match_date in [("2025-26", "2026-01-05"), ("2026-27", "2026-11-10")]:
        insert_players(engine, df, season=season)
        with engine.begin() as conn:
            pid = conn.execute(
                text("SELECT player_id FROM players WHERE name = '李元' AND season = :s"),
                {"s": season},
            ).scalar_one()
        row = dict(
            match_date=match_date, opponent="雲林美津濃", sets_played=5,
            attack_total=10, attack_points=5, block_points=1,
            serve_total=5, serve_points=1, receive_total=5, receive_excellent=3,
            dig_total=5, dig_excellent=2, set_total=0, set_excellent=0,
            total_points=7, is_golden_set=0,
        )
        upsert_stats(engine, pid, [row], season)


def test_get_league_aggregated_stats_filters_by_season(sqlite_engine):
    from src.app.helpers import get_league_aggregated_stats

    _seed_two_seasons(sqlite_engine)
    get_league_aggregated_stats.clear()

    df_2025 = get_league_aggregated_stats("M", "2025-26")
    df_2026 = get_league_aggregated_stats("M", "2026-27")

    # 每個賽季各自只看到「李元」一筆，不會因為換季而出現兩筆同名球員
    assert len(df_2025[df_2025["name"] == "李元"]) == 1
    assert len(df_2026[df_2026["name"] == "李元"]) == 1
    assert df_2025["player_id"].iloc[0] != df_2026["player_id"].iloc[0]


def test_get_match_weeks_filters_by_season(sqlite_engine):
    _seed_two_seasons(sqlite_engine)

    weeks_2025 = get_match_weeks("2025-26")
    weeks_2026 = get_match_weeks("2026-27")

    assert weeks_2025 == [("2026-01-05", "2026-01-05")]
    assert weeks_2026 == [("2026-11-10", "2026-11-10")]


def test_gather_weekly_data_filters_by_season(sqlite_engine):
    _seed_two_seasons(sqlite_engine)

    result_2025 = gather_weekly_data("2026-01-01", "2026-01-10", "2025-26")
    result_2026 = gather_weekly_data("2026-11-01", "2026-11-15", "2026-27")

    assert len(result_2025["matches"]) == 1
    assert len(result_2026["matches"]) == 1

    # 舊賽季範圍查詢新賽季時應為空（season 過濾優先於日期範圍巧合重疊的風險）
    cross_season = gather_weekly_data("2026-01-01", "2026-01-10", "2026-27")
    assert cross_season["matches"] == []
