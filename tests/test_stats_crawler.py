from sqlalchemy import text

from src.etl.db_loader import insert_players, insert_teams
from src.etl.stats_crawler import build_name_to_pid, parse_match_date, upsert_stats
import pandas as pd


def _sample_stat_row(**overrides) -> dict:
    base = dict(
        match_date="2026-01-05", opponent="雲林美津濃", sets_played=3,
        attack_total=10, attack_points=5, block_points=1,
        serve_total=5, serve_points=1, receive_total=5, receive_excellent=3,
        dig_total=5, dig_excellent=2, set_total=0, set_excellent=0,
        total_points=7, is_golden_set=0,
    )
    base.update(overrides)
    return base


def _seed_player(engine, season="2025-26") -> int:
    df = pd.DataFrame([{
        "team_id": 1, "team_name": "屏東台電", "gender": "M",
        "jersey_number": 4, "name": "李元", "position": "OH",
        "dob": None, "height_cm": None, "weight_kg": None,
    }])
    insert_teams(engine, df)
    insert_players(engine, df, season=season)
    with engine.begin() as conn:
        return conn.execute(
            text("SELECT player_id FROM players WHERE name = '李元' AND season = :s"),
            {"s": season},
        ).scalar_one()


def test_parse_match_date_uses_season_year_for_month():
    assert parse_match_date("311/01") == "2025-11-01"  # 11 月屬賽季起始年
    assert parse_match_date("303/05") == "2026-03-05"  # 場次3、3月5日；3 月屬賽季結束年


def test_build_name_to_pid_scoped_to_season(sqlite_engine):
    pid = _seed_player(sqlite_engine, season="2025-26")
    name_map = build_name_to_pid(sqlite_engine, "2025-26")
    assert name_map["李元"] == pid

    other_season_map = build_name_to_pid(sqlite_engine, "2026-27")
    assert "李元" not in other_season_map


def test_upsert_stats_is_idempotent_on_rerun(sqlite_engine):
    pid = _seed_player(sqlite_engine)
    row = _sample_stat_row()

    upsert_stats(sqlite_engine, pid, [row], "2025-26")
    upsert_stats(sqlite_engine, pid, [row], "2025-26")

    with sqlite_engine.begin() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM player_match_stats")).scalar_one()
    assert n == 1, f"重跑相同資料應仍是 1 筆，實際為 {n}"


def test_upsert_stats_updates_corrected_values(sqlite_engine):
    pid = _seed_player(sqlite_engine)
    row = _sample_stat_row()
    upsert_stats(sqlite_engine, pid, [row], "2025-26")

    corrected = _sample_stat_row(total_points=9)
    upsert_stats(sqlite_engine, pid, [corrected], "2025-26")

    with sqlite_engine.begin() as conn:
        pts = conn.execute(text("SELECT total_points FROM player_match_stats")).scalar_one()
    assert pts == 9


def test_upsert_stats_does_not_touch_other_season_rows(sqlite_engine):
    pid = _seed_player(sqlite_engine)
    row = _sample_stat_row()
    upsert_stats(sqlite_engine, pid, [row], "2025-26")

    upsert_stats(sqlite_engine, pid, [row], "2026-27")

    with sqlite_engine.begin() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM player_match_stats")).scalar_one()
        old_pts = conn.execute(
            text("SELECT total_points FROM player_match_stats WHERE season = '2025-26'")
        ).scalar_one()
    assert total == 2
    assert old_pts == 7
