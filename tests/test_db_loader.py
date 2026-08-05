import pandas as pd
from sqlalchemy import text

from src.etl.db_loader import insert_players, insert_teams


def _sample_roster() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "team_id": 1, "team_name": "屏東台電", "gender": "M",
            "jersey_number": 4, "name": "李元", "position": "OH",
            "dob": "2000-01-01", "height_cm": 190.0, "weight_kg": 80.0,
        },
        {
            "team_id": 1, "team_name": "屏東台電", "gender": "M",
            "jersey_number": 7, "name": "王小明", "position": "S",
            "dob": None, "height_cm": None, "weight_kg": None,
        },
    ])


def test_insert_players_is_idempotent_on_rerun(sqlite_engine):
    df = _sample_roster()
    insert_teams(sqlite_engine, df)
    insert_players(sqlite_engine, df, season="2025-26")
    with sqlite_engine.begin() as conn:
        n1 = conn.execute(text("SELECT COUNT(*) FROM players")).scalar_one()
    assert n1 == 2

    insert_teams(sqlite_engine, df)
    insert_players(sqlite_engine, df, season="2025-26")
    with sqlite_engine.begin() as conn:
        n2 = conn.execute(text("SELECT COUNT(*) FROM players")).scalar_one()
    assert n2 == 2, f"重跑後筆數應不變，實際為 {n2}"


def test_insert_players_updates_changed_fields_on_rerun(sqlite_engine):
    df = _sample_roster()
    insert_teams(sqlite_engine, df)
    insert_players(sqlite_engine, df, season="2025-26")

    df2 = df.copy()
    df2.loc[df2["name"] == "李元", "height_cm"] = 191.0
    insert_players(sqlite_engine, df2, season="2025-26")

    with sqlite_engine.begin() as conn:
        height = conn.execute(
            text("SELECT height_cm FROM players WHERE name = '李元'")
        ).scalar_one()
    assert height == 191.0


def test_insert_players_does_not_touch_other_season_rows(sqlite_engine):
    df = _sample_roster()
    insert_teams(sqlite_engine, df)
    insert_players(sqlite_engine, df, season="2025-26")

    insert_players(sqlite_engine, df, season="2026-27")

    with sqlite_engine.begin() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM players")).scalar_one()
        old_height = conn.execute(
            text("SELECT height_cm FROM players WHERE name = '李元' AND season = '2025-26'")
        ).scalar_one()
    assert total == 4, "兩個賽季各 2 筆，應合計 4 筆"
    assert old_height == 190.0, "舊賽季的列不應被新賽季的 upsert 觸碰"
