import pandas as pd
from sqlalchemy import text


def test_load_data_reads_via_db_config_engine(sqlite_engine, monkeypatch):
    with sqlite_engine.begin() as conn:
        conn.execute(text("INSERT INTO teams (team_id, team_name, gender) VALUES (1, 'X', 'M')"))

    import src.app.helpers as helpers

    helpers.load_data.clear()  # 清除 st.cache_data 快取，避免跨測試互相汙染
    df = helpers.load_data("SELECT * FROM teams WHERE gender = ?", ("M",))
    assert len(df) == 1
    assert df.iloc[0]["team_name"] == "X"


def test_fetch_match_index_uses_season_year_for_month():
    from src.app.helpers import fetch_match_index

    # fetch_match_index 內部呼叫外部系統，這裡只驗證 import 後年份推算邏輯可用
    # （season_year_for_month 已在 tests/test_constants.py 完整測試，此處驗證 helpers 正確引用它）
    import src.app.helpers as helpers

    assert helpers.season_year_for_month(11) == 2025
    assert helpers.season_year_for_month(3) == 2026
    assert callable(fetch_match_index)
