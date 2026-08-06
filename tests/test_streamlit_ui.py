"""
Streamlit UI 正確性測試：驗證 7 個 tab 的 st.stop() → return 修正。
st.stop() 會中止整份 script，害後面所有分頁空白；改用 return 只結束當前 tab。
使用 AppTest 模擬渲染，只驗證各檔案「第一個」空資料早退分支，
不追求窮舉每一處 st.stop() 的分支（沿用計畫一「不追求全面覆蓋」的測試策略）。
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

MAIN_PY_PATH = Path(__file__).resolve().parent.parent / "src" / "app" / "main.py"


def test_main_app_smoke_survives_missing_app_settings_table(sqlite_engine):
    """main.py 冒煙測試，對應本輪「Critical」發現的迴歸測試。

    真實情境：v2 DB 由 ETL 的 init_db 建立，但 app_settings 表是後來才加入
    schema.sql 的，既有 DB 從未重跑 init_db，dashboard 本身也不會呼叫
    init_db，所以線上 DB 可能永遠沒有 app_settings 表。修正前，weekly_report_tab
    透過 resolve_llm_config 讀取設定時會讓 OperationalError 一路往上炸穿整個
    script，導致 7 個分頁全部空白；修正後 get_setting 應自癒回傳 None。

    這裡先用 sqlite_engine（跑過完整 init_db）灌好球隊/球員/比賽資料，
    再手動 DROP app_settings 模擬「schema 較舊、缺這張表」的既有 DB。
    """
    import pandas as pd
    from sqlalchemy import text

    from src.etl.db_loader import insert_players, insert_teams
    from src.etl.stats_crawler import upsert_stats
    from src.utils.constants import SEASON

    df = pd.DataFrame([{
        "team_id": 1, "team_name": "屏東台電", "gender": "M",
        "jersey_number": 4, "name": "李元", "position": "OH",
        "dob": None, "height_cm": None, "weight_kg": None,
    }])
    insert_teams(sqlite_engine, df)
    insert_players(sqlite_engine, df, season=SEASON)
    with sqlite_engine.begin() as conn:
        pid = conn.execute(text("SELECT player_id FROM players WHERE name = '李元'")).scalar_one()
    row = dict(
        match_date="2026-01-05", opponent="雲林美津濃", sets_played=3,
        attack_total=10, attack_points=5, block_points=1,
        serve_total=5, serve_points=1, receive_total=5, receive_excellent=3,
        dig_total=5, dig_excellent=2, set_total=0, set_excellent=0,
        total_points=7, is_golden_set=0,
    )
    upsert_stats(sqlite_engine, pid, [row], SEASON)

    # 模擬「較舊、缺 app_settings 表」的既有 DB。
    with sqlite_engine.begin() as conn:
        conn.execute(text("DROP TABLE app_settings"))

    at = AppTest.from_file(str(MAIN_PY_PATH))
    at.run(timeout=60)

    assert not at.exception, f"main.py 渲染時發生例外：{at.exception}"
    tab_labels = [tab.label for tab in at.tabs]
    assert len(tab_labels) == 7


def _assert_returns_without_stopping_script(harness) -> None:
    at = AppTest.from_function(harness)
    at.run(timeout=60)
    assert not at.exception, f"渲染時發生例外：{at.exception}"
    markers = [t.value for t in at.text if t.value == "MARKER_AFTER_RENDER"]
    assert markers, "render() 應以 return 結束當前 tab，其後的程式碼仍應正常執行"


def test_box_score_empty_teams_returns_without_stopping_script(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        import streamlit as st
        from src.app.tabs import box_score

        box_score.render({"season": "2025-26"})
        st.text("MARKER_AFTER_RENDER")

    _assert_returns_without_stopping_script(_harness)


def test_league_pr_empty_league_returns_without_stopping_script(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        import streamlit as st
        from src.app.tabs import league_pr

        ctx = {
            "player_id": 1, "player_name": "測試球員", "player_position": "OH",
            "gender_code": "M", "gender": "男子組", "team_name": "測試隊",
            "season": "2025-26",
        }
        league_pr.render(ctx)
        st.text("MARKER_AFTER_RENDER")

    _assert_returns_without_stopping_script(_harness)


def test_match_trend_empty_data_returns_without_stopping_script(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        import streamlit as st
        from src.app.tabs import match_trend

        ctx = {
            "player_id": 1, "player_name": "測試球員", "player_position": "OH",
            "gender_code": "M", "gender": "男子組", "team_name": "測試隊",
        }
        match_trend.render(ctx)
        st.text("MARKER_AFTER_RENDER")

    _assert_returns_without_stopping_script(_harness)


def test_player_deep_empty_data_returns_without_stopping_script(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        import streamlit as st
        from src.app.tabs import player_deep

        ctx = {
            "player_id": 1, "player_name": "測試球員", "player_position": "OH",
            "gender_code": "M", "gender": "男子組", "season": "2025-26",
        }
        player_deep.render(ctx)
        st.text("MARKER_AFTER_RENDER")

    _assert_returns_without_stopping_script(_harness)


def test_weekly_report_no_weeks_returns_without_stopping_script(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        import streamlit as st
        from src.app.tabs import weekly_report_tab

        weekly_report_tab.render({"season": "2025-26"})
        st.text("MARKER_AFTER_RENDER")

    _assert_returns_without_stopping_script(_harness)
