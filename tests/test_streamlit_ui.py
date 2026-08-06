"""
Streamlit UI 正確性測試：驗證 6 個 tab 的 st.stop() → return 修正。
st.stop() 會中止整份 script，害後面所有分頁空白；改用 return 只結束當前 tab。
使用 AppTest 模擬渲染，只驗證各檔案「第一個」空資料早退分支，
不追求窮舉每一處 st.stop() 的分支（沿用計畫一「不追求全面覆蓋」的測試策略）。
"""

from streamlit.testing.v1 import AppTest


def _assert_returns_without_stopping_script(harness) -> None:
    at = AppTest.from_function(harness)
    at.run(timeout=30)
    assert not at.exception, f"渲染時發生例外：{at.exception}"
    markers = [t.value for t in at.text if t.value == "MARKER_AFTER_RENDER"]
    assert markers, "render() 應以 return 結束當前 tab，其後的程式碼仍應正常執行"


def test_box_score_empty_teams_returns_without_stopping_script(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        import streamlit as st
        from src.app.tabs import box_score

        box_score.render({})
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
            "gender_code": "M", "gender": "男子組",
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

        weekly_report_tab.render({})
        st.text("MARKER_AFTER_RENDER")

    _assert_returns_without_stopping_script(_harness)
