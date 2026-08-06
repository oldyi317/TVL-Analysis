"""
weekly_report_tab 改接 MLIS 後的行為：
- 未設定 MLIS 時顯示引導訊息（不再提示 Gemini API Key）
- 已設定 MLIS 時「產生 AI 戰報」按鈕會呼叫 llm_client.generate_report 並顯示結果
"""

import pandas as pd
from sqlalchemy import text
from streamlit.testing.v1 import AppTest

from src.etl.db_loader import insert_players, insert_teams
from src.etl.stats_crawler import upsert_stats


def _seed_one_match(engine) -> None:
    df = pd.DataFrame([{
        "team_id": 1, "team_name": "屏東台電", "gender": "M",
        "jersey_number": 4, "name": "李元", "position": "OH",
        "dob": None, "height_cm": None, "weight_kg": None,
    }])
    insert_teams(engine, df)
    insert_players(engine, df, season="2025-26")
    with engine.begin() as conn:
        pid = conn.execute(text("SELECT player_id FROM players WHERE name = '李元'")).scalar_one()
    row = dict(
        match_date="2026-01-05", opponent="雲林美津濃", sets_played=3,
        attack_total=10, attack_points=5, block_points=1,
        serve_total=5, serve_points=1, receive_total=5, receive_excellent=3,
        dig_total=5, dig_excellent=2, set_total=0, set_excellent=0,
        total_points=7, is_golden_set=0,
    )
    upsert_stats(engine, pid, [row], "2025-26")


def test_weekly_report_shows_guidance_when_mlis_not_configured(sqlite_engine, monkeypatch):
    for key in ("MLIS_BASE_URL", "MLIS_API_KEY", "MLIS_MODEL"):
        monkeypatch.delenv(key, raising=False)
    _seed_one_match(sqlite_engine)

    def _harness():
        import sys
        sys.path.insert(0, ".")
        from src.app.tabs import weekly_report_tab

        weekly_report_tab.render({})

    at = AppTest.from_function(_harness)
    at.run(timeout=60)

    assert not at.exception
    info_texts = [i.value for i in at.info]
    assert any("系統設定" in t for t in info_texts)


def test_weekly_report_generate_button_calls_llm_client_when_configured(sqlite_engine, monkeypatch):
    from src.app.settings_store import set_setting

    set_setting(sqlite_engine, "mlis_base_url", "http://fake-mlis.local/v1")
    set_setting(sqlite_engine, "mlis_api_key", "test-key")
    set_setting(sqlite_engine, "mlis_model", "qwen-test")
    _seed_one_match(sqlite_engine)

    import src.app.tabs.weekly_report_tab as weekly_report_tab_module
    monkeypatch.setattr(
        weekly_report_tab_module, "generate_report",
        lambda config, system_prompt, user_prompt: "模擬產生的戰報內容",
    )

    def _harness():
        import sys
        sys.path.insert(0, ".")
        from src.app.tabs import weekly_report_tab

        weekly_report_tab.render({})

    at = AppTest.from_function(_harness)
    at.run(timeout=60)
    assert not at.exception

    buttons = [b for b in at.button if b.label == "產生 AI 戰報"]
    assert buttons, "已設定 MLIS 時應顯示「產生 AI 戰報」按鈕"
    buttons[0].click().run(timeout=60)

    markdown_texts = [m.value for m in at.markdown]
    assert any("模擬產生的戰報內容" in t for t in markdown_texts)
