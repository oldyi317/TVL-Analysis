from pathlib import Path

MAIN_PY = Path(__file__).resolve().parents[1] / "src" / "app" / "main.py"


def test_weekly_report_tab_removed():
    source = MAIN_PY.read_text(encoding="utf-8")
    assert "weekly_report" not in source


def test_five_tabs_declared():
    source = MAIN_PY.read_text(encoding="utf-8")
    assert "tab1, tab2, tab3, tab4, tab5 = st.tabs" in source
    assert "tab6" not in source


def test_weekly_report_files_deleted():
    root = MAIN_PY.resolve().parents[3]
    assert not (root / "src" / "etl" / "weekly_report.py").exists()
    assert not (root / "src" / "app" / "tabs" / "weekly_report_tab.py").exists()


def test_sidebar_uses_get_current_roster_not_raw_players_query():
    source = MAIN_PY.read_text(encoding="utf-8")
    assert "get_current_roster" in source
    assert "FROM players \"" not in source  # 舊的直接查 players.team_id 寫法已移除
