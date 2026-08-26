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
