from pathlib import Path

ETL_DIR = Path(__file__).resolve().parents[1] / "src" / "etl"

FILES_TO_CHECK = [
    ETL_DIR / "db_loader.py",
    ETL_DIR / "crawler.py",
    ETL_DIR / "cleaner.py",
    ETL_DIR / "match_crawler.py",
    ETL_DIR / "stats_crawler.py",
]


def test_no_module_not_found_fallback():
    for path in FILES_TO_CHECK:
        source = path.read_text(encoding="utf-8")
        assert "ModuleNotFoundError" not in source, f"{path} 仍有 fallback"


def test_crawler_has_no_dead_fallback_block():
    source = (ETL_DIR / "crawler.py").read_text(encoding="utf-8")
    assert "if TEAM_NAME_SHORT is None" not in source


def test_player_stats_store_does_not_depend_on_its_callers():
    """寫入端只能被三個寫入者依賴，不能反過來；官網寫入器不依賴已停用的歷史工具。"""
    store = (ETL_DIR / "player_stats_store.py").read_text(encoding="utf-8")
    for caller in ("stats_crawler", "season_backfill", "match_stats_writer"):
        assert caller not in store, f"player_stats_store.py 不該提到 {caller}"
    assert "stats_crawler" not in (ETL_DIR / "match_stats_writer.py").read_text(encoding="utf-8")
