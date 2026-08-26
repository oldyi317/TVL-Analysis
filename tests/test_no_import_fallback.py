from pathlib import Path

ETL_DIR = Path(__file__).resolve().parents[1] / "src" / "etl"
APP_DIR = Path(__file__).resolve().parents[1] / "src" / "app"

FILES_TO_CHECK = [
    ETL_DIR / "db_loader.py",
    ETL_DIR / "crawler.py",
    ETL_DIR / "cleaner.py",
    ETL_DIR / "match_crawler.py",
    ETL_DIR / "stats_crawler.py",
    APP_DIR / "helpers.py",
]


def test_no_module_not_found_fallback():
    for path in FILES_TO_CHECK:
        source = path.read_text(encoding="utf-8")
        assert "ModuleNotFoundError" not in source, f"{path} 仍有 fallback"


def test_crawler_has_no_dead_fallback_block():
    source = (ETL_DIR / "crawler.py").read_text(encoding="utf-8")
    assert "if TEAM_NAME_SHORT is None" not in source
