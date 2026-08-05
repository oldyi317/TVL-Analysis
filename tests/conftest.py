"""pytest 共用 fixtures：提供已套用最新 schema 的暫存 SQLite engine。"""

import pytest


@pytest.fixture
def sqlite_engine(tmp_path, monkeypatch):
    """建立套用最新 schema 的暫存 SQLite engine，測試結束後釋放。"""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    import src.utils.db_config as db_config

    db_config.reset_engine()

    from src.etl.db_loader import init_db

    engine = db_config.get_engine()
    init_db(engine)

    yield engine

    db_config.reset_engine()
