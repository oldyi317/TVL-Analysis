from pathlib import Path

import pytest


@pytest.fixture
def tmp_db_path(tmp_path) -> Path:
    """回傳隔離的 SQLite 檔案路徑，測試用，不觸碰正式 DB。"""
    return tmp_path / "test.db"
