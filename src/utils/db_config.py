"""
資料庫連線設定模組
以 DATABASE_URL 環境變數建立 SQLAlchemy engine；未設定時 fallback 至本地 SQLite。
"""

import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "db" / "tvl_database.db"

_engine: Engine | None = None


def _default_sqlite_url() -> str:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DB_PATH}"


def get_engine() -> Engine:
    """
    回傳全域共用的 SQLAlchemy engine（延遲建立，僅建立一次）。
    由 DATABASE_URL 環境變數決定連線目標；未設定時 fallback 至本地 SQLite 檔案。
    """
    global _engine
    if _engine is None:
        database_url = os.environ.get("DATABASE_URL") or _default_sqlite_url()
        _engine = create_engine(database_url, future=True, pool_pre_ping=True)
        if _engine.dialect.name == "sqlite":
            @event.listens_for(_engine, "connect")
            def _enable_sqlite_foreign_keys(dbapi_conn, _record):
                dbapi_conn.execute("PRAGMA foreign_keys = ON")
    return _engine


def reset_engine() -> None:
    """重置快取的 engine（測試或切換 DATABASE_URL 後呼叫）。"""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None
