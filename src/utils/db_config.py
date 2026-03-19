"""
資料庫連線設定模組
統一管理 DB 路徑與連線建立，避免各模組重複定義。
"""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "db" / "tvl_database.db"


def get_connection(foreign_keys: bool = True) -> sqlite3.Connection:
    """
    建立並回傳 SQLite 連線。
    預設啟用 PRAGMA foreign_keys。
    呼叫端負責關閉連線。
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    if foreign_keys:
        conn.execute("PRAGMA foreign_keys = ON")
    return conn
