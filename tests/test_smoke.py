import sqlite3
from pathlib import Path

from src.utils.db_config import get_connection


def test_get_connection_enables_foreign_keys():
    conn = get_connection()
    try:
        row = conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1
    finally:
        conn.close()


def test_schema_sql_is_valid_sqlite():
    schema_path = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    conn = sqlite3.connect(":memory:")
    conn.executescript(sql)
    # 冪等檢查：同一連線重複執行一次，確認 DDL 不會因表已存在而報錯
    conn.executescript(sql)
    conn.close()
