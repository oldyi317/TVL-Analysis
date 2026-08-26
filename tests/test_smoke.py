import sqlite3

from src.utils.db_config import get_connection


def test_get_connection_enables_foreign_keys():
    conn = get_connection()
    try:
        row = conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1
    finally:
        conn.close()


def test_schema_sql_is_valid_sqlite():
    from pathlib import Path

    schema_path = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
    conn = sqlite3.connect(":memory:")
    conn.executescript(schema_path.read_text(encoding="utf-8"))
    # 能無錯執行到底即代表語法正確
    conn.close()
