"""
一次性遷移：players 移除 dob 欄（ADR 0006，生日不公開）。
SQLite 3.35+ 支援 ALTER TABLE DROP COLUMN，不需重建表；執行前自動備份（backup_db.py）。
"""

import sqlite3

from src.etl.backup_db import backup_database
from src.utils.db_config import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _assert_not_migrated(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(players)")}
    if "dob" not in cols:
        raise RuntimeError("players 已無 dob 欄，請勿重複執行。")


def run_migration(conn: sqlite3.Connection) -> dict:
    _assert_not_migrated(conn)
    expected = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]

    original_autocommit = conn.autocommit
    conn.autocommit = False
    try:
        conn.execute("ALTER TABLE players DROP COLUMN dob")
        actual = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        if actual != expected:
            raise RuntimeError(f"筆數不符：預期 {expected}，實際 {actual}，已回滾，資料庫維持遷移前狀態。")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = original_autocommit

    logger.info("dob 遷移完成：players %d 筆，dob 欄已移除", actual)
    return {"players": actual}


def main():
    backup_database()
    conn = get_connection()
    try:
        result = run_migration(conn)
        print("\n===== dob 遷移完成 =====")
        for k, v in result.items():
            print(f"{k}: {v}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
