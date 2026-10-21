"""
一次性遷移：teams 加 short_name／official_name／color／is_active，新增 team_aliases，
並套用 sql/team_seed.sql。既有 teams 列與 team_name 不變。執行前自動備份（backup_db.py）。
"""

import sqlite3

from src.etl.backup_db import backup_database
from src.utils.db_config import PROJECT_ROOT, get_connection
from src.utils.logger import get_logger
from src.utils.teams import apply_team_seed

logger = get_logger(__name__)
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"

# 對應本次種子（12 隊／32 別名），種子演進後此遷移不應再被執行
EXPECTED_TEAMS = 12
EXPECTED_ACTIVE = 9
EXPECTED_INACTIVE = 3
EXPECTED_ALIASES = 32


def _assert_not_migrated(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(teams)")}
    if "short_name" in cols:
        raise RuntimeError("teams 已有 short_name 欄位，遷移已套用，請勿重複執行。")


def _scalar(conn: sqlite3.Connection, sql: str) -> int:
    return conn.execute(sql).fetchone()[0]


def _verify(conn: sqlite3.Connection, before: set[tuple]) -> None:
    after = set(conn.execute("SELECT team_id, gender, team_name FROM teams").fetchall())
    checks = {
        f"teams 應共 {EXPECTED_TEAMS} 列": _scalar(conn, "SELECT COUNT(*) FROM teams") == EXPECTED_TEAMS,
        "既有 teams 列與 team_name 應不變": before <= after,
        "short_name 不得為 NULL": _scalar(conn, "SELECT COUNT(*) FROM teams WHERE short_name IS NULL") == 0,
        f"is_active=1 應共 {EXPECTED_ACTIVE} 列": _scalar(conn, "SELECT COUNT(*) FROM teams WHERE is_active = 1") == EXPECTED_ACTIVE,
        f"is_active=0 應共 {EXPECTED_INACTIVE} 列": _scalar(conn, "SELECT COUNT(*) FROM teams WHERE is_active = 0") == EXPECTED_INACTIVE,
        f"team_aliases 應共 {EXPECTED_ALIASES} 列": _scalar(conn, "SELECT COUNT(*) FROM team_aliases") == EXPECTED_ALIASES,
        "foreign_key_check 應無結果": not conn.execute("PRAGMA foreign_key_check").fetchall(),
    }
    failed = [msg for msg, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"遷移驗證失敗：{'；'.join(failed)}。已回滾，teams 資料維持遷移前狀態。")


def run_migration(conn: sqlite3.Connection) -> dict:
    _assert_not_migrated(conn)

    # executescript 會先隱式 COMMIT，須放在手動交易之前；此時只會建出 team_aliases。
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    conn.execute("PRAGMA foreign_keys = OFF")
    original_autocommit = conn.autocommit
    conn.autocommit = False
    try:
        before = set(conn.execute("SELECT team_id, gender, team_name FROM teams").fetchall())

        conn.execute("ALTER TABLE teams ADD COLUMN short_name TEXT")
        conn.execute("ALTER TABLE teams ADD COLUMN official_name TEXT")
        conn.execute("ALTER TABLE teams ADD COLUMN color TEXT")
        conn.execute("ALTER TABLE teams ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))")

        apply_team_seed(conn)
        _verify(conn, before)
        result = {
            "teams_before": len(before),
            "teams_after": _scalar(conn, "SELECT COUNT(*) FROM teams"),
            "aliases": _scalar(conn, "SELECT COUNT(*) FROM team_aliases"),
        }
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = original_autocommit
        conn.execute("PRAGMA foreign_keys = ON")

    logger.info("team identity 遷移完成：%s", result)
    return result


def main():
    backup_database()
    conn = get_connection()
    try:
        result = run_migration(conn)
        logger.info("遷移結果：%s", result)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
