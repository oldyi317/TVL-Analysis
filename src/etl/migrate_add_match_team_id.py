"""
一次性遷移（expand 步）：matches 加 home_team_id／away_team_id，並以 team_aliases 回填。
home_team／away_team 字串欄位不變。隊名解析不到或性別不符即中止回滾。執行前自動備份（backup_db.py）。
"""

import sqlite3

from src.etl.backup_db import backup_database
from src.utils.db_config import PROJECT_ROOT, get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"

SIDES = ("home", "away")


def _assert_not_migrated(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(matches)")}
    if "home_team_id" in cols:
        raise RuntimeError("matches 已有 home_team_id 欄位，遷移已套用，請勿重複執行。")


def _scalar(conn: sqlite3.Connection, sql: str) -> int:
    return conn.execute(sql).fetchone()[0]


def _unresolved(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    names: set[tuple[str, str]] = set()
    for side in SIDES:
        rows = conn.execute(
            f"""
            SELECT DISTINCT m.{side}_team, m.gender FROM matches m
            LEFT JOIN team_aliases a
              ON a.alias = TRIM(m.{side}_team) AND a.source <> 'ext_stats' AND a.gender = m.gender
            WHERE a.alias IS NULL
            """
        ).fetchall()
        names.update(rows)
    return sorted(names)


def _backfill(conn: sqlite3.Connection) -> None:
    for side in SIDES:
        conn.execute(
            f"""
            UPDATE matches SET {side}_team_id = (
                SELECT a.team_id FROM team_aliases a
                WHERE a.alias = TRIM(matches.{side}_team) AND a.source <> 'ext_stats' AND a.gender = matches.gender
            )
            """
        )


def _verify(conn: sqlite3.Connection) -> None:
    checks = {"foreign_key_check 應無結果": not conn.execute("PRAGMA foreign_key_check").fetchall()}
    for side in SIDES:
        checks[f"{side}_team_id 不得為 NULL"] = (
            _scalar(conn, f"SELECT COUNT(*) FROM matches WHERE {side}_team_id IS NULL") == 0
        )
        checks[f"{side}_team_id 須存在於 teams"] = _scalar(
            conn,
            f"""SELECT COUNT(*) FROM matches m
                LEFT JOIN teams t ON t.team_id = m.{side}_team_id AND t.gender = m.gender
                WHERE t.team_id IS NULL""",
        ) == 0
    failed = [msg for msg, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"遷移驗證失敗：{'；'.join(failed)}。已回滾，matches 維持遷移前狀態。")


def run_migration(conn: sqlite3.Connection) -> dict:
    _assert_not_migrated(conn)

    # executescript 會先隱式 COMMIT，須放在手動交易之前；matches 已存在故不受影響。
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    conn.execute("PRAGMA foreign_keys = OFF")
    original_autocommit = conn.autocommit
    conn.autocommit = False
    try:
        conn.execute("ALTER TABLE matches ADD COLUMN home_team_id INTEGER")
        conn.execute("ALTER TABLE matches ADD COLUMN away_team_id INTEGER")

        unresolved = _unresolved(conn)
        if unresolved:
            listed = "、".join(f"{name}（{gender}）" for name, gender in unresolved)
            raise RuntimeError(f"以下隊名在 team_aliases 解析不到或性別不符，遷移中止並回滾：{listed}")

        _backfill(conn)
        _verify(conn)
        result = {
            "matches": _scalar(conn, "SELECT COUNT(*) FROM matches"),
            "home_filled": _scalar(conn, "SELECT COUNT(*) FROM matches WHERE home_team_id IS NOT NULL"),
            "away_filled": _scalar(conn, "SELECT COUNT(*) FROM matches WHERE away_team_id IS NOT NULL"),
        }
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = original_autocommit
        conn.execute("PRAGMA foreign_keys = ON")

    logger.info("matches team_id 遷移完成：%s", result)
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
