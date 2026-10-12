"""
一次性遷移：新增 seasons 表，matches 加 cup_id（依 match_date 落入的賽季）與 start_time。
SQLite 無法為既有表加 NOT NULL 外鍵欄，須重建表；match_id 原值保留。執行前自動備份（backup_db.py）。
"""

import sqlite3

from src.etl.backup_db import backup_database
from src.utils.db_config import PROJECT_ROOT, get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


def _assert_not_migrated(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(matches)")}
    if "cup_id" in cols:
        raise RuntimeError("matches 已有 cup_id，請勿重複執行。")


def _missing_season_dates(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("""
        SELECT DISTINCT m.match_date FROM matches_old m
        WHERE NOT EXISTS (
            SELECT 1 FROM seasons s WHERE m.match_date BETWEEN s.start_date AND s.end_date
        )
        ORDER BY m.match_date
    """).fetchall()
    return [r[0] for r in rows]


def run_migration(conn: sqlite3.Connection) -> dict:
    _assert_not_migrated(conn)
    expected = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]

    # foreign_keys 開關對 DDL 並無實質防護（見 migrate_add_cup_id），保留以免影響其他連線。
    conn.execute("PRAGMA foreign_keys = OFF")

    # 手動交易模式讓 DDL 與 DML 同屬一個交易，任何例外都能完整回滾。
    original_autocommit = conn.autocommit
    conn.autocommit = False
    try:
        conn.execute("PRAGMA legacy_alter_table = ON")
        conn.execute("ALTER TABLE matches RENAME TO matches_old")
        conn.execute("PRAGMA legacy_alter_table = OFF")
        conn.execute("DROP INDEX IF EXISTS idx_matches_date")

        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

        conn.execute("""
            INSERT INTO matches (match_id, game_id, gender, cup_id, match_date, start_time,
                venue, round_name, game_label, is_golden_set, home_team, away_team,
                home_set1, home_set2, home_set3, home_set4, home_set5, home_total,
                away_set1, away_set2, away_set3, away_set4, away_set5, away_total,
                home_sets_won, away_sets_won)
            SELECT m.match_id, m.game_id, m.gender, s.cup_id, m.match_date, NULL AS start_time,
                m.venue, m.round_name, m.game_label, m.is_golden_set, m.home_team, m.away_team,
                m.home_set1, m.home_set2, m.home_set3, m.home_set4, m.home_set5, m.home_total,
                m.away_set1, m.away_set2, m.away_set3, m.away_set4, m.away_set5, m.away_total,
                m.home_sets_won, m.away_sets_won
            FROM matches_old m
            JOIN seasons s ON m.match_date BETWEEN s.start_date AND s.end_date
        """)

        actual = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        if actual != expected:
            missing = _missing_season_dates(conn)
            raise RuntimeError(
                f"筆數不符：預期 {expected}，實際 {actual}；找不到賽季的日期：{', '.join(missing)}。"
                "已回滾，資料庫維持遷移前狀態。"
            )

        null_cup = conn.execute("SELECT COUNT(*) FROM matches WHERE cup_id IS NULL").fetchone()[0]
        if null_cup > 0:
            raise RuntimeError(f"遷移驗證失敗：{null_cup} 場 cup_id 為 NULL，已回滾，資料庫維持遷移前狀態。")

        per_cup = dict(conn.execute("SELECT cup_id, COUNT(*) FROM matches GROUP BY cup_id").fetchall())

        conn.execute("DROP TABLE matches_old")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = original_autocommit
        conn.execute("PRAGMA foreign_keys = ON")

    logger.info("seasons 遷移完成：%d 場比賽補上 cup_id，分布 %s", actual, per_cup)
    return {"matches_migrated": actual, "matches_per_cup": per_cup}


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
