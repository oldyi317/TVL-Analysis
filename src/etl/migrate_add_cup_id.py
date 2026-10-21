"""
Phase 3 一次性遷移：roster_registrations 加 cup_id 賽季限定鍵。
SQLite 無法 ALTER 既有 UNIQUE，須重建表；registration_id 原值保留，
player_match_stats 的外鍵不受影響。執行前自動備份（backup_db.py）。
"""

import sqlite3

from src.etl.backup_db import backup_database
from src.utils.constants import EXT_CUP_ID as CUP_ID
from src.utils.db_config import PROJECT_ROOT, get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


def _assert_not_migrated(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(roster_registrations)")}
    if "cup_id" in cols:
        raise RuntimeError("roster_registrations 已有 cup_id，請勿重複執行。")


def run_migration(conn: sqlite3.Connection, cup_id: int = CUP_ID) -> dict:
    _assert_not_migrated(conn)
    expected = conn.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()[0]

    # 這個 PRAGMA 只是語意宣告，不是安全機制：ALTER TABLE RENAME／DROP INDEX／
    # DROP TABLE 這類 DDL 在 SQLite 本來就不會回溯檢查外鍵，OFF/ON 開關對本遷移
    # 沒有實質防護作用，保留只是避免其他連線在遷移期間受到 FK 檢查影響。
    conn.execute("PRAGMA foreign_keys = OFF")

    # Python sqlite3 的隱式交易只涵蓋 DML；ALTER/DROP/executescript 這類 DDL
    # 在預設（legacy）交易模式下會逐句自動提交，導致驗證失敗時 rename、drop
    # index、建好的新表都已經永久生效，只有最後的 INSERT 被捨棄，資料庫卡在
    # 半殘狀態。改用 PEP 249 手動交易模式（autocommit = False）之後，DDL 與
    # DML 同屬一個交易，任何例外都能靠 conn.rollback() 完整復原到遷移前狀態。
    original_autocommit = conn.autocommit
    conn.autocommit = False
    try:
        # legacy 模式：RENAME 不改寫 player_match_stats 的 FK 參照名稱，
        # 舊表 drop、新表補位後，FK 仍指向 roster_registrations 本名。
        conn.execute("PRAGMA legacy_alter_table = ON")
        conn.execute("ALTER TABLE roster_registrations RENAME TO roster_registrations_old")
        conn.execute("PRAGMA legacy_alter_table = OFF")

        # 舊索引名稱仍佔用（附掛在改名後的舊表），先清掉，
        # schema.sql 的 CREATE INDEX IF NOT EXISTS 才會建到新表上。
        conn.execute("DROP INDEX IF EXISTS idx_roster_player")
        conn.execute("DROP INDEX IF EXISTS idx_roster_team_gender")
        conn.execute("DROP INDEX IF EXISTS idx_roster_week")

        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

        conn.execute(
            """
            INSERT INTO roster_registrations
                (registration_id, player_id, team_id, gender, cup_id,
                 week_label, week_start_date, jersey_number, position, source)
            SELECT registration_id, player_id, team_id, gender, ?,
                   week_label, week_start_date, jersey_number, position, source
            FROM roster_registrations_old
            """,
            (cup_id,),
        )

        actual = conn.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()[0]
        if actual != expected:
            raise RuntimeError(f"筆數不符：預期 {expected}，實際 {actual}，已回滾，資料庫維持遷移前狀態。")

        orphans = conn.execute("""
            SELECT COUNT(*) FROM player_match_stats s
            WHERE NOT EXISTS (
                SELECT 1 FROM roster_registrations r WHERE r.registration_id = s.registration_id
            )
        """).fetchone()[0]
        if orphans > 0:
            raise RuntimeError(f"遷移驗證失敗：{orphans} 筆孤兒 player_match_stats，已回滾，資料庫維持遷移前狀態。")

        conn.execute("DROP TABLE roster_registrations_old")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = original_autocommit
        conn.execute("PRAGMA foreign_keys = ON")

    logger.info("cup_id 遷移完成：%d 筆登錄補上 cup_id=%d", actual, cup_id)
    return {"registrations_migrated": actual, "cup_id": cup_id, "orphans_found": 0}


def main():
    backup_database()
    conn = get_connection()
    try:
        result = run_migration(conn)
        print("\n===== cup_id 遷移完成 =====")
        for k, v in result.items():
            print(f"{k}: {v}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
