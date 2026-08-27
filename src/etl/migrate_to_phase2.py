"""
Phase 2 一次性遷移腳本
players 拆表為身分層 + roster_registrations，player_match_stats 改掛 registration_id。
執行前一律先備份（見 backup_db.py），此腳本本身也會在開頭再備份一次以防萬一。
"""

import sqlite3

from src.etl.backup_db import backup_database
from src.etl.stats_crawler import crawl_all_rosters
from src.utils.constants import EXT_CUP_ID as CUP_ID
from src.utils.db_config import PROJECT_ROOT, get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


def _rename_old_tables(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE players RENAME TO players_old")
    conn.execute("ALTER TABLE player_match_stats RENAME TO player_match_stats_old")
    conn.commit()
    logger.info("已將舊表重新命名為 players_old / player_match_stats_old")


def _create_new_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    logger.info("已依新版 schema.sql 建立 players / roster_registrations / player_match_stats")


def _migrate_player_identities(conn: sqlite3.Connection) -> int:
    """把 players_old 的身分欄位（保留原 player_id）搬進新 players。"""
    rows = conn.execute(
        "SELECT player_id, name, gender, dob, height_cm, weight_kg FROM players_old"
    ).fetchall()
    conn.executemany(
        "INSERT INTO players (player_id, name, gender, dob, height_cm, weight_kg) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    logger.info("已搬移 %d 筆球員身分資料（player_id 保持不變）", len(rows))
    return len(rows)


def _backfill_registration(conn: sqlite3.Connection, old_player_row: tuple, week_label: str, week_start_date: str) -> int:
    """
    用舊 players_old 的快照（team_id/gender/jersey_number/position）建一筆
    source='backfill' 的登錄記錄。只在 crawl_all_rosters() 抓不到真實出賽
    名單時才會呼叫（例如爬蟲涵蓋範圍外的週次、或該球員該週未被系統記錄）。
    """
    player_id, team_id, gender, jersey_number, position = old_player_row
    cur = conn.execute(
        """
        INSERT INTO roster_registrations
            (player_id, team_id, gender, cup_id, week_label, week_start_date, jersey_number, position, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'backfill')
        ON CONFLICT (player_id, team_id, gender, cup_id, week_label) DO NOTHING
        """,
        (player_id, team_id, gender, CUP_ID, week_label, week_start_date, jersey_number, position),
    )
    if cur.lastrowid and cur.rowcount:
        return cur.lastrowid
    # ON CONFLICT DO NOTHING 命中時要回頭查已存在的那筆
    row = conn.execute(
        """SELECT registration_id FROM roster_registrations
           WHERE player_id = ? AND team_id = ? AND gender = ? AND cup_id = ? AND week_label = ?""",
        (player_id, team_id, gender, CUP_ID, week_label),
    ).fetchone()
    return row[0]


def _migrate_stats(conn: sqlite3.Connection, cup_id: int) -> dict:
    """逐筆把 player_match_stats_old 重新掛到 roster_registrations，回傳統計。"""
    stat_cols = [
        "match_date", "opponent", "sets_played", "attack_total", "attack_points",
        "block_points", "serve_total", "serve_points", "receive_total",
        "receive_excellent", "dig_total", "dig_excellent", "set_total",
        "set_excellent", "total_points", "is_golden_set",
    ]
    old_rows = conn.execute(
        f"SELECT stat_id, player_id, {', '.join(stat_cols)} FROM player_match_stats_old"
    ).fetchall()

    migrated, backfilled = 0, 0
    for old_row in old_rows:
        stat_id, player_id, *values = old_row
        match_date = values[0]

        player_snapshot = conn.execute(
            "SELECT team_id, gender, jersey_number, position FROM players_old WHERE player_id = ?",
            (player_id,),
        ).fetchone()
        if player_snapshot is None:
            logger.error("stat_id=%s 找不到對應 players_old.player_id=%s，跳過", stat_id, player_id)
            continue
        team_id, gender, jersey_number, position = player_snapshot

        week_row = conn.execute(
            "SELECT round_name FROM matches WHERE match_date = ? ORDER BY round_name LIMIT 1",
            (match_date,),
        ).fetchone()
        week_label = week_row[0] if week_row and week_row[0] else f"未比對-{match_date}"

        reg_row = conn.execute(
            """SELECT registration_id FROM roster_registrations
               WHERE player_id = ? AND team_id = ? AND gender = ? AND week_label = ?
                 AND source = 'match_page'""",
            (player_id, team_id, gender, week_label),
        ).fetchone()

        if reg_row:
            registration_id = reg_row[0]
        else:
            start_row = conn.execute(
                "SELECT MIN(match_date) FROM matches WHERE round_name = ? "
                "AND ABS(julianday(match_date) - julianday(?)) < 200",
                (week_label, match_date),
            ).fetchone()
            week_start_date = start_row[0] if start_row and start_row[0] else match_date
            registration_id = _backfill_registration(
                conn, (player_id, team_id, gender, jersey_number, position),
                week_label, week_start_date,
            )
            backfilled += 1

        conn.execute(
            f"""INSERT INTO player_match_stats (registration_id, {', '.join(stat_cols)})
                VALUES ({', '.join(['?'] * (len(stat_cols) + 1))})""",
            (registration_id, *values),
        )
        migrated += 1

    conn.commit()
    return {"stats_migrated": migrated, "stats_backfilled": backfilled}


def _verify(conn: sqlite3.Connection, expected_stat_count: int) -> int:
    """驗證：筆數不減少、無孤兒 FK。回傳孤兒數（應為 0）。"""
    actual = conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0]
    assert actual == expected_stat_count, f"筆數不符：預期 {expected_stat_count}，實際 {actual}"

    orphans = conn.execute("""
        SELECT COUNT(*) FROM player_match_stats s
        WHERE NOT EXISTS (
            SELECT 1 FROM roster_registrations r WHERE r.registration_id = s.registration_id
        )
    """).fetchone()[0]
    return orphans


def _drop_old_tables(conn: sqlite3.Connection) -> None:
    # FK ON 時必須子表先 drop、父表後 drop：player_match_stats_old 仍是
    # players_old 的子表（舊 FK 尚未搬移前的關聯），先 drop players_old
    # 會觸發 FOREIGN KEY constraint failed（已用 in-memory 實驗證實）。
    conn.execute("DROP TABLE player_match_stats_old")
    conn.execute("DROP TABLE players_old")
    conn.commit()
    logger.info("已清除 player_match_stats_old / players_old")


def _assert_not_migrated(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='roster_registrations'"
    ).fetchone()
    if row:
        raise RuntimeError("此資料庫已完成 Phase 2 遷移，請勿重複執行。")


def run_migration(conn: sqlite3.Connection, cup_id: int = CUP_ID) -> dict:
    _assert_not_migrated(conn)
    expected_stat_count = conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0]

    _rename_old_tables(conn)
    _create_new_tables(conn)

    n_players = _migrate_player_identities(conn)

    crawl_stats = crawl_all_rosters(conn, cup_id=cup_id)
    logger.info("出賽名單爬蟲結果：%s", crawl_stats)

    stats_result = _migrate_stats(conn, cup_id)

    orphans = _verify(conn, expected_stat_count)
    if orphans > 0:
        raise RuntimeError(
            f"遷移驗證失敗：發現 {orphans} 筆孤兒 player_match_stats，未清除舊表，"
            "請檢查 _migrate_stats() 邏輯後重跑（新表可安全重建，因為舊表還在）。"
        )

    _drop_old_tables(conn)

    return {
        "players_migrated": n_players,
        "registrations_created": crawl_stats["registrations_upserted"],
        **stats_result,
        "orphans_found": orphans,
    }


def main():
    backup_database()
    conn = get_connection()
    try:
        result = run_migration(conn)
        print("\n===== Phase 2 遷移完成 =====")
        for k, v in result.items():
            print(f"{k}: {v}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
