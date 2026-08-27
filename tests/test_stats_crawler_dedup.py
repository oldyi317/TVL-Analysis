"""
回歸測試：stats_crawler 的去重機制必須在全量、增量模式下皆生效，
避免無 UNIQUE 約束擋不住重複寫入（見 Phase 1 review Finding #1）。
Phase 2 起 player_match_stats 改掛 registration_id，種子資料改建
teams/players/roster_registrations 三層再掛 stats。
"""

import sqlite3
from pathlib import Path

from src.etl.stats_crawler import filter_new_records

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


def _make_conn(tmp_db_path) -> tuple[sqlite3.Connection, int]:
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute(
        "INSERT INTO teams (team_id, team_name, gender) VALUES (1, '測試隊', 'M')"
    )
    conn.execute(
        "INSERT INTO players (name, gender) VALUES ('測試球員', 'M')"
    )
    player_id = conn.execute("SELECT player_id FROM players").fetchone()[0]
    conn.execute(
        """INSERT INTO roster_registrations
           (player_id, team_id, gender, cup_id, week_label, jersey_number, position, source)
           VALUES (?, 1, 'M', 21, '例行賽 Week 1', 5, 'OH', 'match_page')""",
        (player_id,),
    )
    conn.commit()
    return conn, player_id


def _insert_records(conn: sqlite3.Connection, player_id: int, records: list[dict]) -> None:
    """與 stats_crawler.main() 相同的批次寫入邏輯（Phase 2：掛 registration_id）。"""
    registration_id = conn.execute(
        "SELECT registration_id FROM roster_registrations WHERE player_id = ?",
        (player_id,),
    ).fetchone()[0]
    conn.executemany(
        """INSERT INTO player_match_stats
           (registration_id, match_date, opponent, sets_played,
            attack_total, attack_points, block_points,
            serve_total, serve_points,
            receive_total, receive_excellent,
            dig_total, dig_excellent,
            set_total, set_excellent, total_points,
            is_golden_set)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                registration_id,
                r["match_date"], r["opponent"], r["sets_played"],
                r["attack_total"], r["attack_points"], r["block_points"],
                r["serve_total"], r["serve_points"],
                r["receive_total"], r["receive_excellent"],
                r["dig_total"], r["dig_excellent"],
                r["set_total"], r["set_excellent"], r["total_points"],
                r["is_golden_set"],
            )
            for r in records
        ],
    )
    conn.commit()


def test_filter_new_records_prevents_duplicate_insert_on_rerun(tmp_db_path):
    conn, player_id = _make_conn(tmp_db_path)
    try:
        fake_record = {
            "match_date": "2026-01-01",
            "opponent": "對手隊",
            "sets_played": 3,
            "attack_total": 10,
            "attack_points": 5,
            "block_points": 1,
            "serve_total": 8,
            "serve_points": 2,
            "receive_total": 6,
            "receive_excellent": 3,
            "dig_total": 4,
            "dig_excellent": 2,
            "set_total": 0,
            "set_excellent": 0,
            "total_points": 8,
            "is_golden_set": 0,
        }

        # 第一次：同一筆紀錄應成功寫入（走真正的 filter_new_records + insert 路徑）
        new_records = filter_new_records(conn, player_id, [fake_record])
        assert len(new_records) == 1
        _insert_records(conn, player_id, new_records)

        # 第二次：以完全相同的 records 重跑（模擬全量模式重複執行），應被去重濾掉
        new_records_again = filter_new_records(conn, player_id, [fake_record])
        assert new_records_again == []
        _insert_records(conn, player_id, new_records_again)

        total_rows = conn.execute(
            "SELECT COUNT(*) FROM player_match_stats"
        ).fetchone()[0]
        assert total_rows == 1
    finally:
        conn.close()
