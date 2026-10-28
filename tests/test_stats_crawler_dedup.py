"""
回歸測試：stats_crawler 的去重機制必須在全量、增量模式下皆生效，
避免無 UNIQUE 約束擋不住重複寫入（見 Phase 1 review Finding #1）。
Phase 2 起 player_match_stats 改掛 registration_id，種子資料改建
teams/players/roster_registrations 三層再掛 stats。
"""

import sqlite3
from pathlib import Path

from src.etl import stats_crawler
from src.etl.stats_crawler import PlayerResolver, collect_player_rows, insert_stat_rows

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
    conn.execute(
        "INSERT INTO matches (game_id, gender, cup_id, match_date, round_name, home_team, away_team) "
        "VALUES (1, 'M', 21, '2026-01-01', '例行賽 Week 1', '測試隊', '對手隊')"
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


def test_filter_new_records_prevents_duplicate_insert_on_rerun(tmp_db_path, monkeypatch):
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

        monkeypatch.setattr(stats_crawler, "fetch_player_stats", lambda *a: [fake_record])
        resolver = PlayerResolver(conn, 21)
        p = {"ext_player_id": 1, "name": "測試球員", "jersey_number": 5}

        def run_once():
            rows, skipped = collect_player_rows(None, resolver, 1, 1, "M", p)
            insert_stat_rows(conn, rows)
            conn.commit()
            return rows, skipped

        # 第一次：同一筆紀錄應成功寫入
        rows, skipped = run_once()
        assert len(rows) == 1 and skipped == 0

        # 第二次：以完全相同的 records 重跑（模擬全量模式重複執行），應被去重濾掉
        rows, skipped = run_once()
        assert rows == [] and skipped == 1

        total_rows = conn.execute(
            "SELECT COUNT(*) FROM player_match_stats"
        ).fetchone()[0]
        assert total_rows == 1
    finally:
        conn.close()
