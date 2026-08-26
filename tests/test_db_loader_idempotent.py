import sqlite3

import pandas as pd
import pytest

from src.etl.db_loader import init_db, upsert_teams, upsert_players


@pytest.fixture
def conn(tmp_db_path):
    connection = sqlite3.connect(tmp_db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    yield connection
    connection.close()


SAMPLE_DF = pd.DataFrame([
    {"team_id": 1, "team_name": "測試隊", "gender": "M",
     "jersey_number": 10, "name": "測試球員", "position": "OH",
     "dob": "2000-01-01", "height_cm": 190.0, "weight_kg": 80.0},
])


def test_init_db_is_idempotent(conn):
    init_db(conn)
    init_db(conn)  # 第二次執行不應報錯
    tables = {
        row[0] for row in
        conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"teams", "players", "player_match_stats", "matches"} <= tables


def test_init_db_does_not_wipe_existing_stats(conn):
    init_db(conn)
    upsert_teams(conn, SAMPLE_DF)
    upsert_players(conn, SAMPLE_DF)
    player_id = conn.execute(
        "SELECT player_id FROM players WHERE name = '測試球員'"
    ).fetchone()[0]
    conn.execute(
        """INSERT INTO player_match_stats (player_id, match_date, total_points)
           VALUES (?, '2026-01-01', 10)""",
        (player_id,),
    )
    conn.commit()

    # 模擬重跑 db_loader：再次 init_db + upsert
    init_db(conn)
    upsert_teams(conn, SAMPLE_DF)
    upsert_players(conn, SAMPLE_DF)

    remaining = conn.execute(
        "SELECT COUNT(*) FROM player_match_stats"
    ).fetchone()[0]
    assert remaining == 1, "重跑 db_loader 不應清空 player_match_stats"


def test_upsert_players_preserves_player_id_on_rerun(conn):
    init_db(conn)
    upsert_teams(conn, SAMPLE_DF)
    upsert_players(conn, SAMPLE_DF)
    first_id = conn.execute(
        "SELECT player_id FROM players WHERE name = '測試球員'"
    ).fetchone()[0]

    upsert_players(conn, SAMPLE_DF)  # 重跑一次

    rows = conn.execute(
        "SELECT player_id FROM players WHERE name = '測試球員'"
    ).fetchall()
    assert len(rows) == 1, "同一自然鍵不應產生重複列"
    assert rows[0][0] == first_id, "重跑後 player_id 應保持不變"
