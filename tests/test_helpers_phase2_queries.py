import sqlite3
from pathlib import Path
import pytest

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"

GET_CURRENT_ROSTER_SQL = """
    SELECT r.player_id, r.jersey_number, p.name, r.position
    FROM roster_registrations r
    JOIN players p ON r.player_id = p.player_id
    WHERE r.team_id = ? AND r.gender = ?
      AND r.week_start_date = (
          SELECT MAX(week_start_date) FROM roster_registrations
          WHERE team_id = r.team_id AND gender = r.gender
      )
    ORDER BY r.jersey_number IS NULL, r.jersey_number
"""

GET_LEAGUE_AGGREGATED_STATS_SQL = """
    SELECT p.player_id,
           p.name,
           latest.position AS position,
           t.team_name,
           SUM(s.sets_played)       AS total_sets,
           SUM(s.attack_points)     AS atk_pts,
           SUM(s.attack_total)      AS atk_tot,
           SUM(s.block_points)      AS blk_pts,
           SUM(s.serve_points)      AS srv_pts,
           SUM(s.serve_total)       AS srv_tot,
           SUM(s.receive_excellent) AS rcv_exc,
           SUM(s.receive_total)     AS rcv_tot,
           SUM(s.dig_excellent)     AS dig_exc,
           SUM(s.dig_total)         AS dig_tot,
           SUM(s.set_excellent)     AS set_exc,
           SUM(s.set_total)         AS set_tot,
           SUM(s.total_points)      AS total_points,
           COUNT(*)                 AS n_games
    FROM player_match_stats s
    JOIN roster_registrations r ON s.registration_id = r.registration_id
    JOIN players p ON r.player_id = p.player_id
    JOIN (
        SELECT rr.player_id, rr.position, rr.team_id, rr.gender
        FROM roster_registrations rr
        WHERE rr.registration_id = (
            SELECT rr2.registration_id FROM roster_registrations rr2
            WHERE rr2.player_id = rr.player_id
            ORDER BY rr2.week_start_date DESC, rr2.registration_id DESC
            LIMIT 1
        )
    ) latest ON latest.player_id = p.player_id
    JOIN teams t ON t.team_id = latest.team_id AND t.gender = latest.gender
    WHERE latest.gender = ?
    GROUP BY p.player_id
    HAVING SUM(s.sets_played) >= 5
"""


def _seed(conn):
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute("INSERT INTO teams VALUES (5, '新北中纖', 'F')")
    conn.execute("INSERT INTO players (name, gender) VALUES ('球員A', 'F')")
    conn.execute("INSERT INTO players (name, gender) VALUES ('球員B', 'F')")
    pid_a = conn.execute("SELECT player_id FROM players WHERE name='球員A'").fetchone()[0]
    pid_b = conn.execute("SELECT player_id FROM players WHERE name='球員B'").fetchone()[0]
    # 球員A：第1週背號2，第2週背號5（換背號）
    conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, week_start_date, jersey_number, position, source) "
        "VALUES (?, 5, 'F', '例行賽 Week 1', '2025-11-01', 2, 'OP', 'match_page')", (pid_a,),
    )
    conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, week_start_date, jersey_number, position, source) "
        "VALUES (?, 5, 'F', '例行賽 Week 2', '2025-11-08', 5, 'OP', 'match_page')", (pid_a,),
    )
    # 球員B：只在第1週出現過
    conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, week_start_date, jersey_number, position, source) "
        "VALUES (?, 5, 'F', '例行賽 Week 1', '2025-11-01', 9, 'MB', 'match_page')", (pid_b,),
    )
    conn.commit()
    return pid_a, pid_b


def test_get_current_roster_uses_latest_week_per_team(tmp_path):
    tmp_db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(tmp_db_path))
    pid_a, pid_b = _seed(conn)

    rows = conn.execute(GET_CURRENT_ROSTER_SQL, (5, "F")).fetchall()

    # 只有球員A有第2週的紀錄（week_start_date 最大），球員B停留在第1週不應出現
    assert rows == [(pid_a, 5, "球員A", "OP")]
    conn.close()


def test_aggregated_stats_handles_same_week_different_teams_no_double_count(tmp_path):
    """Finding 1: 同球員同一 week_start_date 登入不同隊時，不應翻倍 SUM。"""
    tmp_db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(tmp_db_path))
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys = ON")

    # 建立兩隊
    conn.execute("INSERT INTO teams VALUES (1, '隊1', 'F')")
    conn.execute("INSERT INTO teams VALUES (2, '隊2', 'F')")

    # 建立球員
    conn.execute("INSERT INTO players (name, gender) VALUES ('轉隊球員', 'F')")
    pid = conn.execute("SELECT player_id FROM players WHERE name='轉隊球員'").fetchone()[0]

    # 同一週內在兩隊各登錄一次（季中轉隊情境）
    cur1 = conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, week_start_date, jersey_number, position, source) "
        "VALUES (?, 1, 'F', '例行賽 Week 1', '2025-11-01', 10, 'OP', 'match_page')", (pid,),
    )
    reg1_id = cur1.lastrowid
    cur2 = conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, week_start_date, jersey_number, position, source) "
        "VALUES (?, 2, 'F', '例行賽 Week 1', '2025-11-01', 5, 'MB', 'match_page')", (pid,),
    )
    reg2_id = cur2.lastrowid

    # 各登錄各掛一筆統計（註：player_match_stats 的 sets_played 預設為某值，確保 >= 5）
    conn.execute(
        "INSERT INTO player_match_stats (registration_id, sets_played, attack_points, attack_total, block_points, serve_points, serve_total, "
        "receive_excellent, receive_total, dig_excellent, dig_total, set_excellent, set_total, total_points) "
        "VALUES (?, 5, 10, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15)",
        (reg1_id,),
    )
    conn.execute(
        "INSERT INTO player_match_stats (registration_id, sets_played, attack_points, attack_total, block_points, serve_points, serve_total, "
        "receive_excellent, receive_total, dig_excellent, dig_total, set_excellent, set_total, total_points) "
        "VALUES (?, 6, 20, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25)",
        (reg2_id,),
    )
    conn.commit()

    # 執行彙總 SQL，verify 該球員只出現一列且 SUM 不翻倍
    rows = conn.execute(GET_LEAGUE_AGGREGATED_STATS_SQL, ("F",)).fetchall()

    assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
    row = rows[0]
    # row[0] = player_id, row[1] = name, row[2] = position, row[3] = team_name,
    # row[16] = total_points, row[17] = n_games
    assert row[0] == pid
    assert row[1] == '轉隊球員'
    assert row[16] == 40  # 15 + 25 (not doubled)
    assert row[17] == 2   # 2 games

    conn.close()


def test_aggregated_stats_includes_player_with_null_week_start_date(tmp_path):
    """Finding 2: 若某球員所有登錄的 week_start_date 皆 NULL，應仍出現在彙總。"""
    tmp_db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(tmp_db_path))
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys = ON")

    # 建立隊伍
    conn.execute("INSERT INTO teams VALUES (3, '隊3', 'F')")

    # 建立球員
    conn.execute("INSERT INTO players (name, gender) VALUES ('幽靈球員', 'F')")
    pid = conn.execute("SELECT player_id FROM players WHERE name='幽靈球員'").fetchone()[0]

    # 登錄時 week_start_date 為 NULL（week_label 必填，用 backfill 標記）
    cur = conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, week_start_date, jersey_number, position, source) "
        "VALUES (?, 3, 'F', 'backfill', NULL, 7, 'L', 'backfill')", (pid,),
    )
    reg_id = cur.lastrowid

    # 掛一筆統計
    conn.execute(
        "INSERT INTO player_match_stats (registration_id, sets_played, attack_points, attack_total, block_points, serve_points, serve_total, "
        "receive_excellent, receive_total, dig_excellent, dig_total, set_excellent, set_total, total_points) "
        "VALUES (?, 5, 5, 15, 2, 3, 20, 10, 15, 8, 12, 2, 5, 20)",
        (reg_id,),
    )
    conn.commit()

    # 執行彙總 SQL，verify 該球員仍出現在結果中
    rows = conn.execute(GET_LEAGUE_AGGREGATED_STATS_SQL, ("F",)).fetchall()

    assert len(rows) == 1, f"Expected 1 row (player with NULL week_start_date), got {len(rows)}"
    row = rows[0]
    assert row[0] == pid
    assert row[1] == '幽靈球員'
    assert row[17] == 1  # n_games = 1

    conn.close()
