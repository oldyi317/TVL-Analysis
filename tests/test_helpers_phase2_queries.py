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
    ORDER BY r.jersey_number
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
