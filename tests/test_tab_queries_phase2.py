import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"

PLAYER_STATS_SQL = """
    SELECT s.* FROM player_match_stats s
    JOIN roster_registrations r ON s.registration_id = r.registration_id
    WHERE r.player_id = ?
    ORDER BY s.match_date
"""


def _seed(conn):
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute("INSERT INTO teams VALUES (5, '新北中纖', 'F')")
    conn.execute("INSERT INTO players (name, gender) VALUES ('球員A', 'F')")
    pid = conn.execute("SELECT player_id FROM players").fetchone()[0]
    conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, jersey_number, position, source) "
        "VALUES (?, 5, 'F', '例行賽 Week 1', 2, 'OP', 'match_page')", (pid,),
    )
    rid1 = conn.execute("SELECT registration_id FROM roster_registrations").fetchone()[0]
    conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, jersey_number, position, source) "
        "VALUES (?, 5, 'F', '例行賽 Week 2', 5, 'OP', 'match_page')", (pid,),
    )
    rid2 = conn.execute(
        "SELECT registration_id FROM roster_registrations WHERE week_label = '例行賽 Week 2'"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO player_match_stats (registration_id, match_date, total_points) VALUES (?, '2025-11-01', 10)", (rid1,),
    )
    conn.execute(
        "INSERT INTO player_match_stats (registration_id, match_date, total_points) VALUES (?, '2025-11-08', 15)", (rid2,),
    )
    conn.commit()
    return pid


def test_player_stats_query_returns_all_weeks_across_registrations(tmp_db_path):
    conn = sqlite3.connect(tmp_db_path)
    pid = _seed(conn)

    rows = conn.execute(PLAYER_STATS_SQL, (pid,)).fetchall()

    assert len(rows) == 2, "應該撈到該球員橫跨兩週不同 registration 的全部統計"
    conn.close()


BOX_SCORE_SQL = """
    SELECT p.name, r.position, s.total_points
    FROM player_match_stats s
    JOIN roster_registrations r ON s.registration_id = r.registration_id
    JOIN players p ON r.player_id = p.player_id
    WHERE r.team_id = ? AND r.gender = ?
      AND s.match_date = ?
    ORDER BY s.total_points DESC
"""


def test_box_score_query_reflects_position_at_time_of_match(tmp_db_path):
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute("INSERT INTO teams VALUES (5, '新北中纖', 'F')")
    conn.execute("INSERT INTO players (name, gender) VALUES ('球員A', 'F')")
    pid = conn.execute("SELECT player_id FROM players").fetchone()[0]

    # 第1週登記為 OP，第2週改登記為 MB（模擬位置調整）
    conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, jersey_number, position, source) "
        "VALUES (?, 5, 'F', '例行賽 Week 1', 2, 'OP', 'match_page')", (pid,),
    )
    rid1 = conn.execute("SELECT registration_id FROM roster_registrations").fetchone()[0]
    conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, jersey_number, position, source) "
        "VALUES (?, 5, 'F', '例行賽 Week 2', 2, 'MB', 'match_page')", (pid,),
    )
    rid2 = conn.execute(
        "SELECT registration_id FROM roster_registrations WHERE week_label = '例行賽 Week 2'"
    ).fetchone()[0]

    conn.execute(
        "INSERT INTO player_match_stats (registration_id, match_date, total_points) VALUES (?, '2025-11-01', 10)", (rid1,),
    )
    conn.execute(
        "INSERT INTO player_match_stats (registration_id, match_date, total_points) VALUES (?, '2025-11-08', 15)", (rid2,),
    )
    conn.commit()

    week1_rows = conn.execute(BOX_SCORE_SQL, (5, "F", "2025-11-01")).fetchall()
    week2_rows = conn.execute(BOX_SCORE_SQL, (5, "F", "2025-11-08")).fetchall()

    assert week1_rows == [("球員A", "OP", 10)]
    assert week2_rows == [("球員A", "MB", 15)]
    conn.close()
