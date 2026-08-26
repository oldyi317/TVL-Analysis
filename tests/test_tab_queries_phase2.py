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
