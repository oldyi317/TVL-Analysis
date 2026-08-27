import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"

PLAYER_STATS_SQL = """
    SELECT s.* FROM player_match_stats s
    JOIN roster_registrations r ON s.registration_id = r.registration_id
    WHERE r.player_id = ? AND r.cup_id = ?
    ORDER BY s.match_date
"""


def _seed(conn):
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute("INSERT INTO teams VALUES (5, '新北中纖', 'F')")
    conn.execute("INSERT INTO players (name, gender) VALUES ('球員A', 'F')")
    pid = conn.execute("SELECT player_id FROM players").fetchone()[0]
    conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, cup_id, week_label, jersey_number, position, source) "
        "VALUES (?, 5, 'F', 21, '例行賽 Week 1', 2, 'OP', 'match_page')", (pid,),
    )
    rid1 = conn.execute("SELECT registration_id FROM roster_registrations").fetchone()[0]
    conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, cup_id, week_label, jersey_number, position, source) "
        "VALUES (?, 5, 'F', 21, '例行賽 Week 2', 5, 'OP', 'match_page')", (pid,),
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

    rows = conn.execute(PLAYER_STATS_SQL, (pid, 21)).fetchall()

    assert len(rows) == 2, "應該撈到該球員橫跨兩週不同 registration 的全部統計"
    conn.close()


BOX_SCORE_SQL = """
    SELECT p.name, r.position, s.total_points
    FROM player_match_stats s
    JOIN roster_registrations r ON s.registration_id = r.registration_id
    JOIN players p ON r.player_id = p.player_id
    WHERE r.team_id = ? AND r.gender = ? AND r.cup_id = ?
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
        "INSERT INTO roster_registrations (player_id, team_id, gender, cup_id, week_label, jersey_number, position, source) "
        "VALUES (?, 5, 'F', 21, '例行賽 Week 1', 2, 'OP', 'match_page')", (pid,),
    )
    rid1 = conn.execute("SELECT registration_id FROM roster_registrations").fetchone()[0]
    conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, cup_id, week_label, jersey_number, position, source) "
        "VALUES (?, 5, 'F', 21, '例行賽 Week 2', 2, 'MB', 'match_page')", (pid,),
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

    week1_rows = conn.execute(BOX_SCORE_SQL, (5, "F", 21, "2025-11-01")).fetchall()
    week2_rows = conn.execute(BOX_SCORE_SQL, (5, "F", 21, "2025-11-08")).fetchall()

    assert week1_rows == [("球員A", "OP", 10)]
    assert week2_rows == [("球員A", "MB", 15)]
    conn.close()


MATCH_SELECTOR_SQL = """
    SELECT DISTINCT s.match_date, s.opponent
    FROM player_match_stats s
    JOIN roster_registrations r ON s.registration_id = r.registration_id
    WHERE r.team_id = ? AND r.gender = ? AND r.cup_id = ?
    ORDER BY s.match_date
"""


def test_match_selector_query_returns_distinct_matches_per_team_and_date(tmp_db_path):
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute("INSERT INTO teams VALUES (5, '新北中纖', 'F')")
    conn.execute("INSERT INTO players (name, gender) VALUES ('球員A', 'F')")
    pid = conn.execute("SELECT player_id FROM players").fetchone()[0]

    # 建立兩週登錄
    conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, cup_id, week_label, jersey_number, position, source) "
        "VALUES (?, 5, 'F', 21, '例行賽 Week 1', 2, 'OP', 'match_page')", (pid,),
    )
    rid1 = conn.execute("SELECT registration_id FROM roster_registrations").fetchone()[0]
    conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, cup_id, week_label, jersey_number, position, source) "
        "VALUES (?, 5, 'F', 21, '例行賽 Week 2', 2, 'OP', 'match_page')", (pid,),
    )
    rid2 = conn.execute(
        "SELECT registration_id FROM roster_registrations WHERE week_label = '例行賽 Week 2'"
    ).fetchone()[0]

    # 兩週各一場比賽，對手名稱不同
    conn.execute(
        "INSERT INTO player_match_stats (registration_id, match_date, opponent, total_points) VALUES (?, '2025-11-01', '台北排協', 10)", (rid1,),
    )
    conn.execute(
        "INSERT INTO player_match_stats (registration_id, match_date, opponent, total_points) VALUES (?, '2025-11-08', '高雄隊', 15)", (rid2,),
    )
    conn.commit()

    rows = conn.execute(MATCH_SELECTOR_SQL, (5, "F", 21)).fetchall()

    assert len(rows) == 2, "應該撈到兩週各一場比賽的 DISTINCT 列表"
    assert rows == [("2025-11-01", "台北排協"), ("2025-11-08", "高雄隊")]
    conn.close()


def test_match_selector_query_excludes_other_seasons(tmp_db_path):
    """他季（cup_id=20）同名週次的登錄與統計不得混入當季比賽選單。"""
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute("INSERT INTO teams VALUES (5, '新北中纖', 'F')")
    conn.execute("INSERT INTO players (name, gender) VALUES ('球員A', 'F')")
    pid = conn.execute("SELECT player_id FROM players").fetchone()[0]

    # 當季登錄 + 一場比賽
    conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, cup_id, week_label, jersey_number, position, source) "
        "VALUES (?, 5, 'F', 21, '例行賽 Week 1', 2, 'OP', 'match_page')", (pid,),
    )
    rid_now = conn.execute("SELECT registration_id FROM roster_registrations").fetchone()[0]
    conn.execute(
        "INSERT INTO player_match_stats (registration_id, match_date, opponent, total_points) VALUES (?, '2025-11-01', '台北排協', 10)", (rid_now,),
    )

    # 他季（cup_id=20）同名週次登錄 + 一場比賽
    conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, cup_id, week_label, jersey_number, position, source) "
        "VALUES (?, 5, 'F', 20, '例行賽 Week 1', 2, 'OP', 'match_page')", (pid,),
    )
    rid_old = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO player_match_stats (registration_id, match_date, opponent, total_points) VALUES (?, '2024-11-01', '高雄隊', 15)", (rid_old,),
    )
    conn.commit()

    rows = conn.execute(MATCH_SELECTOR_SQL, (5, "F", 21)).fetchall()

    assert rows == [("2025-11-01", "台北排協")], f"他季比賽混入了當季選單：{rows}"
    conn.close()
