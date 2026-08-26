import sqlite3
from pathlib import Path

from src.etl.stats_crawler import resolve_registration_for_stats

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


def _make_conn(tmp_db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute("INSERT INTO teams (team_id, team_name, gender) VALUES (5, '新北中纖', 'F')")
    conn.execute(
        "INSERT INTO matches (game_id, gender, match_date, round_name, home_team, away_team) "
        "VALUES (1, 'F', '2025-11-01', '例行賽 Week 1', '新北中纖', '義力營造')"
    )
    conn.execute("INSERT INTO players (name, gender) VALUES ('張瓈文', 'F')")
    conn.commit()
    return conn


def test_resolve_registration_reuses_existing_match_page_row(tmp_db_path):
    conn = _make_conn(tmp_db_path)
    try:
        pid = conn.execute("SELECT player_id FROM players").fetchone()[0]
        conn.execute(
            """INSERT INTO roster_registrations
               (player_id, team_id, gender, week_label, jersey_number, position, source)
               VALUES (?, 5, 'F', '例行賽 Week 1', 2, 'OP', 'match_page')""",
            (pid,),
        )
        conn.commit()

        rid = resolve_registration_for_stats(conn, pid, 5, "F", "2025-11-01")

        row = conn.execute(
            "SELECT source, jersey_number FROM roster_registrations WHERE registration_id = ?",
            (rid,),
        ).fetchone()
        assert row == ("match_page", 2), "已有真實登錄時，不得覆蓋成 backfill"
    finally:
        conn.close()


def test_resolve_registration_creates_backfill_when_missing(tmp_db_path):
    conn = _make_conn(tmp_db_path)
    try:
        pid = conn.execute("SELECT player_id FROM players").fetchone()[0]

        rid = resolve_registration_for_stats(conn, pid, 5, "F", "2025-11-01")

        row = conn.execute(
            "SELECT source, jersey_number, position, week_label FROM roster_registrations WHERE registration_id = ?",
            (rid,),
        ).fetchone()
        assert row == ("backfill", None, None, "例行賽 Week 1")
    finally:
        conn.close()


def test_resolve_registration_is_idempotent(tmp_db_path):
    conn = _make_conn(tmp_db_path)
    try:
        pid = conn.execute("SELECT player_id FROM players").fetchone()[0]

        rid_first = resolve_registration_for_stats(conn, pid, 5, "F", "2025-11-01")
        rid_second = resolve_registration_for_stats(conn, pid, 5, "F", "2025-11-01")

        assert rid_first == rid_second
        count = conn.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()[0]
        assert count == 1
    finally:
        conn.close()
