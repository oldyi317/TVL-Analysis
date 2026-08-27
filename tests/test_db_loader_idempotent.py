import sqlite3

import pandas as pd
import pytest

from src.etl.db_loader import init_db, upsert_teams, upsert_player_identity


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
    init_db(conn)
    tables = {
        row[0] for row in
        conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"teams", "players", "roster_registrations", "player_match_stats", "matches"} <= tables


def test_upsert_player_identity_does_not_write_team_or_jersey(conn):
    init_db(conn)
    upsert_teams(conn, SAMPLE_DF)
    upsert_player_identity(conn, SAMPLE_DF)

    cols = {row[1] for row in conn.execute("PRAGMA table_info(players)")}
    assert "team_id" not in cols
    assert "jersey_number" not in cols
    assert "position" not in cols

    row = conn.execute(
        "SELECT name, gender, dob, height_cm, weight_kg FROM players WHERE name = '測試球員'"
    ).fetchone()
    assert row == ("測試球員", "M", "2000-01-01", 190.0, 80.0)


def test_upsert_player_identity_preserves_player_id_on_rerun(conn):
    init_db(conn)
    upsert_teams(conn, SAMPLE_DF)
    upsert_player_identity(conn, SAMPLE_DF)
    first_id = conn.execute(
        "SELECT player_id FROM players WHERE name = '測試球員'"
    ).fetchone()[0]

    upsert_player_identity(conn, SAMPLE_DF)

    rows = conn.execute(
        "SELECT player_id FROM players WHERE name = '測試球員'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == first_id


def test_init_db_does_not_wipe_existing_registrations(conn):
    init_db(conn)
    upsert_teams(conn, SAMPLE_DF)
    upsert_player_identity(conn, SAMPLE_DF)
    player_id = conn.execute(
        "SELECT player_id FROM players WHERE name = '測試球員'"
    ).fetchone()[0]
    conn.execute(
        """INSERT INTO roster_registrations
           (player_id, team_id, gender, cup_id, week_label, jersey_number, position, source)
           VALUES (?, 1, 'M', 21, '例行賽 Week 1', 10, 'OH', 'match_page')""",
        (player_id,),
    )
    conn.commit()

    init_db(conn)
    upsert_teams(conn, SAMPLE_DF)
    upsert_player_identity(conn, SAMPLE_DF)

    remaining = conn.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()[0]
    assert remaining == 1
