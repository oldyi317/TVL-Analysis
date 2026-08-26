import sqlite3
from pathlib import Path

import pytest

from src.etl.backup_db import backup_database

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    yield connection
    connection.close()


def test_v2_tables_exist(conn):
    tables = {
        row[0] for row in
        conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    }
    assert tables == {"teams", "players", "roster_registrations", "player_match_stats", "matches"}


def test_players_has_no_team_columns(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(players)")}
    assert "team_id" not in cols
    assert "jersey_number" not in cols
    assert "position" not in cols
    assert cols == {"player_id", "name", "gender", "dob", "height_cm", "weight_kg"}


def test_roster_registrations_unique_constraint(conn):
    conn.execute("INSERT INTO teams (team_id, team_name, gender) VALUES (1, '測試隊', 'F')")
    conn.execute("INSERT INTO players (name, gender) VALUES ('測試球員', 'F')")
    pid = conn.execute("SELECT player_id FROM players").fetchone()[0]
    conn.execute(
        """INSERT INTO roster_registrations
           (player_id, team_id, gender, week_label, jersey_number, position, source)
           VALUES (?, 1, 'F', '例行賽 Week 1', 5, 'OH', 'match_page')""",
        (pid,),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO roster_registrations
               (player_id, team_id, gender, week_label, jersey_number, position, source)
               VALUES (?, 1, 'F', '例行賽 Week 1', 6, 'MB', 'match_page')""",
            (pid,),
        )


def test_player_match_stats_fk_to_registration(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO player_match_stats (registration_id, match_date, total_points) VALUES (99999, '2026-01-01', 1)"
        )


def test_backup_database_creates_timestamped_copy(tmp_path):
    source = tmp_path / "fake.db"
    source.write_bytes(b"fake sqlite content")
    backup_path = backup_database(db_path=source)
    assert backup_path.exists()
    assert backup_path.name.startswith("fake.db.bak-")
    assert backup_path.read_bytes() == b"fake sqlite content"
