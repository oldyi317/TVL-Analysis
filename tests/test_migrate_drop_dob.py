import sqlite3
from pathlib import Path

import pytest

from src.etl.migrate_drop_dob import run_migration

# 遷移前的 players（含 dob）——固定快照，不隨 schema.sql 演進
PLAYERS_WITH_DOB = """
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
    gender TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    dob DATE, height_cm REAL, weight_kg REAL
);
"""


@pytest.fixture
def conn(tmp_db_path: Path):
    connection = sqlite3.connect(tmp_db_path)
    connection.executescript(PLAYERS_WITH_DOB)
    connection.execute(
        "INSERT INTO players (name, gender, dob, height_cm, weight_kg) VALUES ('甲', 'M', '2000-01-01', 190.0, 80.0)"
    )
    connection.execute(
        "INSERT INTO players (name, gender, dob, height_cm, weight_kg) VALUES ('乙', 'F', '2005-05-05', 170.5, 60.0)"
    )
    connection.commit()
    yield connection
    connection.close()


def test_drop_dob_keeps_rows_and_other_columns(conn):
    result = run_migration(conn)

    cols = [row[1] for row in conn.execute("PRAGMA table_info(players)")]
    assert "dob" not in cols
    assert result["players"] == 2
    assert conn.execute("SELECT COUNT(*) FROM players").fetchone()[0] == 2
    rows = conn.execute("SELECT name, gender, height_cm, weight_kg FROM players ORDER BY player_id").fetchall()
    assert rows == [("甲", "M", 190.0, 80.0), ("乙", "F", 170.5, 60.0)]


def test_rerun_raises(conn):
    run_migration(conn)
    with pytest.raises(RuntimeError):
        run_migration(conn)
