import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from src.etl.migrate_to_phase2 import run_migration

SCHEMA_V1 = """
CREATE TABLE teams (
    team_id INTEGER NOT NULL, team_name TEXT NOT NULL, gender TEXT NOT NULL,
    PRIMARY KEY (team_id, gender)
);
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT, team_id INTEGER NOT NULL,
    gender TEXT NOT NULL, jersey_number INTEGER, name TEXT, position TEXT,
    dob DATE, height_cm REAL, weight_kg REAL
);
CREATE TABLE player_match_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT, player_id INTEGER NOT NULL,
    match_date DATE, opponent TEXT, sets_played INTEGER, attack_total INTEGER,
    attack_points INTEGER, block_points INTEGER, serve_total INTEGER,
    serve_points INTEGER, receive_total INTEGER, receive_excellent INTEGER,
    dig_total INTEGER, dig_excellent INTEGER, set_total INTEGER,
    set_excellent INTEGER, total_points INTEGER, is_golden_set INTEGER DEFAULT 0,
    FOREIGN KEY (player_id) REFERENCES players (player_id)
);
CREATE TABLE matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL,
    gender TEXT NOT NULL, match_date DATE NOT NULL, round_name TEXT,
    home_team TEXT NOT NULL, away_team TEXT NOT NULL
);
"""


def _seed_v1_db(tmp_db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("PRAGMA foreign_keys = ON")  # 對齊正式連線行為：get_connection() 開 FK，測試種子不得關閉它
    conn.executescript(SCHEMA_V1)
    conn.execute("INSERT INTO teams VALUES (5, '新北中纖', 'F')")
    conn.execute(
        "INSERT INTO players (team_id, gender, jersey_number, name, position) "
        "VALUES (5, 'F', 2, '張瓈文', 'OP')"
    )
    conn.execute(
        "INSERT INTO matches (game_id, gender, match_date, round_name, home_team, away_team) "
        "VALUES (1, 'F', '2025-11-01', '例行賽 Week 1', '新北中纖', '義力營造')"
    )
    conn.execute(
        "INSERT INTO player_match_stats (player_id, match_date, opponent, sets_played, total_points) "
        "VALUES (1, '2025-11-01', '義力營造', 5, 20)"
    )
    conn.commit()
    return conn


def test_migration_preserves_stat_row_count(tmp_db_path):
    conn = _seed_v1_db(tmp_db_path)

    with patch(
        "src.etl.migrate_to_phase2.crawl_all_rosters",
        return_value={"matches_scanned": 0, "matches_skipped": 0, "registrations_upserted": 0, "new_players": 0},
    ):
        result = run_migration(conn, cup_id=21)

    assert result["stats_migrated"] == 1
    assert result["orphans_found"] == 0

    final_count = conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0]
    assert final_count == 1
    conn.close()


def test_migration_backfills_when_no_match_page_registration(tmp_db_path):
    conn = _seed_v1_db(tmp_db_path)

    with patch(
        "src.etl.migrate_to_phase2.crawl_all_rosters",
        return_value={"matches_scanned": 0, "matches_skipped": 0, "registrations_upserted": 0, "new_players": 0},
    ):
        result = run_migration(conn, cup_id=21)

    assert result["stats_backfilled"] == 1
    source = conn.execute("SELECT source FROM roster_registrations").fetchone()[0]
    assert source == "backfill"
    conn.close()


def test_migration_preserves_player_id(tmp_db_path):
    conn = _seed_v1_db(tmp_db_path)

    with patch(
        "src.etl.migrate_to_phase2.crawl_all_rosters",
        return_value={"matches_scanned": 0, "matches_skipped": 0, "registrations_upserted": 0, "new_players": 0},
    ):
        run_migration(conn, cup_id=21)

    row = conn.execute("SELECT player_id, name FROM players").fetchone()
    assert row == (1, "張瓈文")
    conn.close()


def test_migration_drops_old_tables(tmp_db_path):
    conn = _seed_v1_db(tmp_db_path)

    with patch(
        "src.etl.migrate_to_phase2.crawl_all_rosters",
        return_value={"matches_scanned": 0, "matches_skipped": 0, "registrations_upserted": 0, "new_players": 0},
    ):
        run_migration(conn, cup_id=21)

    tables = {
        row[0] for row in
        conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "players_old" not in tables
    assert "player_match_stats_old" not in tables
    conn.close()


def test_rerun_migration_raises_and_leaves_data_untouched(tmp_db_path):
    """Finding 4: 已遷移過的 DB 再跑一次應直接拋錯，不得動到既有資料。"""
    conn = _seed_v1_db(tmp_db_path)

    with patch(
        "src.etl.migrate_to_phase2.crawl_all_rosters",
        return_value={"matches_scanned": 0, "matches_skipped": 0, "registrations_upserted": 0, "new_players": 0},
    ):
        first_result = run_migration(conn, cup_id=21)

        players_before = conn.execute("SELECT player_id, name FROM players").fetchall()
        stats_before = conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0]

        with pytest.raises(RuntimeError):
            run_migration(conn, cup_id=21)

    players_after = conn.execute("SELECT player_id, name FROM players").fetchall()
    stats_after = conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0]

    assert players_after == players_before
    assert stats_after == stats_before
    assert stats_after == first_result["stats_migrated"]
    conn.close()
