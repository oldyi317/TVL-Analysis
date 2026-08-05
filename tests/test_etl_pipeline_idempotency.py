import pandas as pd
from sqlalchemy import text

from src.etl.db_loader import insert_players, insert_teams
from src.etl.match_crawler import upsert_match
from src.etl.stats_crawler import upsert_stats


def _roster_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "team_id": 1, "team_name": "屏東台電", "gender": "M",
        "jersey_number": 4, "name": "李元", "position": "OH",
        "dob": None, "height_cm": None, "weight_kg": None,
    }])


def _stat_row() -> dict:
    return dict(
        match_date="2026-01-05", opponent="雲林美津濃", sets_played=3,
        attack_total=10, attack_points=5, block_points=1,
        serve_total=5, serve_points=1, receive_total=5, receive_excellent=3,
        dig_total=5, dig_excellent=2, set_total=0, set_excellent=0,
        total_points=7, is_golden_set=0,
    )


def _match_row(season: str) -> dict:
    return dict(
        game_id=301, gender="M", season=season, match_date="2026-01-05",
        venue="台南", round_name="例行賽 Week 5", game_label="Game 301",
        is_golden_set=0, home_team="屏東台電", away_team="雲林美津濃",
        home_set1=25, home_set2=25, home_set3=25, home_set4=None, home_set5=None,
        home_total=75, away_set1=20, away_set2=18, away_set3=22,
        away_set4=None, away_set5=None, away_total=60,
        home_sets_won=3, away_sets_won=0,
    )


def _run_pipeline_once(engine, season: str) -> None:
    df = _roster_df()
    insert_teams(engine, df)
    insert_players(engine, df, season=season)
    with engine.begin() as conn:
        pid = conn.execute(
            text("SELECT player_id FROM players WHERE name = '李元' AND season = :s"),
            {"s": season},
        ).scalar_one()
    upsert_stats(engine, pid, [_stat_row()], season)
    upsert_match(engine, _match_row(season))


def _table_counts(engine) -> dict[str, int]:
    with engine.begin() as conn:
        return {
            table: conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            for table in ["teams", "players", "player_match_stats", "matches"]
        }


def _table_snapshot(engine, table: str) -> list[tuple]:
    """Fetch all rows from table, ordered by primary key for consistent comparison."""
    pk_map = {
        "teams": "team_id",
        "players": "player_id",
        "player_match_stats": "stat_id",
        "matches": "match_id",
    }
    pk = pk_map.get(table)
    with engine.begin() as conn:
        rows = conn.execute(
            text(f"SELECT * FROM {table} ORDER BY {pk}")
        ).fetchall()
    return [tuple(row) for row in rows]


def test_full_pipeline_rerun_is_idempotent(sqlite_engine):
    _run_pipeline_once(sqlite_engine, "2025-26")
    snapshots_first = {
        table: _table_snapshot(sqlite_engine, table)
        for table in ["teams", "players", "player_match_stats", "matches"]
    }

    _run_pipeline_once(sqlite_engine, "2025-26")
    snapshots_second = {
        table: _table_snapshot(sqlite_engine, table)
        for table in ["teams", "players", "player_match_stats", "matches"]
    }

    # Verify full data snapshots are identical (every column value)
    assert snapshots_first == snapshots_second
    # Verify exact row counts
    assert len(snapshots_second["teams"]) == 1
    assert len(snapshots_second["players"]) == 1
    assert len(snapshots_second["player_match_stats"]) == 1
    assert len(snapshots_second["matches"]) == 1


def test_new_season_rerun_does_not_touch_old_season_rows(sqlite_engine):
    _run_pipeline_once(sqlite_engine, "2025-26")
    _run_pipeline_once(sqlite_engine, "2026-27")

    counts = _table_counts(sqlite_engine)
    assert counts["players"] == 2
    assert counts["player_match_stats"] == 2
    assert counts["matches"] == 2

    with sqlite_engine.begin() as conn:
        old_total = conn.execute(
            text("SELECT home_total FROM matches WHERE season = '2025-26'")
        ).scalar_one()
    assert old_total == 75


def test_database_url_switch_between_sqlite_and_postgresql(monkeypatch, tmp_path):
    import src.utils.db_config as db_config
    from src.etl.db_loader import init_db

    # Test 1: Dialect detection via DATABASE_URL
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(db_config, "DB_PATH", tmp_path / "default.db")
    db_config.reset_engine()
    sqlite_eng = db_config.get_engine()
    assert sqlite_eng.dialect.name == "sqlite"

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/tvl")
    db_config.reset_engine()
    pg_eng = db_config.get_engine()
    assert pg_eng.dialect.name == "postgresql"

    # Test 2: SQL compilation in both dialects
    from sqlalchemy import text as sa_text

    upsert_sql = sa_text(
        "INSERT INTO teams (team_id, team_name, gender) VALUES (:team_id, :team_name, :gender) "
        "ON CONFLICT (team_id, gender) DO UPDATE SET team_name = excluded.team_name"
    )
    compiled_sqlite = str(upsert_sql.compile(dialect=sqlite_eng.dialect))
    compiled_pg = str(upsert_sql.compile(dialect=pg_eng.dialect))
    assert "ON CONFLICT" in compiled_sqlite
    assert "ON CONFLICT" in compiled_pg

    # Test 3: Real isolation with two separate SQLite files
    db_a_path = str(tmp_path / "db_a.db")
    db_b_path = str(tmp_path / "db_b.db")

    # Create engine for DB A
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_a_path}")
    db_config.reset_engine()
    eng_a = db_config.get_engine()
    init_db(eng_a)

    # Create engine for DB B
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_b_path}")
    db_config.reset_engine()
    eng_b = db_config.get_engine()
    init_db(eng_b)

    # Write different data to each DB
    team_a_df = pd.DataFrame([{
        "team_id": 1, "team_name": "A隊", "gender": "M",
        "jersey_number": 1, "name": "選手A", "position": "OH",
        "dob": None, "height_cm": None, "weight_kg": None,
    }])
    insert_teams(eng_a, team_a_df)

    team_b_df = pd.DataFrame([{
        "team_id": 2, "team_name": "B隊", "gender": "M",
        "jersey_number": 2, "name": "選手B", "position": "OP",
        "dob": None, "height_cm": None, "weight_kg": None,
    }])
    insert_teams(eng_b, team_b_df)

    # Verify isolation: A's DB has only A's data
    with eng_a.begin() as conn:
        a_teams = conn.execute(text("SELECT team_name FROM teams ORDER BY team_id")).scalars().all()
    assert a_teams == ["A隊"], f"Expected ['A隊'] in DB A, got {a_teams}"

    # Verify isolation: B's DB has only B's data
    with eng_b.begin() as conn:
        b_teams = conn.execute(text("SELECT team_name FROM teams ORDER BY team_id")).scalars().all()
    assert b_teams == ["B隊"], f"Expected ['B隊'] in DB B, got {b_teams}"

    # Cleanup
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db_config.reset_engine()
