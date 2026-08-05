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


def test_full_pipeline_rerun_is_idempotent(sqlite_engine):
    _run_pipeline_once(sqlite_engine, "2025-26")
    counts_first = _table_counts(sqlite_engine)

    _run_pipeline_once(sqlite_engine, "2025-26")
    counts_second = _table_counts(sqlite_engine)

    assert counts_first == counts_second
    assert counts_second == {"teams": 1, "players": 1, "player_match_stats": 1, "matches": 1}


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


def test_database_url_switch_between_sqlite_and_postgresql(monkeypatch):
    import src.utils.db_config as db_config

    monkeypatch.delenv("DATABASE_URL", raising=False)
    db_config.reset_engine()
    sqlite_eng = db_config.get_engine()
    assert sqlite_eng.dialect.name == "sqlite"

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/tvl")
    db_config.reset_engine()
    pg_eng = db_config.get_engine()
    assert pg_eng.dialect.name == "postgresql"

    # 驗證同一段 upsert SQL 可被兩種方言的 compiler 編譯（不需要真正連線）
    from sqlalchemy import text as sa_text

    upsert_sql = sa_text(
        "INSERT INTO teams (team_id, team_name, gender) VALUES (:team_id, :team_name, :gender) "
        "ON CONFLICT (team_id, gender) DO UPDATE SET team_name = excluded.team_name"
    )
    compiled_sqlite = str(upsert_sql.compile(dialect=sqlite_eng.dialect))
    compiled_pg = str(upsert_sql.compile(dialect=pg_eng.dialect))
    assert "ON CONFLICT" in compiled_sqlite
    assert "ON CONFLICT" in compiled_pg

    monkeypatch.delenv("DATABASE_URL", raising=False)
    db_config.reset_engine()
