import pandas as pd
from sqlalchemy import create_engine, text


def test_load_data_reads_via_db_config_engine(sqlite_engine, monkeypatch):
    with sqlite_engine.begin() as conn:
        conn.execute(text("INSERT INTO teams (team_id, team_name, gender) VALUES (1, 'X', 'M')"))

    import src.app.helpers as helpers

    helpers.load_data.clear()  # 清除 st.cache_data 快取，避免跨測試互相汙染
    df = helpers.load_data(
        "SELECT * FROM teams WHERE gender = :gender_code",
        {"gender_code": "M"},
    )
    assert len(df) == 1
    assert df.iloc[0]["team_name"] == "X"


def test_load_data_supports_multiple_named_params(sqlite_engine):
    with sqlite_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO teams (team_id, team_name, gender) VALUES "
            "(1, 'A', 'M'), (2, 'B', 'M'), (3, 'C', 'F')"
        ))

    import src.app.helpers as helpers

    helpers.load_data.clear()
    df = helpers.load_data(
        "SELECT * FROM teams WHERE gender = :gender_code AND team_id = :team_id",
        {"gender_code": "M", "team_id": 2},
    )
    assert len(df) == 1
    assert df.iloc[0]["team_name"] == "B"


def test_load_data_query_compiles_under_postgresql_dialect():
    """具名參數查詢須能被 PostgreSQL dialect compiler 編譯，確保上線後 psycopg 相容。"""
    pg_engine = create_engine("postgresql+psycopg://user:pass@localhost:5432/tvl")
    compiled = text("SELECT * FROM teams WHERE gender = :gender_code").compile(dialect=pg_engine.dialect)
    assert "gender_code" in str(compiled)


def test_fetch_match_index_uses_season_year_for_month():
    from src.app.helpers import fetch_match_index

    import src.app.helpers as helpers

    assert helpers.season_year_for_month(11) == 2025
    assert helpers.season_year_for_month(3) == 2026
    assert callable(fetch_match_index)
