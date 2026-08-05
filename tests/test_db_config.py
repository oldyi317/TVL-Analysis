import src.utils.db_config as db_config


def test_get_engine_defaults_to_sqlite_when_database_url_unset(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db_config.reset_engine()
    engine = db_config.get_engine()
    assert engine.dialect.name == "sqlite"
    db_config.reset_engine()


def test_get_engine_uses_database_url_when_set(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/tvl"
    )
    db_config.reset_engine()
    engine = db_config.get_engine()
    assert engine.dialect.name == "postgresql"
    assert engine.driver == "psycopg"
    db_config.reset_engine()
    monkeypatch.delenv("DATABASE_URL", raising=False)


def test_get_engine_enables_sqlite_foreign_keys(monkeypatch, tmp_path):
    db_path = tmp_path / "fk_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    db_config.reset_engine()
    engine = db_config.get_engine()
    with engine.connect() as conn:
        from sqlalchemy import text

        fk_status = conn.exec_driver_sql("PRAGMA foreign_keys").scalar()
    assert fk_status == 1
    db_config.reset_engine()
    monkeypatch.delenv("DATABASE_URL", raising=False)
