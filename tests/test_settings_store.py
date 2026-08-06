from sqlalchemy import create_engine

from src.app.llm_client import resolve_llm_config
from src.app.settings_store import get_setting, set_setting


def test_get_setting_returns_none_when_missing(sqlite_engine):
    assert get_setting(sqlite_engine, "does_not_exist") is None


def _schema_less_engine(tmp_path, name: str):
    """建立完全沒跑過 init_db 的 SQLite engine，模擬 dashboard 從未呼叫 ETL 的情境。"""
    return create_engine(f"sqlite:///{tmp_path / name}", future=True)


def test_get_setting_on_schema_less_db_returns_none_without_raising(tmp_path):
    """app_settings 表不存在時（dashboard 從未呼叫 init_db），get_setting 應自癒回傳 None。"""
    engine = _schema_less_engine(tmp_path, "schema_less_get.db")
    assert get_setting(engine, "mlis_base_url") is None


def test_resolve_llm_config_on_schema_less_db_returns_none_without_raising(tmp_path, monkeypatch):
    """對應 Critical 發現：dashboard 未呼叫 init_db 時，resolve_llm_config 不應拋出 OperationalError。"""
    for key in ("MLIS_BASE_URL", "MLIS_API_KEY", "MLIS_MODEL"):
        monkeypatch.delenv(key, raising=False)
    engine = _schema_less_engine(tmp_path, "schema_less_resolve.db")
    assert resolve_llm_config(engine) is None


def test_set_setting_on_schema_less_db_creates_table_and_persists(tmp_path):
    """set_setting 應能在完全沒有 schema 的 DB 上自行建表並成功寫入。"""
    engine = _schema_less_engine(tmp_path, "schema_less_set.db")
    set_setting(engine, "mlis_model", "qwen2.5-72b")
    assert get_setting(engine, "mlis_model") == "qwen2.5-72b"


def test_set_setting_then_get_setting_roundtrip(sqlite_engine):
    set_setting(sqlite_engine, "mlis_model", "qwen2.5-72b")
    assert get_setting(sqlite_engine, "mlis_model") == "qwen2.5-72b"


def test_set_setting_upserts_existing_key(sqlite_engine):
    set_setting(sqlite_engine, "mlis_model", "qwen-a")
    set_setting(sqlite_engine, "mlis_model", "qwen-b")
    assert get_setting(sqlite_engine, "mlis_model") == "qwen-b"


def test_set_setting_does_not_affect_other_keys(sqlite_engine):
    set_setting(sqlite_engine, "mlis_base_url", "http://a")
    set_setting(sqlite_engine, "mlis_model", "qwen-a")
    assert get_setting(sqlite_engine, "mlis_base_url") == "http://a"
    assert get_setting(sqlite_engine, "mlis_model") == "qwen-a"
