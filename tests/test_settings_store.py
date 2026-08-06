from src.app.settings_store import get_setting, set_setting


def test_get_setting_returns_none_when_missing(sqlite_engine):
    assert get_setting(sqlite_engine, "does_not_exist") is None


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
