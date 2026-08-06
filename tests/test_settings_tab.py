"""AppTest UI 測試：系統設定分頁（endpoint / model / API key + 測試連線）。"""

from streamlit.testing.v1 import AppTest


def test_settings_tab_shows_empty_state_when_nothing_saved(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        from src.app.tabs import settings_tab

        settings_tab.render({})

    at = AppTest.from_function(_harness)
    at.run(timeout=60)
    assert not at.exception
    captions = [c.value for c in at.caption]
    assert any("尚未設定" in c for c in captions)


def test_settings_tab_save_then_reload_shows_saved_values(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        from src.app.tabs import settings_tab

        settings_tab.render({})

    at = AppTest.from_function(_harness)
    at.run(timeout=60)

    at.text_input(key="settings_base_url").set_value("http://mlis.example/v1").run()
    at.text_input(key="settings_model").set_value("qwen2.5-72b").run()
    at.text_input(key="settings_api_key").set_value("super-secret-key-1234").run()
    at.button(key="FormSubmitter:mlis_settings_form-儲存設定").click().run(timeout=60)

    assert not at.exception

    from src.app.settings_store import get_setting
    from src.utils.db_config import get_engine

    engine = get_engine()
    assert get_setting(engine, "mlis_base_url") == "http://mlis.example/v1"
    assert get_setting(engine, "mlis_model") == "qwen2.5-72b"
    assert get_setting(engine, "mlis_api_key") == "super-secret-key-1234"

    # 迴歸測試：儲存後「同一次」render（提交表單觸發的這次 rerun）就應顯示剛儲存的遮罩
    # 後 API Key，而不是表單提交前讀到的舊值（舊值在此案例中是空字串 → 會誤顯示「尚未設定」）。
    captions = [c.value for c in at.caption]
    assert any("目前已儲存的 API Key" in c and "1234" in c for c in captions), (
        f"儲存後同一次 render 應立刻反映剛儲存的值，實際 captions={captions}"
    )


def test_settings_tab_blank_submission_keeps_prior_values(sqlite_engine):
    """驗證：表單欄位留空提交時，既存設定不被覆蓋。"""
    from src.app.settings_store import set_setting

    # 先儲存初始值
    set_setting(sqlite_engine, "mlis_base_url", "http://initial.example/v1")
    set_setting(sqlite_engine, "mlis_model", "initial-model")
    set_setting(sqlite_engine, "mlis_api_key", "initial-key")

    def _harness():
        import sys
        sys.path.insert(0, ".")
        from src.app.tabs import settings_tab

        settings_tab.render({})

    at = AppTest.from_function(_harness)
    at.run(timeout=60)

    # 清空所有欄位並提交（模擬意外清空）
    at.text_input(key="settings_base_url").set_value("").run()
    at.text_input(key="settings_model").set_value("").run()
    at.text_input(key="settings_api_key").set_value("").run()
    at.button(key="FormSubmitter:mlis_settings_form-儲存設定").click().run(timeout=60)

    assert not at.exception

    from src.app.settings_store import get_setting
    from src.utils.db_config import get_engine

    engine = get_engine()
    # 驗證初始值未被覆蓋
    assert get_setting(engine, "mlis_base_url") == "http://initial.example/v1"
    assert get_setting(engine, "mlis_model") == "initial-model"
    assert get_setting(engine, "mlis_api_key") == "initial-key"


def test_settings_tab_test_connection_shows_success(sqlite_engine, monkeypatch):
    from src.app.settings_store import set_setting

    set_setting(sqlite_engine, "mlis_base_url", "http://mlis.example/v1")
    set_setting(sqlite_engine, "mlis_model", "qwen-test")
    set_setting(sqlite_engine, "mlis_api_key", "test-key")

    import src.app.tabs.settings_tab as settings_tab_module
    monkeypatch.setattr(settings_tab_module, "test_connection", lambda config: (True, "連線成功"))

    def _harness():
        import sys
        sys.path.insert(0, ".")
        from src.app.tabs import settings_tab

        settings_tab.render({})

    at = AppTest.from_function(_harness)
    at.run(timeout=60)
    at.button(key="settings_test_connection").click().run(timeout=60)

    assert not at.exception
    successes = [s.value for s in at.success]
    assert any("連線成功" in s for s in successes)
