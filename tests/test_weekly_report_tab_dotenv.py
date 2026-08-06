"""驗證 load_dotenv 改為選用：.env 不存在時應直接跳過，不拋例外。"""

import os


def test_load_env_if_present_skips_missing_file(tmp_path, monkeypatch):
    from src.app.tabs.weekly_report_tab import _load_env_if_present

    monkeypatch.delenv("TVL_TEST_ENV_KEY", raising=False)
    missing_path = tmp_path / "does_not_exist.env"

    _load_env_if_present(missing_path)  # 不應拋出例外

    assert "TVL_TEST_ENV_KEY" not in os.environ


def test_load_env_if_present_loads_existing_file(tmp_path, monkeypatch):
    from src.app.tabs.weekly_report_tab import _load_env_if_present

    monkeypatch.delenv("TVL_TEST_ENV_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("TVL_TEST_ENV_KEY=hello\n", encoding="utf-8")

    _load_env_if_present(env_file)

    assert os.environ.get("TVL_TEST_ENV_KEY") == "hello"
