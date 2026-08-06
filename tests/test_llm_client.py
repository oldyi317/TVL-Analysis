"""
llm_client 測試：全部使用 httpx.MockTransport 模擬 MLIS 的 OpenAI 相容 endpoint，
不連真實網路（真實 endpoint 於 PCAI 上另行驗證）。
"""

import json

import httpx
import pytest
from openai import OpenAI

from src.app.llm_client import (
    LLMConfig,
    generate_report,
    resolve_llm_config,
    test_connection as check_connection,
)


def _make_mock_client(handler) -> OpenAI:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    return OpenAI(base_url="http://fake-mlis.local/v1", api_key="test-key", http_client=http_client, max_retries=0)


def _success_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["model"] == "qwen-test"
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": "qwen-test",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "測試戰報內容"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def test_generate_report_returns_content_on_success():
    config = LLMConfig(base_url="http://fake-mlis.local/v1", api_key="test-key", model="qwen-test")
    client = _make_mock_client(_success_handler)

    result = generate_report(config, "系統提示", "使用者提示", client=client)

    assert result == "測試戰報內容"


def test_generate_report_retries_then_raises_friendly_error(monkeypatch):
    import src.app.llm_client as llm_client

    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)  # 測試不等待真實間隔

    call_count = {"n": 0}

    def _failing_handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(500, json={"error": {"message": "internal error"}})

    config = LLMConfig(base_url="http://fake-mlis.local/v1", api_key="test-key", model="qwen-test")
    client = _make_mock_client(_failing_handler)

    with pytest.raises(RuntimeError, match="MLIS 服務呼叫失敗"):
        generate_report(config, "系統提示", "使用者提示", client=client)

    assert call_count["n"] == 3  # 初次 + 最多 2 次重試


def test_test_connection_reports_success(monkeypatch):
    config = LLMConfig(base_url="http://fake-mlis.local/v1", api_key="test-key", model="qwen-test")

    import src.app.llm_client as llm_client
    monkeypatch.setattr(llm_client, "_build_client", lambda cfg: _make_mock_client(_success_handler))

    ok, message = check_connection(config)

    assert ok is True
    assert message == "連線成功"


def test_test_connection_reports_failure(monkeypatch):
    def _failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "unauthorized"}})

    config = LLMConfig(base_url="http://fake-mlis.local/v1", api_key="bad-key", model="qwen-test")

    import src.app.llm_client as llm_client
    monkeypatch.setattr(llm_client, "_build_client", lambda cfg: _make_mock_client(_failing_handler))

    ok, message = check_connection(config)

    assert ok is False
    assert "連線失敗" in message


def test_resolve_llm_config_prefers_db_over_env(sqlite_engine, monkeypatch):
    from src.app.settings_store import set_setting

    monkeypatch.setenv("MLIS_BASE_URL", "http://env-endpoint/v1")
    monkeypatch.setenv("MLIS_API_KEY", "env-key")
    monkeypatch.setenv("MLIS_MODEL", "env-model")

    set_setting(sqlite_engine, "mlis_base_url", "http://db-endpoint/v1")
    set_setting(sqlite_engine, "mlis_api_key", "db-key")
    set_setting(sqlite_engine, "mlis_model", "db-model")

    config = resolve_llm_config(sqlite_engine)

    assert config == LLMConfig(base_url="http://db-endpoint/v1", api_key="db-key", model="db-model")


def test_resolve_llm_config_falls_back_to_env_when_db_empty(sqlite_engine, monkeypatch):
    monkeypatch.setenv("MLIS_BASE_URL", "http://env-endpoint/v1")
    monkeypatch.setenv("MLIS_API_KEY", "env-key")
    monkeypatch.setenv("MLIS_MODEL", "env-model")

    config = resolve_llm_config(sqlite_engine)

    assert config == LLMConfig(base_url="http://env-endpoint/v1", api_key="env-key", model="env-model")


def test_resolve_llm_config_returns_none_when_nothing_set(sqlite_engine, monkeypatch):
    for key in ("MLIS_BASE_URL", "MLIS_API_KEY", "MLIS_MODEL"):
        monkeypatch.delenv(key, raising=False)

    assert resolve_llm_config(sqlite_engine) is None
