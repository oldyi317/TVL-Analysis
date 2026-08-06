"""
Tab 7：系統設定
設定 PCAI MLIS 的 OpenAI 相容 endpoint（base URL、model、API key），供「每周戰報」分頁呼叫。
整個 dashboard 已在平台 SSO 之後，本頁不另做角色權限。
"""

import streamlit as st

from src.app.llm_client import LLMConfig, test_connection
from src.app.settings_store import get_setting, set_setting
from src.utils.db_config import get_engine

SETTING_KEYS = {
    "base_url": "mlis_base_url",
    "model": "mlis_model",
    "api_key": "mlis_api_key",
    "verify_ssl": "mlis_verify_ssl",
}


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() not in ("false", "0", "no")


def _mask_secret(value: str) -> str:
    """固定寬度遮罩，不洩漏金鑰長度：長度 > 8 時保留最後 4 碼，其餘一律 8 個遮罩字元。"""
    if not value:
        return ""
    if len(value) > 8:
        return f"{'•' * 8}{value[-4:]}"
    return "•" * 8


def render(ctx: dict) -> None:
    st.subheader("系統設定")
    st.caption("設定 PCAI MLIS 的 OpenAI 相容 endpoint，供「每周戰報」分頁呼叫 AI 產生戰報。")

    engine = get_engine()

    current_base_url = get_setting(engine, SETTING_KEYS["base_url"]) or ""
    current_model = get_setting(engine, SETTING_KEYS["model"]) or ""
    current_api_key = get_setting(engine, SETTING_KEYS["api_key"]) or ""
    current_verify_ssl_raw = get_setting(engine, SETTING_KEYS["verify_ssl"])
    current_verify_ssl = _parse_bool(current_verify_ssl_raw) if current_verify_ssl_raw is not None else True

    with st.form("mlis_settings_form"):
        base_url = st.text_input(
            "Endpoint Base URL", value=current_base_url,
            placeholder="http://mlis-qwen.example.svc.cluster.local/v1",
            key="settings_base_url",
        )
        model = st.text_input(
            "Model 名稱", value=current_model,
            placeholder="qwen2.5-72b-instruct",
            key="settings_model",
        )
        api_key = st.text_input(
            "API Key（留空表示不變更既有金鑰）", value="", type="password",
            placeholder=_mask_secret(current_api_key) or "尚未設定",
            key="settings_api_key",
        )
        verify_ssl = st.checkbox(
            "驗證 TLS 憑證（內部自簽憑證環境可取消勾選）",
            value=current_verify_ssl,
            key="settings_verify_ssl",
        )
        submitted = st.form_submit_button("儲存設定")

    if submitted:
        if base_url.strip():
            set_setting(engine, SETTING_KEYS["base_url"], base_url.strip())
        if model.strip():
            set_setting(engine, SETTING_KEYS["model"], model.strip())
        if api_key.strip():
            set_setting(engine, SETTING_KEYS["api_key"], api_key.strip())
        set_setting(engine, SETTING_KEYS["verify_ssl"], "true" if verify_ssl else "false")
        st.success("設定已儲存。")
        # 重新讀取：若不重新讀取，儲存當下這次 rerun 仍會用表單提交「前」讀到的舊
        # current_api_key 渲染下方的遮罩提示，導致剛存完金鑰卻仍顯示「尚未設定」。
        current_base_url = get_setting(engine, SETTING_KEYS["base_url"]) or ""
        current_model = get_setting(engine, SETTING_KEYS["model"]) or ""
        current_api_key = get_setting(engine, SETTING_KEYS["api_key"]) or ""

    st.markdown("---")
    if current_api_key:
        st.caption(f"目前已儲存的 API Key：`{_mask_secret(current_api_key)}`")
    else:
        st.caption("尚未設定 API Key。")

    if st.button("測試連線", key="settings_test_connection"):
        test_base_url = (base_url or current_base_url).strip()
        test_model = (model or current_model).strip()
        test_api_key = (api_key or current_api_key).strip()
        if not (test_base_url and test_model and test_api_key):
            st.warning("請先完整填寫 Endpoint Base URL、Model 名稱與 API Key。")
        else:
            config = LLMConfig(base_url=test_base_url, api_key=test_api_key, model=test_model)
            ok, message = test_connection(config)
            if ok:
                st.success(message)
            else:
                st.error(message)
