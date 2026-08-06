"""
LLM 呼叫層：透過 OpenAI 相容 API 呼叫 PCAI MLIS 部署的模型，取代原本的 Gemini 呼叫。
設定讀取順序：DB app_settings（UI 設定）→ 環境變數 → 皆無則回傳 None，由呼叫端顯示引導訊息。
"""

import os
import time
from dataclasses import dataclass

from openai import OpenAI
from sqlalchemy.engine import Engine

from src.app.settings_store import get_setting
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.0


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str


def resolve_llm_config(engine: Engine) -> LLMConfig | None:
    """讀取順序：DB app_settings → 環境變數；任一欄位缺漏則回傳 None。"""
    base_url = get_setting(engine, "mlis_base_url") or os.environ.get("MLIS_BASE_URL")
    api_key = get_setting(engine, "mlis_api_key") or os.environ.get("MLIS_API_KEY")
    model = get_setting(engine, "mlis_model") or os.environ.get("MLIS_MODEL")
    if not (base_url and api_key and model):
        return None
    return LLMConfig(base_url=base_url, api_key=api_key, model=model)


def _build_client(config: LLMConfig) -> OpenAI:
    """建立 OpenAI 相容 client。max_retries=0：openai SDK 預設會自己重試 2 次，
    若不關閉，會與下方 generate_report 的重試邏輯疊加，讓一次失敗變成多達 9 次實際請求。"""
    return OpenAI(base_url=config.base_url, api_key=config.api_key, max_retries=0)


def generate_report(
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    *,
    client: OpenAI | None = None,
) -> str:
    """呼叫 MLIS OpenAI 相容 endpoint 產生戰報文字，最多重試 MAX_RETRIES 次、間隔 RETRY_DELAY_SECONDS 秒。"""
    active_client = client or _build_client(config)

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = active_client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=8192,
                temperature=0.7,
            )
            content = response.choices[0].message.content or ""
            if not content:
                raise ValueError("MLIS 回應內容為空")
            return content
        except Exception as e:  # noqa: BLE001 - 統一轉為友善錯誤，由呼叫端顯示
            last_error = e
            logger.warning("MLIS 呼叫失敗（第 %d/%d 次）：%s", attempt + 1, MAX_RETRIES + 1, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"MLIS 服務呼叫失敗，已重試 {MAX_RETRIES} 次：{last_error}"
    ) from last_error


def test_connection(config: LLMConfig) -> tuple[bool, str]:
    """實際打一次 endpoint 驗證設定是否可用，供「系統設定」頁的測試連線按鈕使用。"""
    try:
        client = _build_client(config)
        client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
        )
        return True, "連線成功"
    except Exception as e:  # noqa: BLE001
        logger.warning("MLIS 連線測試失敗：%s", e)
        return False, f"連線失敗（{type(e).__name__}），詳細錯誤已寫入 log"
