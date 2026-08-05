"""
統一 Logger 設定模組
所有模組透過 get_logger(__name__) 取得 logger，避免重複呼叫 basicConfig。
LOG_LEVEL 環境變數可覆寫預設等級（預設 INFO）。
"""

import logging
import os

_CONFIGURED = False


def _resolve_level(level_name: str) -> int:
    """將等級名稱字串轉為 logging 等級數值，無法辨識時 fallback 為 INFO。"""
    return getattr(logging, level_name.upper(), logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    取得已設定好格式的 Logger。
    首次呼叫時設定 root handler，後續呼叫直接回傳。
    """
    global _CONFIGURED
    if not _CONFIGURED:
        level = _resolve_level(os.environ.get("LOG_LEVEL", "INFO"))
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
