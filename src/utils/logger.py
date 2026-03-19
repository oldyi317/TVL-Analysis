"""
統一 Logger 設定模組
所有模組透過 get_logger(__name__) 取得 logger，避免重複呼叫 basicConfig。
"""

import logging

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """
    取得已設定好格式的 Logger。
    首次呼叫時設定 root handler，後續呼叫直接回傳。
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
