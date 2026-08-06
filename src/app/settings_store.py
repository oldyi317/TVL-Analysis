"""
app_settings key-value 表存取模組。
供「系統設定」分頁與 llm_client 的設定讀取順序（DB → 環境變數）使用。
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine


def get_setting(engine: Engine, key: str) -> str | None:
    """讀取單一設定值，不存在時回傳 None。"""
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT value FROM app_settings WHERE key = :key"),
            {"key": key},
        ).first()
    return row[0] if row else None


def set_setting(engine: Engine, key: str, value: str) -> None:
    """寫入或更新單一設定值（upsert）。"""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO app_settings (key, value)
                VALUES (:key, :value)
                ON CONFLICT (key) DO UPDATE SET value = excluded.value
            """),
            {"key": key, "value": value},
        )
