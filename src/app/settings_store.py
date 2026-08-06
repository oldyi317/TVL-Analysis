"""
app_settings key-value 表存取模組。
供「系統設定」分頁與 llm_client 的設定讀取順序（DB → 環境變數）使用。
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, ProgrammingError


def ensure_settings_table(engine: Engine) -> None:
    """確保 app_settings 表存在（冪等）。dashboard 不會呼叫 ETL 的 init_db，
    表不存在時靠這個自癒，避免整個 app 因 OperationalError 崩潰。"""
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        ))


def get_setting(engine: Engine, key: str) -> str | None:
    """讀取單一設定值，不存在時回傳 None；app_settings 表尚未建立時也回傳 None（不拋例外）。"""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT value FROM app_settings WHERE key = :key"),
                {"key": key},
            ).first()
        return row[0] if row else None
    except (OperationalError, ProgrammingError):
        return None


def set_setting(engine: Engine, key: str, value: str) -> None:
    """寫入或更新單一設定值（upsert）。先確保表存在，讓儲存動作在任何環境下都能成功。"""
    ensure_settings_table(engine)
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO app_settings (key, value)
                VALUES (:key, :value)
                ON CONFLICT (key) DO UPDATE SET value = excluded.value
            """),
            {"key": key, "value": value},
        )
