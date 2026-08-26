"""
TVL 資料庫載入模組
讀取 raw CSV → 經 cleaner 清洗 → 正規化拆分為 teams / players 兩表並寫入 SQLite。
"""

import sqlite3
import numpy as np
import pandas as pd

from src.etl.cleaner import load_raw, clean, quality_report
from src.utils.db_config import PROJECT_ROOT, DB_PATH, get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "all_teams_roster.csv"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


def init_db(conn: sqlite3.Connection) -> None:
    """讀取 schema.sql 建立資料表（CREATE TABLE IF NOT EXISTS，冪等，不清空既有資料）。"""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema_sql)
    logger.info("資料庫 Schema 已確認存在（未清空既有資料）")


def load_csv() -> pd.DataFrame:
    """讀取 raw CSV 並經 cleaner 清洗，確保資料品質後回傳。"""
    df = load_raw(CSV_PATH)
    df = clean(df)
    quality_report(df)
    logger.info("清洗後資料：%d 筆", len(df))
    return df


def upsert_teams(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """萃取唯一球隊組合並 upsert 進 teams 表（複合主鍵 team_id + gender，已存在則略過）。"""
    teams = (
        df[["team_id", "team_name", "gender"]]
        .drop_duplicates()
        .sort_values(["gender", "team_id"])
    )
    conn.executemany(
        "INSERT OR IGNORE INTO teams (team_id, team_name, gender) VALUES (?, ?, ?)",
        teams.values.tolist(),
    )
    conn.commit()
    logger.info("已 upsert teams 表：%d 筆", len(teams))


def _find_existing_player_id(conn: sqlite3.Connection, gender: str, name: str) -> int | None:
    """用自然鍵 (name, gender) 找既有 player_id，找不到回傳 None。"""
    row = conn.execute(
        "SELECT player_id FROM players WHERE name = ? AND gender = ?",
        (name, gender),
    ).fetchone()
    return row[0] if row else None


def upsert_player_identity(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """
    用自然鍵 (name, gender) upsert players 表的身分欄位（dob/height_cm/weight_kg）。
    team_id/jersey_number/position 由 roster_registrations 維護，本函式不寫入。
    """
    identity_cols = ["gender", "name", "dob", "height_cm", "weight_kg"]
    identities = df[identity_cols].drop_duplicates(subset=["name", "gender"])

    n_inserted = 0
    n_updated = 0
    for row in identities.itertuples(index=False):
        existing_id = _find_existing_player_id(conn, row.gender, row.name)
        if existing_id is None:
            conn.execute(
                "INSERT INTO players (name, gender, dob, height_cm, weight_kg) VALUES (?, ?, ?, ?, ?)",
                (row.name, row.gender, row.dob, row.height_cm, row.weight_kg),
            )
            n_inserted += 1
        else:
            conn.execute(
                "UPDATE players SET dob = ?, height_cm = ?, weight_kg = ? WHERE player_id = ?",
                (row.dob, row.height_cm, row.weight_kg, existing_id),
            )
            n_updated += 1

    conn.commit()
    logger.info("players 身分層 upsert 完成：新增 %d 筆、更新 %d 筆", n_inserted, n_updated)


def verify(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    驗證查詢：女子組身高最高的 10 位球員（身分層驗證，不再依賴 position，
    position 已搬到 roster_registrations，見 Phase 2 遷移）。
    """
    query = """
        SELECT name, gender, height_cm
        FROM players
        WHERE gender = 'F' AND height_cm IS NOT NULL
        ORDER BY height_cm DESC
    """
    return pd.read_sql_query(query, conn)


def main():
    conn = get_connection()

    try:
        init_db(conn)
        df = load_csv()
        upsert_teams(conn, df)
        upsert_player_identity(conn, df)

        result = verify(conn)
        print("\n===== 驗證查詢：女子組身高前 10 高的球員 =====")
        print(result.head(10).to_string(index=False))
    finally:
        conn.close()

    logger.info("資料庫載入完成：%s", DB_PATH)


if __name__ == "__main__":
    main()
