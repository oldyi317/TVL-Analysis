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


def _find_existing_player_id(
    conn: sqlite3.Connection, team_id: int, gender: str,
    jersey_number, name: str,
) -> int | None:
    """用自然鍵 (team_id, gender, jersey_number, name) 找既有 player_id，找不到回傳 None。"""
    row = conn.execute(
        """SELECT player_id FROM players
           WHERE team_id = ? AND gender = ? AND name = ?
             AND (jersey_number = ? OR (jersey_number IS NULL AND ? IS NULL))""",
        (team_id, gender, name, jersey_number, jersey_number),
    ).fetchone()
    return row[0] if row else None


def upsert_players(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """
    用自然鍵 (team_id, gender, jersey_number, name) upsert players 表。
    已存在的球員只更新 position/dob/height_cm/weight_kg，保留原 player_id，
    避免 player_match_stats 的 FK 因 player_id 改變而斷裂。
    """
    player_cols = [
        "team_id", "gender", "jersey_number", "name",
        "position", "dob", "height_cm", "weight_kg",
    ]
    players = df[player_cols]

    n_inserted = 0
    n_updated = 0
    for row in players.itertuples(index=False):
        existing_id = _find_existing_player_id(
            conn, row.team_id, row.gender, row.jersey_number, row.name,
        )
        if existing_id is None:
            conn.execute(
                """INSERT INTO players
                   (team_id, gender, jersey_number, name, position, dob, height_cm, weight_kg)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (row.team_id, row.gender, row.jersey_number, row.name,
                 row.position, row.dob, row.height_cm, row.weight_kg),
            )
            n_inserted += 1
        else:
            conn.execute(
                """UPDATE players
                   SET position = ?, dob = ?, height_cm = ?, weight_kg = ?
                   WHERE player_id = ?""",
                (row.position, row.dob, row.height_cm, row.weight_kg, existing_id),
            )
            n_updated += 1

    conn.commit()
    logger.info("players 表 upsert 完成：新增 %d 筆、更新 %d 筆", n_inserted, n_updated)


def verify(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    驗證查詢：女子組中位置為舉球員 (S) 且身高 > 170 cm 的球員。
    """
    query = """
        SELECT p.name, t.team_name, p.height_cm
        FROM players p
        JOIN teams t ON p.team_id = t.team_id AND p.gender = t.gender
        WHERE p.gender = 'F'
          AND p.position = 'S'
          AND p.height_cm > 170
        ORDER BY p.height_cm DESC
    """
    return pd.read_sql_query(query, conn)


def main():
    conn = get_connection()

    try:
        init_db(conn)
        df = load_csv()
        upsert_teams(conn, df)
        upsert_players(conn, df)

        result = verify(conn)
        print("\n===== 驗證查詢：女子組舉球員 (S)，身高 > 170cm =====")
        print(result.head(10).to_string(index=False))
    finally:
        conn.close()

    logger.info("資料庫載入完成：%s", DB_PATH)


if __name__ == "__main__":
    main()
