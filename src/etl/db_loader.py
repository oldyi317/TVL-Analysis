"""
TVL 資料庫載入模組
讀取 raw CSV → 經 cleaner 清洗 → 正規化拆分為 teams / players 兩表並寫入 SQLite。
"""

import sqlite3
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from src.etl.cleaner import load_raw, clean, quality_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "all_teams_roster.csv"
DB_PATH = PROJECT_ROOT / "data" / "db" / "tvl_database.db"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


def init_db(conn: sqlite3.Connection) -> None:
    """讀取 schema.sql 建立資料表（DROP + CREATE，可重複執行）。"""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema_sql)
    logger.info("資料庫 Schema 建立完成")


def load_csv() -> pd.DataFrame:
    """讀取 raw CSV 並經 cleaner 清洗，確保資料品質後回傳。"""
    df = load_raw(CSV_PATH)
    df = clean(df)
    quality_report(df)
    logger.info("清洗後資料：%d 筆", len(df))
    return df


def insert_teams(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """萃取唯一球隊組合並寫入 teams 表（複合主鍵 team_id + gender）。"""
    teams = (
        df[["team_id", "team_name", "gender"]]
        .drop_duplicates()
        .sort_values(["gender", "team_id"])
    )
    conn.executemany(
        "INSERT INTO teams (team_id, team_name, gender) VALUES (?, ?, ?)",
        teams.values.tolist(),
    )
    conn.commit()
    logger.info("已寫入 teams 表：%d 筆", len(teams))


def insert_players(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """萃取球員欄位並寫入 players 表（player_id 自動遞增）。"""
    player_cols = [
        "team_id", "gender", "jersey_number", "name",
        "position", "dob", "height_cm", "weight_kg",
    ]
    players = df[player_cols]
    conn.executemany(
        """INSERT INTO players
           (team_id, gender, jersey_number, name, position, dob, height_cm, weight_kg)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        players.values.tolist(),
    )
    conn.commit()
    logger.info("已寫入 players 表：%d 筆", len(players))


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
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        init_db(conn)
        df = load_csv()
        insert_teams(conn, df)
        insert_players(conn, df)

        # 驗證查詢
        result = verify(conn)
        print("\n===== 驗證查詢：女子組舉球員 (S)，身高 > 170cm =====")
        print(result.head(10).to_string(index=False))
    finally:
        conn.close()

    logger.info("資料庫載入完成：%s", DB_PATH)


if __name__ == "__main__":
    main()
