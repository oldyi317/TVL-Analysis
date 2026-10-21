"""
TVL 資料庫載入模組
讀取 raw CSV → 經 cleaner 清洗 → 寫入 players 身分層與 season_rosters 賽季報名名單（球隊身分只來自 sql/team_seed.sql）。
"""

import argparse
import sqlite3
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

from src.etl.cleaner import load_raw, clean, quality_report
from src.utils.db_config import PROJECT_ROOT, DB_PATH, get_connection
from src.utils.logger import get_logger
from src.utils.teams import UnknownTeamError, apply_team_seed, resolve_team

logger = get_logger(__name__)

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "all_teams_roster.csv"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


def init_db(conn: sqlite3.Connection) -> None:
    """讀取 schema.sql 建立資料表（CREATE TABLE IF NOT EXISTS，冪等，不清空既有資料）。"""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema_sql)
    apply_team_seed(conn)
    logger.info("資料庫 Schema 已確認存在（未清空既有資料）")


OPTIONAL_COLS = ["height_cm", "weight_kg", "position"]


def load_csv(path: Path = CSV_PATH) -> pd.DataFrame:
    """讀取 raw CSV 並經 cleaner 清洗，確保資料品質後回傳；缺的身分與位置欄位補 None。"""
    df = load_raw(path)
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = None
    df = clean(df)
    quality_report(df)
    logger.info("清洗後資料：%d 筆", len(df))
    return df


def _find_existing_player_id(conn: sqlite3.Connection, gender: str, name: str) -> int | None:
    """用自然鍵 (name, gender) 找既有 player_id，找不到回傳 None。"""
    row = conn.execute(
        "SELECT player_id FROM players WHERE name = ? AND gender = ?",
        (name, gender),
    ).fetchone()
    return row[0] if row else None


def upsert_player_identity(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """
    用自然鍵 (name, gender) upsert players 表的身分欄位（height_cm/weight_kg）。
    team_id/jersey_number/position 由 roster_registrations 維護，本函式不寫入。
    """
    identity_cols = ["gender", "name", "height_cm", "weight_kg"]
    identities = df[identity_cols].drop_duplicates(subset=["name", "gender"])

    n_inserted = 0
    n_updated = 0
    for row in identities.itertuples(index=False):
        existing_id = _find_existing_player_id(conn, row.gender, row.name)
        if existing_id is None:
            conn.execute(
                "INSERT INTO players (name, gender, height_cm, weight_kg) VALUES (?, ?, ?, ?)",
                (row.name, row.gender, row.height_cm, row.weight_kg),
            )
            n_inserted += 1
        else:
            conn.execute(
                "UPDATE players SET height_cm = COALESCE(?, height_cm), "
                "weight_kg = COALESCE(?, weight_kg) WHERE player_id = ?",
                (row.height_cm, row.weight_kg, existing_id),
            )
            n_updated += 1

    conn.commit()
    logger.info("players 身分層 upsert 完成：新增 %d 筆、更新 %d 筆", n_inserted, n_updated)


def _sync_target_cup_id(conn: sqlite3.Connection, today: str) -> int | None:
    """matches 最新賽季；該季已結束（休賽期）回傳 None，避免把新名單寫進舊季。"""
    row = conn.execute(
        "SELECT cup_id, end_date FROM seasons WHERE cup_id = (SELECT MAX(cup_id) FROM matches)"
    ).fetchone()
    if row is None or today > row[1]:
        return None
    return row[0]


def _team_rows(conn: sqlite3.Connection, team_df: pd.DataFrame, team_id: int, gender: str, cup_id: int) -> list[tuple]:
    rows = []
    for row in team_df.itertuples(index=False):
        player_id = _find_existing_player_id(conn, gender, row.name)
        if player_id is None:
            logger.warning("找不到球員 %s（%s），略過", row.name, row.team_name)
            continue
        jersey = None if pd.isna(row.jersey_number) else int(row.jersey_number)
        position = getattr(row, "position", None)
        position = None if position is None or pd.isna(position) else position
        rows.append((player_id, team_id, gender, cup_id, jersey, position))
    return rows


def sync_season_rosters(
    conn: sqlite3.Connection, today: str, df: pd.DataFrame, cup_id: int | None = None
) -> None:
    """
    CSV → season_rosters 賽季報名名單（ADR 0004）：預設寫入 matches 最新且未結束的賽季，
    指定 cup_id 則直接寫入該季（歷史名單回填用）。
    隊伍依 (組別, 隊名) 解析本地 team_id，每隊整隊覆蓋；解析不到的隊伍與 0 人的隊伍略過不動。
    同一球員多列（如上、下半季）以最後一列為準。
    """
    if cup_id is None:
        cup_id = _sync_target_cup_id(conn, today)
    if cup_id is None:
        logger.warning("matches 最新賽季不存在或已結束，略過賽季報名名單")
        return

    for (csv_gender, team_name), team_df in df.groupby(["gender", "team_name"], sort=False):
        try:
            team_id, gender = resolve_team(conn, team_name)
        except UnknownTeamError as e:
            logger.warning("%s，略過該隊賽季報名名單", e)
            continue
        if gender != csv_gender:
            logger.warning("%s 解析為組別 %s，與名單組別 %s 不符，略過", team_name, gender, csv_gender)
            continue
        rows = _team_rows(conn, team_df, team_id, gender, cup_id)
        if not rows:
            logger.warning("%s 沒有可寫入的球員，保留既有賽季報名名單", team_name)
            continue
        conn.execute(
            "DELETE FROM season_rosters WHERE team_id = ? AND gender = ? AND cup_id = ?",
            (team_id, gender, cup_id),
        )
        conn.executemany(
            "INSERT OR REPLACE INTO season_rosters "
            "(player_id, team_id, gender, cup_id, jersey_number, position) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        logger.info("%s 賽季報名名單：%d 人（cup_id=%d）", team_name, len(rows), cup_id)
    conn.commit()


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
    parser = argparse.ArgumentParser(description="roster CSV → players 身分層 + season_rosters 賽季報名名單")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help=f"名單 CSV 路徑（預設 {CSV_PATH}）")
    parser.add_argument("--cup-id", type=int, default=None, help="指定寫入的賽季（歷史名單回填用；預設為 matches 最新且未結束的賽季）")
    args = parser.parse_args()
    conn = get_connection()

    try:
        init_db(conn)
        df = load_csv(args.csv)
        upsert_player_identity(conn, df)
        sync_season_rosters(conn, date.today().isoformat(), df, cup_id=args.cup_id)

        result = verify(conn)
        print("\n===== 驗證查詢：女子組身高前 10 高的球員 =====")
        print(result.head(10).to_string(index=False))
    finally:
        conn.close()

    logger.info("資料庫載入完成：%s", DB_PATH)


if __name__ == "__main__":
    main()
