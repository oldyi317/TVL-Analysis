"""
TVL 資料庫載入模組
讀取 raw CSV → 經 cleaner 清洗 → 正規化拆分為 teams / players 兩表並 upsert 至資料庫（冪等）。
"""

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.etl.cleaner import load_raw, clean, quality_report
from src.utils.constants import SEASON
from src.utils.db_config import PROJECT_ROOT, get_engine
from src.utils.logger import get_logger

logger = get_logger(__name__)

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "all_teams_roster.csv"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"
SCHEMA_PATH_POSTGRES = PROJECT_ROOT / "sql" / "schema_postgres.sql"


def init_db(engine: Engine) -> None:
    """依 engine 方言選擇對應 schema 檔並逐句執行（CREATE TABLE IF NOT EXISTS，冪等）。"""
    path = SCHEMA_PATH_POSTGRES if engine.dialect.name == "postgresql" else SCHEMA_PATH
    schema_sql = path.read_text(encoding="utf-8")
    statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    logger.info("資料庫 Schema 建立完成（%s）", path.name)


def load_csv() -> pd.DataFrame:
    """讀取 raw CSV 並經 cleaner 清洗，確保資料品質後回傳。"""
    df = load_raw(CSV_PATH)
    df = clean(df)
    quality_report(df)
    logger.info("清洗後資料：%d 筆", len(df))
    return df


def insert_teams(engine: Engine, df: pd.DataFrame) -> None:
    """萃取唯一球隊組合並 upsert 至 teams 表（複合主鍵 team_id + gender）。"""
    teams = (
        df[["team_id", "team_name", "gender"]]
        .drop_duplicates()
        .sort_values(["gender", "team_id"])
    )
    rows = teams.to_dict("records")
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO teams (team_id, team_name, gender)
                VALUES (:team_id, :team_name, :gender)
                ON CONFLICT (team_id, gender) DO UPDATE SET
                    team_name = excluded.team_name
            """),
            rows,
        )
    logger.info("已 upsert teams 表：%d 筆", len(rows))


def insert_players(engine: Engine, df: pd.DataFrame, season: str = SEASON) -> None:
    """萃取球員欄位並 upsert 至 players 表（唯一鍵：team_id+gender+season+name）。"""
    player_cols = [
        "team_id", "gender", "jersey_number", "name",
        "position", "dob", "height_cm", "weight_kg",
    ]
    players = df[player_cols].copy()
    players["season"] = season
    rows = players.to_dict("records")
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO players
                    (team_id, gender, season, jersey_number, name, position, dob, height_cm, weight_kg)
                VALUES
                    (:team_id, :gender, :season, :jersey_number, :name, :position, :dob, :height_cm, :weight_kg)
                ON CONFLICT (team_id, gender, season, name) DO UPDATE SET
                    jersey_number = excluded.jersey_number,
                    position      = excluded.position,
                    dob           = excluded.dob,
                    height_cm     = excluded.height_cm,
                    weight_kg     = excluded.weight_kg
            """),
            rows,
        )
    logger.info("已 upsert players 表：%d 筆（season=%s）", len(rows), season)


def verify(engine: Engine) -> pd.DataFrame:
    """驗證查詢：女子組中位置為舉球員 (S) 且身高 > 170 cm 的球員。"""
    query = """
        SELECT p.name, t.team_name, p.height_cm
        FROM players p
        JOIN teams t ON p.team_id = t.team_id AND p.gender = t.gender
        WHERE p.gender = 'F'
          AND p.position = 'S'
          AND p.height_cm > 170
        ORDER BY p.height_cm DESC
    """
    return pd.read_sql_query(query, engine)


def main():
    engine = get_engine()

    init_db(engine)
    df = load_csv()
    insert_teams(engine, df)
    insert_players(engine, df)

    result = verify(engine)
    print("\n===== 驗證查詢：女子組舉球員 (S)，身高 > 170cm =====")
    print(result.head(10).to_string(index=False))

    logger.info("資料庫載入完成")


if __name__ == "__main__":
    main()
