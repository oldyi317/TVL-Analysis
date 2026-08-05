"""
一次性資料遷移工具
讀取現有 SQLite（data/db/tvl_database.db，舊 schema 無 season 欄位），
寫入 DATABASE_URL 指向的目標資料庫（新 schema，含 season），
並為既有資料標上目前賽季（SEASON 設定）。
部署後執行一次，之後由 Airflow（計畫三）增量維護。
"""

import sqlite3
from pathlib import Path

from sqlalchemy import text

from src.etl.db_loader import init_db
from src.utils.constants import SEASON
from src.utils.db_config import PROJECT_ROOT, get_engine
from src.utils.logger import get_logger

logger = get_logger(__name__)

SOURCE_DB_PATH = PROJECT_ROOT / "data" / "db" / "tvl_database.db"


def _read_source_tables(sqlite_path: Path) -> dict:
    """讀取舊 SQLite DB 的四張表，回傳 {table_name: [dict, ...]}。"""
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {}
        for table in ["teams", "players", "player_match_stats", "matches"]:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            tables[table] = [dict(r) for r in rows]
        return tables
    finally:
        conn.close()


def migrate(sqlite_path: Path = SOURCE_DB_PATH, season: str = SEASON) -> dict[str, int]:
    """執行一次性遷移，回傳各表搬移筆數統計。"""
    if not sqlite_path.exists():
        raise FileNotFoundError(f"來源 SQLite 檔案不存在：{sqlite_path}")

    engine = get_engine()
    if engine.dialect.name == "sqlite":
        target_db = engine.url.database
        if target_db and target_db != ":memory:":
            if Path(target_db).resolve() == sqlite_path.resolve():
                raise RuntimeError(
                    "目標不可與來源相同，請設定 DATABASE_URL 指向新資料庫"
                    f"（例如 sqlite:///data/db/tvl_v2.db）。來源檔案：{sqlite_path}"
                )

    source = _read_source_tables(sqlite_path)
    init_db(engine)  # 於目標 DB 建立新 schema（含 season 欄位），冪等

    counts = {}
    with engine.begin() as conn:
        for row in source["teams"]:
            conn.execute(text("""
                INSERT INTO teams (team_id, team_name, gender)
                VALUES (:team_id, :team_name, :gender)
                ON CONFLICT (team_id, gender) DO UPDATE SET team_name = excluded.team_name
            """), row)
        counts["teams"] = len(source["teams"])

        for row in source["players"]:
            conn.execute(text("""
                INSERT INTO players
                    (player_id, team_id, gender, season, jersey_number, name, position, dob, height_cm, weight_kg)
                VALUES
                    (:player_id, :team_id, :gender, :season, :jersey_number, :name, :position, :dob, :height_cm, :weight_kg)
                ON CONFLICT (player_id) DO UPDATE SET
                    jersey_number = excluded.jersey_number,
                    position      = excluded.position,
                    dob           = excluded.dob,
                    height_cm     = excluded.height_cm,
                    weight_kg     = excluded.weight_kg
            """), {**row, "season": season})
        counts["players"] = len(source["players"])

        for row in source["player_match_stats"]:
            conn.execute(text("""
                INSERT INTO player_match_stats
                    (stat_id, player_id, season, match_date, opponent, sets_played,
                     attack_total, attack_points, block_points,
                     serve_total, serve_points,
                     receive_total, receive_excellent,
                     dig_total, dig_excellent,
                     set_total, set_excellent, total_points, is_golden_set)
                VALUES
                    (:stat_id, :player_id, :season, :match_date, :opponent, :sets_played,
                     :attack_total, :attack_points, :block_points,
                     :serve_total, :serve_points,
                     :receive_total, :receive_excellent,
                     :dig_total, :dig_excellent,
                     :set_total, :set_excellent, :total_points, :is_golden_set)
                ON CONFLICT (stat_id) DO UPDATE SET
                    sets_played       = excluded.sets_played,
                    attack_total      = excluded.attack_total,
                    attack_points     = excluded.attack_points,
                    block_points      = excluded.block_points,
                    serve_total       = excluded.serve_total,
                    serve_points      = excluded.serve_points,
                    receive_total     = excluded.receive_total,
                    receive_excellent = excluded.receive_excellent,
                    dig_total         = excluded.dig_total,
                    dig_excellent     = excluded.dig_excellent,
                    set_total         = excluded.set_total,
                    set_excellent     = excluded.set_excellent,
                    total_points      = excluded.total_points
            """), {**row, "season": season})
        counts["player_match_stats"] = len(source["player_match_stats"])

        for row in source["matches"]:
            conn.execute(text("""
                INSERT INTO matches (
                    match_id, game_id, gender, season, match_date, venue, round_name, game_label,
                    is_golden_set, home_team, away_team,
                    home_set1, home_set2, home_set3, home_set4, home_set5, home_total,
                    away_set1, away_set2, away_set3, away_set4, away_set5, away_total,
                    home_sets_won, away_sets_won
                ) VALUES (
                    :match_id, :game_id, :gender, :season, :match_date, :venue, :round_name, :game_label,
                    :is_golden_set, :home_team, :away_team,
                    :home_set1, :home_set2, :home_set3, :home_set4, :home_set5, :home_total,
                    :away_set1, :away_set2, :away_set3, :away_set4, :away_set5, :away_total,
                    :home_sets_won, :away_sets_won
                )
                ON CONFLICT (game_id, gender, season) DO UPDATE SET
                    match_date=excluded.match_date, venue=excluded.venue,
                    round_name=excluded.round_name, game_label=excluded.game_label,
                    is_golden_set=excluded.is_golden_set,
                    home_team=excluded.home_team, away_team=excluded.away_team,
                    home_set1=excluded.home_set1, home_set2=excluded.home_set2,
                    home_set3=excluded.home_set3, home_set4=excluded.home_set4,
                    home_set5=excluded.home_set5, home_total=excluded.home_total,
                    away_set1=excluded.away_set1, away_set2=excluded.away_set2,
                    away_set3=excluded.away_set3, away_set4=excluded.away_set4,
                    away_set5=excluded.away_set5, away_total=excluded.away_total,
                    home_sets_won=excluded.home_sets_won, away_sets_won=excluded.away_sets_won
            """), {**row, "season": season})
        counts["matches"] = len(source["matches"])

        # 校正 PostgreSQL identity 序列，避免後續自動產生的 ID 與剛遷入的既有 ID 衝突
        # （SQLite 沒有序列概念，此區塊只在方言為 postgresql 時執行）
        if engine.dialect.name == "postgresql":
            for table, pk in [
                ("players", "player_id"),
                ("player_match_stats", "stat_id"),
                ("matches", "match_id"),
            ]:
                conn.execute(text(
                    f"SELECT setval(pg_get_serial_sequence('{table}', '{pk}'), "
                    f"COALESCE((SELECT MAX({pk}) FROM {table}), 1))"
                ))

    logger.info("遷移完成：%s", counts)
    return counts


def main():
    counts = migrate()
    print("\n===== 一次性資料遷移完成 =====")
    for table, n in counts.items():
        print(f"{table}: {n} 筆")


if __name__ == "__main__":
    main()
