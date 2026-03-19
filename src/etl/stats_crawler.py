"""
TVL 球員逐場數據爬蟲模組
從外部數據系統 (114.35.229.141) 抓取球員逐場統計，
透過球員姓名與本地 DB 關聯後寫入 player_match_stats 事實表。
"""

import re
import time
import sqlite3
import requests
import pandas as pd
from bs4 import BeautifulSoup

from pathlib import Path

try:
    from src.utils.db_config import DB_PATH, get_connection
    from src.utils.logger import get_logger
    from src.utils.constants import (
        EXT_BASE, EXT_CUP_ID as CUP_ID, EXT_HEADERS as HEADERS,
        SEASON_YEAR_MAP, DEFAULT_YEAR, EXT_TEAM_MAP,
    )
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    get_logger = logging.getLogger
    DB_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "tvl_database.db"

    def get_connection(foreign_keys=True):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        if foreign_keys:
            conn.execute("PRAGMA foreign_keys = ON")
        return conn

    EXT_BASE = "http://114.35.229.141"
    CUP_ID = 21
    SEASON_YEAR_MAP = {11: 2025, 12: 2025}
    DEFAULT_YEAR = 2026
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    EXT_TEAM_MAP = {
        1: (1, "M"), 2: (2, "M"), 3: (7, "M"), 4: (4, "M"), 5: (5, "M"),
        6: (4, "F"), 7: (3, "F"), 8: (5, "F"), 9: (7, "F"),
    }

logger = get_logger(__name__)


def normalize_name(name: str) -> str:
    """正規化姓名：去除全形/半形空白、轉小寫、去除不間斷空白。"""
    return re.sub(r"[\s\u3000\xa0]+", "", name).lower()


def safe_int(val: str) -> int | None:
    """安全轉換整數，失敗回傳 None。"""
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def parse_match_date(raw: str) -> str | None:
    """
    從 '311/01' 格式中萃取日期並轉為 YYYY-MM-DD。
    前面的數字是場次編號，後面 MM/DD 是日期。
    """
    m = re.search(r"(\d{1,2})/(\d{2})$", raw)
    if not m:
        return None
    month = int(m.group(1))
    day = int(m.group(2))
    year = SEASON_YEAR_MAP.get(month, DEFAULT_YEAR)
    return f"{year}-{month:02d}-{day:02d}"


def init_stats_table(conn: sqlite3.Connection) -> None:
    """建立 player_match_stats 事實表（若不存在）。"""
    conn.execute("DROP TABLE IF EXISTS player_match_stats")
    conn.execute("""
        CREATE TABLE player_match_stats (
            stat_id           INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id         INTEGER NOT NULL,
            match_date        DATE,
            opponent          TEXT,
            sets_played       INTEGER,
            attack_total      INTEGER,
            attack_points     INTEGER,
            block_points      INTEGER,
            serve_total       INTEGER,
            serve_points      INTEGER,
            receive_total     INTEGER,
            receive_excellent INTEGER,
            dig_total         INTEGER,
            dig_excellent     INTEGER,
            set_total         INTEGER,
            set_excellent     INTEGER,
            total_points      INTEGER,
            FOREIGN KEY (player_id) REFERENCES players (player_id)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pms_player_id ON player_match_stats(player_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pms_match_date ON player_match_stats(match_date)"
    )
    conn.commit()
    logger.info("player_match_stats 表已建立")


def build_name_to_pid(conn: sqlite3.Connection) -> dict[str, int]:
    """建立 {正規化姓名: player_id} 的查找表。"""
    rows = conn.execute("SELECT player_id, name FROM players").fetchall()
    return {normalize_name(name): pid for pid, name in rows}


def fetch_player_list(team_id: int) -> list[dict]:
    """
    從外部系統取得某隊的球員清單。
    回傳 [{'ext_player_id': int, 'name': str}, ...]
    """
    url = f"{EXT_BASE}/_handler/PlayerList.ashx"
    r = requests.get(
        url, params={"CupID": CUP_ID, "TeamID": team_id},
        headers=HEADERS, timeout=15,
    )
    soup = BeautifulSoup(r.text, "html.parser")
    players = []
    for opt in soup.find_all("option"):
        text = opt.get_text(strip=True)  # e.g. "No.2-黃宇晨"
        ext_id = opt.get("value")
        name = text.split("-", 1)[1] if "-" in text else text
        players.append({"ext_player_id": int(ext_id), "name": name})
    return players


def fetch_player_stats(team_id: int, ext_player_id: int) -> list[dict]:
    """
    抓取單一球員的逐場數據表，回傳字典列表。
    跳過表頭行與最後的「累計」行。
    """
    url = f"{EXT_BASE}/_handler/Player.ashx"
    r = requests.get(
        url,
        params={
            "CupID": CUP_ID,
            "PlayerID": ext_player_id,
            "TeamID": team_id,
        },
        headers=HEADERS, timeout=15,
    )
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    rows = table.find_all("tr")
    # row[0]: 大分類表頭, row[1]: 子分類表頭, row[2:-1]: 數據, row[-1]: 累計
    records = []
    for row in rows[2:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
        # 跳過累計行
        if not cells or cells[0] == "累計":
            continue
        # 欄位映射：共 15 欄
        # [0]場次日期 [1]對戰隊伍 [2]局數
        # [3]攻擊總 [4]攻擊得 [5]攔網得
        # [6]發球總 [7]發球得 [8]接發總 [9]接發好
        # [10]防守總 [11]防守好 [12]舉球總 [13]舉球好
        # [14]總得分
        if len(cells) < 15:
            continue

        record = {
            "match_date": parse_match_date(cells[0]),
            "opponent": cells[1] or None,
            "sets_played": safe_int(cells[2]),
            "attack_total": safe_int(cells[3]),
            "attack_points": safe_int(cells[4]),
            "block_points": safe_int(cells[5]),
            "serve_total": safe_int(cells[6]),
            "serve_points": safe_int(cells[7]),
            "receive_total": safe_int(cells[8]),
            "receive_excellent": safe_int(cells[9]),
            "dig_total": safe_int(cells[10]),
            "dig_excellent": safe_int(cells[11]),
            "set_total": safe_int(cells[12]),
            "set_excellent": safe_int(cells[13]),
            "total_points": safe_int(cells[14]),
        }
        records.append(record)

    return records


def get_existing_dates(conn: sqlite3.Connection, player_id: int) -> set[str]:
    """取得某球員已存在的比賽日期集合（用於增量比對）。"""
    rows = conn.execute(
        "SELECT DISTINCT match_date FROM player_match_stats WHERE player_id = ?",
        (player_id,),
    ).fetchall()
    return {r[0] for r in rows}


def main(incremental: bool = False):
    """
    主流程：抓取所有球員逐場數據並寫入 DB。

    Parameters
    ----------
    incremental : bool
        True = 增量模式，只新增尚未存在的比賽紀錄（不清除既有資料）。
        False = 全量模式，DROP + CREATE 事實表後重新抓取。
    """
    conn = get_connection()

    if not incremental:
        init_stats_table(conn)
    else:
        logger.info("增量模式：保留既有資料，僅新增缺少的比賽紀錄")

    name_map = build_name_to_pid(conn)

    total_inserted = 0
    total_skipped = 0
    total_new_players = 0

    for ext_team_id in range(1, 10):
        players = fetch_player_list(ext_team_id)
        logger.info(
            "TeamID=%d: %d 位球員", ext_team_id, len(players)
        )

        for p in players:
            ext_pid = p["ext_player_id"]
            name = p["name"]
            norm_name = normalize_name(name)

            # 正規化比對
            player_id = name_map.get(norm_name)

            # Late Arriving Dimension：查無此人則動態新增
            if player_id is None:
                db_team_id, gender = EXT_TEAM_MAP[ext_team_id]
                logger.info(
                    "[動態新增] 發現新球員: %s，自動寫入 players 表。",
                    name,
                )
                cursor = conn.execute(
                    "INSERT INTO players (name, team_id, gender) VALUES (?, ?, ?)",
                    (name, db_team_id, gender),
                )
                conn.commit()
                player_id = cursor.lastrowid
                name_map[norm_name] = player_id
                total_new_players += 1

            # 抓取逐場數據
            try:
                records = fetch_player_stats(ext_team_id, ext_pid)
            except Exception as e:
                logger.error(
                    "抓取球員 [%s] 數據失敗: %s", name, e
                )
                continue

            if not records:
                continue

            # 增量模式：過濾已存在的日期
            if incremental:
                existing = get_existing_dates(conn, player_id)
                new_records = [r for r in records if r["match_date"] not in existing]
                total_skipped += len(records) - len(new_records)
                records = new_records
                if not records:
                    continue

            # 批次寫入
            conn.executemany(
                """INSERT INTO player_match_stats
                   (player_id, match_date, opponent, sets_played,
                    attack_total, attack_points, block_points,
                    serve_total, serve_points,
                    receive_total, receive_excellent,
                    dig_total, dig_excellent,
                    set_total, set_excellent, total_points)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        player_id,
                        r["match_date"], r["opponent"], r["sets_played"],
                        r["attack_total"], r["attack_points"], r["block_points"],
                        r["serve_total"], r["serve_points"],
                        r["receive_total"], r["receive_excellent"],
                        r["dig_total"], r["dig_excellent"],
                        r["set_total"], r["set_excellent"], r["total_points"],
                    )
                    for r in records
                ],
            )
            total_inserted += len(records)

            time.sleep(0.5)

        conn.commit()

    # 統計與驗證
    total_rows = conn.execute(
        "SELECT COUNT(*) FROM player_match_stats"
    ).fetchone()[0]

    total_players = conn.execute(
        "SELECT COUNT(*) FROM players"
    ).fetchone()[0]

    mode_label = "增量" if incremental else "全量"
    print(f"\n===== {mode_label}寫入完成 =====")
    print(f"player_match_stats 總筆數：{total_rows}")
    print(f"本次新增：{total_inserted} 筆")
    if incremental:
        print(f"跳過（已存在）：{total_skipped} 筆")
    print(f"動態新增球員數：{total_new_players}")
    print(f"players 表總人數：{total_players}")

    print(f"\n===== 前 3 筆資料 =====")
    df = pd.read_sql_query(
        """SELECT s.stat_id, p.name, s.match_date, s.opponent,
                  s.sets_played, s.attack_points, s.block_points,
                  s.serve_points, s.total_points
           FROM player_match_stats s
           JOIN players p ON s.player_id = p.player_id
           LIMIT 3""",
        conn,
    )
    print(df.to_string(index=False))

    conn.close()
    logger.info("事實表載入完成：%s", DB_PATH)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TVL 球員逐場數據爬蟲")
    parser.add_argument(
        "--incremental", "-i", action="store_true",
        help="增量模式：僅新增尚未存在的比賽紀錄，不清除既有資料",
    )
    args = parser.parse_args()
    main(incremental=args.incremental)
