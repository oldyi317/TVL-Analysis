"""
TVL 官網比賽結果爬蟲模組
從 tvl.ctvba.org.tw 的 /game/ (男子組) 與 /wgame/ (女子組) 頁面
抓取各局比分、比賽資訊，並寫入 matches 表。
"""

import re
import time
import sqlite3

import requests
from bs4 import BeautifulSoup
from pathlib import Path

try:
    from src.utils.db_config import DB_PATH, get_connection
    from src.utils.logger import get_logger
    from src.utils.constants import EXT_HEADERS as HEADERS
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

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

logger = get_logger(__name__)

BASE_URL = "https://tvl.ctvba.org.tw"

# 官網隊名 → DB 簡寫對照（官網隊名可能有差異）
TEAM_ALIAS = {
    "臺北鯨華": "臺北鯨華",
    "新北中纖": "新北中纖",
    "高雄台電": "高雄台電",
    "義力營造": "義力營造",
    "屏東台電": "屏東台電",
    "雲林美津濃": "雲林美津濃",
    "臺北國北獅": "臺北國北獅",
    "桃園臺灣產險": "桃園臺產",
    "臺中獅子王": "獅子王",
}


def normalize_team(raw: str) -> str:
    """將官網隊名轉為 DB 簡寫。"""
    return TEAM_ALIAS.get(raw, raw)


def init_matches_table(conn: sqlite3.Connection) -> None:
    """建立 matches 比賽結果表（若不存在）。"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id         INTEGER NOT NULL,
            gender          TEXT NOT NULL CHECK (gender IN ('M', 'F')),
            match_date      DATE NOT NULL,
            venue           TEXT,
            round_name      TEXT,
            game_label      TEXT,
            is_golden_set   INTEGER NOT NULL DEFAULT 0,
            home_team       TEXT NOT NULL,
            away_team       TEXT NOT NULL,
            home_set1       INTEGER,
            home_set2       INTEGER,
            home_set3       INTEGER,
            home_set4       INTEGER,
            home_set5       INTEGER,
            home_total      INTEGER,
            away_set1       INTEGER,
            away_set2       INTEGER,
            away_set3       INTEGER,
            away_set4       INTEGER,
            away_set5       INTEGER,
            away_total      INTEGER,
            home_sets_won   INTEGER,
            away_sets_won   INTEGER,
            UNIQUE (game_id, gender)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date)"
    )
    conn.commit()
    logger.info("matches 表已確認存在")


def _safe_int(val: str) -> int | None:
    """安全轉換整數，空字串或失敗回傳 None。"""
    try:
        return int(val) if val and val.strip() else None
    except (ValueError, TypeError):
        return None


def scrape_match_page(prefix: str, game_id: int) -> dict | None:
    """
    抓取單場比賽頁面，回傳結構化 dict。

    Parameters
    ----------
    prefix : 'game' (男子組) 或 'wgame' (女子組)
    game_id : 官網頁面 ID

    Returns
    -------
    dict or None (頁面不存在或無資料)
    """
    gender = "M" if prefix == "game" else "F"
    url = f"{BASE_URL}/{prefix}/{game_id}"

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        r.encoding = "utf-8"
    except requests.RequestException as e:
        logger.error("無法取得 %s: %s", url, e)
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # ── 比分表 ──────────────────────────────────────────────
    score_table = soup.find("table", class_="match_table")
    if not score_table:
        return None

    rows = score_table.find_all("tr")
    if len(rows) < 3:
        return None

    # Row 1 (index 1): home team, Row 2 (index 2): away team
    home_cells = [td.get_text(strip=True) for td in rows[1].find_all("td")]
    away_cells = [td.get_text(strip=True) for td in rows[2].find_all("td")]

    if not home_cells or not home_cells[0]:
        return None

    home_team = normalize_team(home_cells[0])
    away_team = normalize_team(away_cells[0])

    # 各局分數: cells[1:6], 總分: cells[6]
    home_sets = [_safe_int(home_cells[i]) if i < len(home_cells) else None for i in range(1, 6)]
    away_sets = [_safe_int(away_cells[i]) if i < len(away_cells) else None for i in range(1, 6)]
    home_total = _safe_int(home_cells[6]) if len(home_cells) > 6 else None
    away_total = _safe_int(away_cells[6]) if len(away_cells) > 6 else None

    # 計算勝局數
    home_sets_won = sum(
        1 for h, a in zip(home_sets, away_sets)
        if h is not None and a is not None and h > a
    )
    away_sets_won = sum(
        1 for h, a in zip(home_sets, away_sets)
        if h is not None and a is not None and a > h
    )

    # ── Game header ─────────────────────────────────────────
    gh = soup.find("div", class_="game_header")
    gh_text = gh.get_text(" | ", strip=True) if gh else ""

    # 日期: 2026-03-22 16:30:00
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", gh_text)
    match_date = date_m.group(1) if date_m else None

    # 場館
    venue = None
    venue_m = re.search(r"\d{2}:\d{2}:\d{2}\s*\|?\s*(.+?)\s*\|", gh_text)
    if venue_m:
        venue = venue_m.group(1).strip()

    # 賽事階段 (例行賽/挑戰賽/總決賽)
    round_name = None
    round_m = re.search(r"(例行賽|挑戰賽|總決賽|季後賽|明星賽)\s*Week\s*\d+", gh_text)
    if round_m:
        round_name = round_m.group(0)

    # Game label (Game 115-1 (黃金決勝局))
    game_label = None
    label_m = re.search(r"(Game\s*\S+(?:\s*\(.*?\))?)", gh_text)
    if label_m:
        game_label = label_m.group(1).strip()

    # 黃金決勝局判定
    is_golden = 1 if "黃金決勝局" in gh_text else 0

    if not match_date:
        logger.warning("[%s/%d] 無法解析日期，跳過", prefix, game_id)
        return None

    return {
        "game_id": game_id,
        "gender": gender,
        "match_date": match_date,
        "venue": venue,
        "round_name": round_name,
        "game_label": game_label,
        "is_golden_set": is_golden,
        "home_team": home_team,
        "away_team": away_team,
        "home_set1": home_sets[0],
        "home_set2": home_sets[1],
        "home_set3": home_sets[2],
        "home_set4": home_sets[3],
        "home_set5": home_sets[4],
        "home_total": home_total,
        "away_set1": away_sets[0],
        "away_set2": away_sets[1],
        "away_set3": away_sets[2],
        "away_set4": away_sets[3],
        "away_set5": away_sets[4],
        "away_total": away_total,
        "home_sets_won": home_sets_won,
        "away_sets_won": away_sets_won,
    }


def upsert_match(conn: sqlite3.Connection, match: dict) -> bool:
    """寫入或更新單場比賽紀錄，回傳是否為新增。"""
    existing = conn.execute(
        "SELECT match_id FROM matches WHERE game_id = ? AND gender = ?",
        (match["game_id"], match["gender"]),
    ).fetchone()

    if existing:
        conn.execute("""
            UPDATE matches SET
                match_date=?, venue=?, round_name=?, game_label=?,
                is_golden_set=?, home_team=?, away_team=?,
                home_set1=?, home_set2=?, home_set3=?, home_set4=?, home_set5=?,
                home_total=?, away_set1=?, away_set2=?, away_set3=?, away_set4=?,
                away_set5=?, away_total=?, home_sets_won=?, away_sets_won=?
            WHERE game_id=? AND gender=?
        """, (
            match["match_date"], match["venue"], match["round_name"],
            match["game_label"], match["is_golden_set"],
            match["home_team"], match["away_team"],
            match["home_set1"], match["home_set2"], match["home_set3"],
            match["home_set4"], match["home_set5"], match["home_total"],
            match["away_set1"], match["away_set2"], match["away_set3"],
            match["away_set4"], match["away_set5"], match["away_total"],
            match["home_sets_won"], match["away_sets_won"],
            match["game_id"], match["gender"],
        ))
        return False
    else:
        conn.execute("""
            INSERT INTO matches (
                game_id, gender, match_date, venue, round_name, game_label,
                is_golden_set, home_team, away_team,
                home_set1, home_set2, home_set3, home_set4, home_set5, home_total,
                away_set1, away_set2, away_set3, away_set4, away_set5, away_total,
                home_sets_won, away_sets_won
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            match["game_id"], match["gender"], match["match_date"],
            match["venue"], match["round_name"], match["game_label"],
            match["is_golden_set"], match["home_team"], match["away_team"],
            match["home_set1"], match["home_set2"], match["home_set3"],
            match["home_set4"], match["home_set5"], match["home_total"],
            match["away_set1"], match["away_set2"], match["away_set3"],
            match["away_set4"], match["away_set5"], match["away_total"],
            match["home_sets_won"], match["away_sets_won"],
        ))
        return True


def scrape_all_matches(
    prefixes: list[str] | None = None,
    id_range: range | None = None,
    delay: float = 0.5,
) -> dict:
    """
    批次抓取官網比賽結果並寫入 DB。

    Parameters
    ----------
    prefixes : ['game', 'wgame']
    id_range : 要掃描的 game_id 範圍
    delay : 請求間隔秒數
    """
    if prefixes is None:
        prefixes = ["game", "wgame"]
    if id_range is None:
        id_range = range(220, 400)

    conn = get_connection()
    init_matches_table(conn)

    stats = {"inserted": 0, "updated": 0, "skipped": 0, "golden_sets": 0}

    for prefix in prefixes:
        consecutive_empty = 0
        for game_id in id_range:
            match = scrape_match_page(prefix, game_id)
            if match is None:
                consecutive_empty += 1
                # 連續 20 個空頁面就停止該組別
                if consecutive_empty >= 20:
                    logger.info(
                        "[%s] 連續 %d 個空頁面，停止掃描",
                        prefix, consecutive_empty,
                    )
                    break
                continue

            consecutive_empty = 0
            is_new = upsert_match(conn, match)

            if is_new:
                stats["inserted"] += 1
            else:
                stats["updated"] += 1

            if match["is_golden_set"]:
                stats["golden_sets"] += 1

            logger.info(
                "[%s/%d] %s %s vs %s %s%s",
                prefix, game_id, match["match_date"],
                match["home_team"], match["away_team"],
                "NEW" if is_new else "UPD",
                " ★Golden Set" if match["is_golden_set"] else "",
            )

            time.sleep(delay)

        conn.commit()

    # 統計
    total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    golden = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE is_golden_set = 1"
    ).fetchone()[0]
    conn.close()

    stats["total"] = total
    stats["total_golden"] = golden
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TVL 官網比賽結果爬蟲")
    parser.add_argument(
        "--range-start", type=int, default=220,
        help="起始 game_id (預設 220)",
    )
    parser.add_argument(
        "--range-end", type=int, default=400,
        help="結束 game_id (預設 400)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="請求間隔秒數 (預設 0.5)",
    )
    args = parser.parse_args()

    stats = scrape_all_matches(
        id_range=range(args.range_start, args.range_end),
        delay=args.delay,
    )

    print(f"\n===== 比賽結果爬取完成 =====")
    print(f"matches 表總筆數：{stats['total']}")
    print(f"本次新增：{stats['inserted']} 場")
    print(f"本次更新：{stats['updated']} 場")
    print(f"黃金決勝局：{stats['total_golden']} 場")


if __name__ == "__main__":
    main()
