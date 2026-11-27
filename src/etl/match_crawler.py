"""
TVL 官網比賽結果爬蟲模組
從 tvl.ctvba.org.tw 的 /game/ (男子組) 與 /wgame/ (女子組) 頁面
抓取各局比分、比賽資訊，並寫入 matches 表。
"""

import argparse
from contextlib import contextmanager
from datetime import date
import re
import time
import sqlite3

import requests
from bs4 import BeautifulSoup
from pathlib import Path

from src.utils.db_config import get_connection
from src.utils.logger import get_logger
from src.utils.constants import (
    EXT_HEADERS as HEADERS,
    GAME_PAGE_PREFIXES,
    ROUND_STAGE_ALIAS,
    TVL_BASE_URL as BASE_URL,
)
from src.utils.teams import resolve_team_id

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"

logger = get_logger(__name__)

ROUND_NAME_RE = re.compile(r"(例行賽|挑戰賽|總決賽|總冠軍賽|季後賽|明星賽)\s*Week\s*(\d+)")  # 官網原字


def parse_round_name(text: str) -> str | None:
    m = ROUND_NAME_RE.search(text)
    if not m:
        return None
    stage = ROUND_STAGE_ALIAS.get(m.group(1), m.group(1))
    return f"{stage} Week {m.group(2)}"


def init_matches_table(conn: sqlite3.Connection) -> None:
    """確保 matches 表存在（讀 schema.sql，冪等）。"""
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    _assert_matches_migrated(conn)
    logger.info("matches 表已確認存在（DDL 來源：schema.sql）")


def _assert_matches_migrated(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(matches)")}
    if "cup_id" not in cols:
        raise RuntimeError(
            "matches 表 schema 過舊（缺 cup_id），請改用 repo 內已遷移的 data/db/tvl_database.db（git checkout 還原）。"
        )
    if "home_team_id" not in cols:
        raise RuntimeError(
            "matches 表 schema 過舊（缺 home_team_id），請改用 repo 內已遷移的 data/db/tvl_database.db（git checkout 還原）。"
        )


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
    dict or None (頁面不存在、無資料，或連線層錯誤)

    Raises
    ------
    requests.HTTPError : 官網有回應但狀態碼是 4xx／5xx
    """
    gender = "M" if prefix == GAME_PAGE_PREFIXES["M"] else "F"
    url = f"{BASE_URL}/{prefix}/{game_id}"

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
    except requests.RequestException as e:
        logger.error("無法取得 %s: %s", url, e)
        return None
    # 官網對不存在的場次也回 200（2026-11-19 實測），4xx／5xx 就是被擋或出錯（如 Cloudflare 的 403 封鎖頁）：
    # 不當成空頁，往上拋讓整輪失敗
    r.raise_for_status()
    r.encoding = "utf-8"

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

    home_team = home_cells[0]
    away_team = away_cells[0]

    # 各局分數: cells[1:6], 總分: cells[6]
    home_sets = [_safe_int(home_cells[i]) if i < len(home_cells) else None for i in range(1, 6)]
    away_sets = [_safe_int(away_cells[i]) if i < len(away_cells) else None for i in range(1, 6)]
    home_total = _safe_int(home_cells[6]) if len(home_cells) > 6 else None
    away_total = _safe_int(away_cells[6]) if len(away_cells) > 6 else None

    # 計算勝局數；各局全空（未開打）寫 None 而非 0
    if all(s is None for s in home_sets + away_sets):
        home_sets_won = away_sets_won = None
    else:
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

    time_m = re.search(r"(\d{2}:\d{2}):\d{2}", gh_text)
    start_time = time_m.group(1) if time_m else None

    # 場館
    venue = None
    venue_m = re.search(r"\d{2}:\d{2}:\d{2}\s*\|?\s*(.+?)\s*\|", gh_text)
    if venue_m:
        venue = venue_m.group(1).strip()

    # 賽事階段 (例行賽/挑戰賽/總決賽)
    round_name = parse_round_name(gh_text)

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
        "start_time": start_time,
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


def resolve_cup_id(conn: sqlite3.Connection, match_date: str) -> int:
    row = conn.execute(
        "SELECT cup_id FROM seasons WHERE ? BETWEEN start_date AND end_date",
        (match_date,),
    ).fetchone()
    if row is None:
        raise RuntimeError(
            f"找不到涵蓋 {match_date} 的賽季，請在 sql/schema.sql 的 seasons 種子"
            "與 DB 的 seasons 表新增該賽季。"
        )
    return row[0]


def upsert_match(conn: sqlite3.Connection, match: dict) -> bool:
    """寫入或更新單場比賽紀錄，回傳是否為新增。round_name／game_label 單場頁解析不到時保留既有值（ADR 0008）。"""
    existing = conn.execute(
        "SELECT match_id FROM matches WHERE game_id = ? AND gender = ?",
        (match["game_id"], match["gender"]),
    ).fetchone()
    cup_id = resolve_cup_id(conn, match["match_date"])
    ctx = f"{match['gender']}/{match['game_id']}"
    home_team_id = resolve_team_id(conn, match["home_team"], match["gender"], f"{ctx} home")
    away_team_id = resolve_team_id(conn, match["away_team"], match["gender"], f"{ctx} away")

    if existing:
        conn.execute("""
            UPDATE matches SET
                cup_id=?, match_date=?, start_time=COALESCE(start_time, ?),
                venue=COALESCE(venue, ?), round_name=COALESCE(?, round_name),
                game_label=COALESCE(?, game_label),
                is_golden_set=?, home_team=?, away_team=?, home_team_id=?, away_team_id=?,
                home_set1=?, home_set2=?, home_set3=?, home_set4=?, home_set5=?,
                home_total=?, away_set1=?, away_set2=?, away_set3=?, away_set4=?,
                away_set5=?, away_total=?, home_sets_won=?, away_sets_won=?
            WHERE game_id=? AND gender=?
        """, (
            cup_id, match["match_date"], match.get("start_time"), match["venue"],
            match["round_name"], match["game_label"], match["is_golden_set"],
            match["home_team"], match["away_team"], home_team_id, away_team_id,
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
                game_id, gender, cup_id, match_date, start_time, venue, round_name, game_label,
                is_golden_set, home_team, away_team, home_team_id, away_team_id,
                home_set1, home_set2, home_set3, home_set4, home_set5, home_total,
                away_set1, away_set2, away_set3, away_set4, away_set5, away_total,
                home_sets_won, away_sets_won
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            match["game_id"], match["gender"], cup_id, match["match_date"],
            match.get("start_time"), match["venue"], match["round_name"], match["game_label"],
            match["is_golden_set"], match["home_team"], match["away_team"],
            home_team_id, away_team_id,
            match["home_set1"], match["home_set2"], match["home_set3"],
            match["home_set4"], match["home_set5"], match["home_total"],
            match["away_set1"], match["away_set2"], match["away_set3"],
            match["away_set4"], match["away_set5"], match["away_total"],
            match["home_sets_won"], match["away_sets_won"],
        ))
        return True


def _record_match(conn: sqlite3.Connection, prefix: str, game_id: int, match: dict, stats: dict) -> None:
    is_new = upsert_match(conn, match)
    stats["inserted" if is_new else "updated"] += 1
    if match["is_golden_set"]:
        stats["golden_sets"] += 1
    logger.info(
        "[%s/%d] %s %s vs %s %s%s",
        prefix, game_id, match["match_date"],
        match["home_team"], match["away_team"],
        "NEW" if is_new else "UPD",
        "（黃金決勝局）" if match["is_golden_set"] else "",
    )


@contextmanager
def _crawl_connection():
    conn = get_connection()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _finalize_stats(conn: sqlite3.Connection, stats: dict) -> dict:
    # 全部抓完才 commit：中途被官網擋下時，這一輪什麼都不寫入
    conn.commit()
    stats["total"] = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    stats["total_golden"] = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE is_golden_set = 1"
    ).fetchone()[0]
    return stats


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
        prefixes = list(GAME_PAGE_PREFIXES.values())
    if id_range is None:
        id_range = range(220, 400)

    with _crawl_connection() as conn:
        init_matches_table(conn)
        stats = {"requested": 0, "inserted": 0, "updated": 0, "skipped": 0, "golden_sets": 0}

        for prefix in prefixes:
            consecutive_empty = 0
            for game_id in id_range:
                stats["requested"] += 1
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
                _record_match(conn, prefix, game_id, match, stats)
                time.sleep(delay)

        return _finalize_stats(conn, stats)


def scrape_matches_by_date(target_date: str, delay: float = 0.5) -> dict:
    """只對資料庫中指定日期、本季的賽事抓官網賽事頁並寫回比分。"""
    with _crawl_connection() as conn:
        init_matches_table(conn)
        targets = conn.execute(
            """
            SELECT game_id, gender FROM matches
            WHERE match_date = ? AND cup_id = (SELECT MAX(cup_id) FROM seasons)
            ORDER BY CASE gender WHEN 'M' THEN 0 ELSE 1 END, game_id
            """,
            (target_date,),
        ).fetchall()

        stats = {
            "requested": len(targets),
            "inserted": 0,
            "updated": 0,
            "skipped": 0,
            "golden_sets": 0,
        }
        logger.info("[%s] 本季賽事 %d 場", target_date, len(targets))

        for index, (game_id, gender) in enumerate(targets):
            if index:
                time.sleep(delay)
            prefix = GAME_PAGE_PREFIXES[gender]
            match = scrape_match_page(prefix, game_id)
            if match is None:
                stats["skipped"] += 1
                logger.warning("[%s/%d] 未取得比分，略過", prefix, game_id)
                continue
            _record_match(conn, prefix, game_id, match, stats)

        return _finalize_stats(conn, stats)


def _iso_date(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise argparse.ArgumentTypeError(f"日期格式須為 YYYY-MM-DD：{value!r}")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"無效的日期：{value!r}") from exc
    return value


def main():
    parser = argparse.ArgumentParser(description="TVL 官網比賽結果爬蟲")
    parser.add_argument(
        "--date", type=_iso_date, default=None, metavar="YYYY-MM-DD",
        help="只抓指定日期的本季賽事（不重掃區間）",
    )
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

    if args.date is not None:
        stats = scrape_matches_by_date(args.date, delay=args.delay)
    else:
        stats = scrape_all_matches(
            id_range=range(args.range_start, args.range_end),
            delay=args.delay,
        )

    print("\n===== 比賽結果爬取完成 =====")
    print(f"本次請求：{stats['requested']} 頁")
    print(f"matches 表總筆數：{stats['total']}")
    print(f"本次新增：{stats['inserted']} 場")
    print(f"本次更新：{stats['updated']} 場")
    print(f"黃金決勝局：{stats['total_golden']} 場")


if __name__ == "__main__":
    main()
