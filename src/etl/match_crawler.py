"""
TVL 官網比賽結果爬蟲模組
從 tvl.ctvba.org.tw 的 /game/ (男子組) 與 /wgame/ (女子組) 頁面
抓取各局比分、比賽資訊，並 upsert 至 matches 表（依 season 隔離，冪等）。
"""

import re
import time

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.etl.db_loader import init_db
from src.utils.db_config import get_engine
from src.utils.logger import get_logger
from src.utils.constants import EXT_HEADERS as HEADERS, SEASON, TEAM_ALIAS

logger = get_logger(__name__)

BASE_URL = "https://tvl.ctvba.org.tw"


def normalize_team(raw: str) -> str:
    """將官網隊名轉為 DB 簡寫。"""
    return TEAM_ALIAS.get(raw, raw)


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

    score_table = soup.find("table", class_="match_table")
    if not score_table:
        return None

    rows = score_table.find_all("tr")
    if len(rows) < 3:
        return None

    home_cells = [td.get_text(strip=True) for td in rows[1].find_all("td")]
    away_cells = [td.get_text(strip=True) for td in rows[2].find_all("td")]

    if not home_cells or not home_cells[0]:
        return None

    home_team = normalize_team(home_cells[0])
    away_team = normalize_team(away_cells[0])

    home_sets = [_safe_int(home_cells[i]) if i < len(home_cells) else None for i in range(1, 6)]
    away_sets = [_safe_int(away_cells[i]) if i < len(away_cells) else None for i in range(1, 6)]
    home_total = _safe_int(home_cells[6]) if len(home_cells) > 6 else None
    away_total = _safe_int(away_cells[6]) if len(away_cells) > 6 else None

    home_sets_won = sum(
        1 for h, a in zip(home_sets, away_sets)
        if h is not None and a is not None and h > a
    )
    away_sets_won = sum(
        1 for h, a in zip(home_sets, away_sets)
        if h is not None and a is not None and a > h
    )

    gh = soup.find("div", class_="game_header")
    gh_text = gh.get_text(" | ", strip=True) if gh else ""

    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", gh_text)
    match_date = date_m.group(1) if date_m else None

    venue = None
    venue_m = re.search(r"\d{2}:\d{2}:\d{2}\s*\|?\s*(.+?)\s*\|", gh_text)
    if venue_m:
        venue = venue_m.group(1).strip()

    round_name = None
    round_m = re.search(r"(例行賽|挑戰賽|總決賽|季後賽|明星賽)\s*Week\s*\d+", gh_text)
    if round_m:
        round_name = round_m.group(0)

    game_label = None
    label_m = re.search(r"(Game\s*\S+(?:\s*\(.*?\))?)", gh_text)
    if label_m:
        game_label = label_m.group(1).strip()

    is_golden = 1 if "黃金決勝局" in gh_text else 0

    if not match_date:
        logger.warning("[%s/%d] 無法解析日期，跳過", prefix, game_id)
        return None

    return {
        "game_id": game_id,
        "gender": gender,
        "season": SEASON,
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


def upsert_match(engine: Engine, match: dict) -> None:
    """Upsert 單場比賽紀錄（唯一鍵：game_id + gender + season，冪等）。"""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO matches (
                game_id, gender, season, match_date, venue, round_name, game_label,
                is_golden_set, home_team, away_team,
                home_set1, home_set2, home_set3, home_set4, home_set5, home_total,
                away_set1, away_set2, away_set3, away_set4, away_set5, away_total,
                home_sets_won, away_sets_won
            ) VALUES (
                :game_id, :gender, :season, :match_date, :venue, :round_name, :game_label,
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
        """), match)


def scrape_all_matches(
    prefixes: list[str] | None = None,
    id_range: range | None = None,
    delay: float = 0.5,
) -> dict:
    """
    批次抓取官網比賽結果並 upsert 至 DB。

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

    engine = get_engine()
    init_db(engine)

    stats = {"upserted": 0, "golden_sets": 0}

    for prefix in prefixes:
        consecutive_empty = 0
        for game_id in id_range:
            match = scrape_match_page(prefix, game_id)
            if match is None:
                consecutive_empty += 1
                if consecutive_empty >= 20:
                    logger.info("[%s] 連續 %d 個空頁面，停止掃描", prefix, consecutive_empty)
                    break
                continue

            consecutive_empty = 0
            upsert_match(engine, match)
            stats["upserted"] += 1

            if match["is_golden_set"]:
                stats["golden_sets"] += 1

            logger.info(
                "[%s/%d] %s %s vs %s%s",
                prefix, game_id, match["match_date"],
                match["home_team"], match["away_team"],
                " ★Golden Set" if match["is_golden_set"] else "",
            )

            time.sleep(delay)

    with engine.begin() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM matches")).scalar_one()
        golden = conn.execute(
            text("SELECT COUNT(*) FROM matches WHERE is_golden_set = 1")
        ).scalar_one()

    stats["total"] = total
    stats["total_golden"] = golden
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TVL 官網比賽結果爬蟲")
    parser.add_argument("--range-start", type=int, default=220, help="起始 game_id (預設 220)")
    parser.add_argument("--range-end", type=int, default=400, help="結束 game_id (預設 400)")
    parser.add_argument("--delay", type=float, default=0.5, help="請求間隔秒數 (預設 0.5)")
    args = parser.parse_args()

    stats = scrape_all_matches(
        id_range=range(args.range_start, args.range_end),
        delay=args.delay,
    )

    print(f"\n===== 比賽結果爬取完成 =====")
    print(f"matches 表總筆數：{stats['total']}")
    print(f"本次 upsert：{stats['upserted']} 場")
    print(f"黃金決勝局：{stats['total_golden']} 場")


if __name__ == "__main__":
    main()
