"""
TVL 官網賽事列表爬蟲
從 tvl.ctvba.org.tw 的 /schedule（男子組）與 /wschedule（女子組）解析每場的
game_id、日期、時間、場館、兩隊與 game_label，upsert 進 matches（鍵 game_id + gender）。
依 ADR 0003 的欄位分工：本爬蟲永不寫各局比分與 sets_won。
round_name 在 2026-10-15 抓到的列表頁上全部空白，故只在列內出現「例行賽 Week N」等字樣時才寫，
否則保留既有值（由 match_crawler 從單場頁補）。
解析一律防禦式：缺值寫 NULL、缺關鍵欄位跳過該列；時間/場館待定列尚無樣本（#28）。
"""
import re
import sqlite3

import requests
from bs4 import BeautifulSoup

from src.etl.match_crawler import _resolve_match_team_id, normalize_team, parse_round_name, resolve_cup_id
from src.utils.constants import EXT_HEADERS as HEADERS, SCHEDULE_PAGES, TVL_BASE_URL
from src.utils.db_config import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

_GAME_LINK = re.compile(r"/w?game/(\d+)$")
_YEAR_LINK = re.compile(r"^/w?schedule/\d+$")
MAX_YEAR_HOPS = 3
_MONTH_CLASS = re.compile(r"^d-(\d{4})-\d{2}$")
_MONTH_DAY = re.compile(r"(\d{1,2})/(\d{1,2})")
_TIME = re.compile(r"(\d{1,2}):(\d{2})")


def _text(node) -> str:
    return node.get_text(" ", strip=True) if node else ""


def _parse_date(row) -> str | None:
    year = next((m.group(1) for c in row.get("class", []) if (m := _MONTH_CLASS.match(c))), None)
    md = _MONTH_DAY.search(_text(row.select_one("div.match_row_datetime h5.fs16")))
    if year is None or md is None:
        return None
    return f"{year}-{int(md.group(1)):02d}-{int(md.group(2)):02d}"


def _parse_time(row) -> str | None:
    tm = _TIME.search(_text(row.select_one("div.match_row_datetime h6")))
    return f"{int(tm.group(1)):02d}:{tm.group(2)}" if tm else None


def _team_name(block) -> str:
    return _text(block.select_one("h6.text-black span.PC_only") or block.select_one("h6.text-black"))


def _parse_row(row, gender: str) -> dict | None:
    link = row.find("a", href=_GAME_LINK)
    if link is None:
        logger.warning("[%s] 列表列缺 game 連結，跳過", gender)
        return None
    game_id = int(_GAME_LINK.search(link["href"]).group(1))

    match_date = _parse_date(row)
    if match_date is None:
        logger.warning("[%s/%s] 無法解析日期，跳過", gender, game_id)
        return None

    teams = row.select("div.col-lg-3")
    home, away = (_team_name(teams[0]), _team_name(teams[1])) if len(teams) >= 2 else ("", "")
    if not home or not away:
        logger.warning("[%s/%s] 無法解析兩隊隊名，跳過", gender, game_id)
        return None

    info = row.select_one("div.col-lg-4")
    game_label = (_text(info.select_one("h5.fs14")) or None) if info else None
    return {
        "game_id": game_id,
        "gender": gender,
        "match_date": match_date,
        "start_time": _parse_time(row),
        "venue": (_text(info.select_one("h5.fs12")) or None) if info else None,
        "round_name": parse_round_name(_text(row)),
        "game_label": game_label,
        "is_golden_set": 1 if game_label and "黃金決勝局" in game_label else 0,
        "home_team": normalize_team(home),
        "away_team": normalize_team(away),
    }


def parse_schedule_page(html: str, gender: str) -> tuple[list[dict], int]:
    """解析列表頁，回傳 (成功解析的場次, 頁面上的場次列數)。"""
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("div.match_row")
    matches = [m for row in rows if (m := _parse_row(row, gender)) is not None]
    return matches, len(rows)


def next_year_path(html: str) -> str | None:
    """列表頁預設顯示的年度可能落後新賽季，回傳「下一年度」連結路徑（無則 None）。"""
    link = BeautifulSoup(html, "html.parser").find("a", attrs={"aria-label": "Next"}, href=_YEAR_LINK)
    return link["href"] if link else None


def upsert_schedule_match(conn: sqlite3.Connection, match: dict, cup_id: int) -> bool:
    """寫入或更新賽事基本資料，回傳是否為新增。比分欄一律不動；頁面缺值的欄位保留既有值。"""
    home_team_id = _resolve_match_team_id(conn, match, "home")
    away_team_id = _resolve_match_team_id(conn, match, "away")
    existing = conn.execute(
        "SELECT match_id FROM matches WHERE game_id = ? AND gender = ?",
        (match["game_id"], match["gender"]),
    ).fetchone()
    golden = match["is_golden_set"] if match["game_label"] else None

    if existing:
        conn.execute("""
            UPDATE matches SET
                cup_id=?, match_date=?, start_time=COALESCE(?, start_time), venue=COALESCE(?, venue),
                round_name=COALESCE(?, round_name), game_label=COALESCE(?, game_label),
                is_golden_set=COALESCE(?, is_golden_set),
                home_team=?, away_team=?, home_team_id=?, away_team_id=?
            WHERE game_id=? AND gender=?
        """, (
            cup_id, match["match_date"], match["start_time"], match["venue"],
            match["round_name"], match["game_label"], golden,
            match["home_team"], match["away_team"], home_team_id, away_team_id,
            match["game_id"], match["gender"],
        ))
        return False

    conn.execute("""
        INSERT INTO matches (
            game_id, gender, cup_id, match_date, start_time, venue, round_name, game_label, is_golden_set,
            home_team, away_team, home_team_id, away_team_id
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        match["game_id"], match["gender"], cup_id, match["match_date"], match["start_time"],
        match["venue"], match["round_name"], match["game_label"], match["is_golden_set"],
        match["home_team"], match["away_team"], home_team_id, away_team_id,
    ))
    return True


def _fetch(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.error("抓取 %s 失敗：%s，整頁跳過", url, e)
        return None
    r.encoding = "utf-8"
    return r.text


def _ingest_page(conn: sqlite3.Connection, html: str, gender: str, path: str, stats: dict) -> None:
    matches, on_page = parse_schedule_page(html, gender)
    stats["parsed"] += len(matches)
    stats["on_page"] += on_page
    logger.info("[%s] %s 解析出 %d 場／頁面上 %d 場", gender, path, len(matches), on_page)

    try:
        cup_ids = {m["match_date"]: resolve_cup_id(conn, m["match_date"]) for m in matches}
    except RuntimeError as e:
        logger.error("[%s] %s %s 整頁跳過。", gender, path, e)
        return

    for m in matches:
        if upsert_schedule_match(conn, m, cup_ids[m["match_date"]]):
            stats["inserted"] += 1
        else:
            stats["updated"] += 1
    conn.commit()


def scrape_schedule_pages(conn: sqlite3.Connection, genders: tuple[str, ...] = ("M", "F")) -> dict[str, dict]:
    """逐性別抓列表頁（預設年度＋沿「下一年度」連結）並 upsert；任一頁失敗只記 log，不中止其他頁。"""
    results = {}
    for gender in genders:
        stats = {"parsed": 0, "on_page": 0, "inserted": 0, "updated": 0}
        results[gender] = stats
        path = SCHEDULE_PAGES[gender]
        for _ in range(MAX_YEAR_HOPS + 1):
            html = _fetch(TVL_BASE_URL + path)
            if html is None:
                break
            _ingest_page(conn, html, gender, path, stats)
            path = next_year_path(html)
            if path is None:
                break
        logger.info("[%s] 新增 %d 場、更新 %d 場", gender, stats["inserted"], stats["updated"])
    return results


def main():
    conn = get_connection()
    try:
        results = scrape_schedule_pages(conn)
    finally:
        conn.close()

    print("\n===== 賽事列表爬取完成 =====")
    for gender, s in results.items():
        print(f"[{gender}] 解析 {s['parsed']}/{s['on_page']} 場，新增 {s['inserted']}、更新 {s['updated']}")


if __name__ == "__main__":
    main()
