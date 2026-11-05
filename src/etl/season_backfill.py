"""
歷史賽季一次性回填（ADR 0010）：從外部統計系統 statistics.ctvba.org.tw 的 Match.ashx
一頁補齊賽事各局得分、週登錄與單場數據，並產出待人工確認清單。

只在隔離 DB（--db 必填）上跑，不進每日更新；cup 22 起拒絕執行（ADR 0009）。
用法：python -m src.etl.season_backfill --cup-id 19 --db <路徑> --review-out <清單.json>
"""

import argparse
import json
import random
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.etl.stats_crawler import (
    PlayerResolver, WeekLabelUnresolved, ext_team_map_for, filter_new_records, find_registration_id, insert_stat_rows,
    normalize_name, reject_external_path_for_official_cup, resolve_week_label, safe_int,
    upsert_roster_registration,
)
from src.utils.constants import (
    BACKFILL_SEASONS, EXT_BASE, EXT_HEADERS, EXT_TEAM_MAPS, EXT_TEAM_SEASON_NAMES,
    MATCH_POSITION_MAP, ROUND_STAGE_ALIAS,
)
from src.utils.db_config import get_connection
from src.utils.logger import get_logger
from src.utils.seasons import Season, get_season, resolve_date_in_season
from src.utils.teams import apply_team_seed

logger = get_logger(__name__)

STAT_FIELDS = [
    "attack_points", "attack_total", "block_points", "serve_points", "serve_total", "receive_excellent",
    "receive_total", "dig_excellent", "dig_total", "set_excellent", "set_total", "total_points",
]
STAGE_PATTERN = "(" + "|".join(["挑戰賽", "總決賽", *ROUND_STAGE_ALIAS, "季後賽"]) + ")"
BLOCK_MARKERS = ("challenge-platform", "cf-chl")


class BackfillBlocked(Exception):
    """外部系統回 403 或驗證挑戰：立即停止，不重試、不換標頭。"""


@dataclass
class TeamSide:
    name: str
    ext_team_id: int
    team_id: int
    sets: list[int | None]
    total: int | None
    sets_won: int | None
    players: list[dict] = field(default_factory=list)


@dataclass
class MatchPage:
    cup_id: int
    gender: str
    match_date: str
    start_time: str | None
    venue: str | None
    official_no: int | None
    week: int | None
    stage: str
    round_name: str | None
    game_label: str | None
    is_golden_set: int
    title: str
    home: TeamSide
    away: TeamSide


@dataclass
class BackfillResult:
    pages: int = 0
    new_matches: int = 0
    registrations: int = 0
    stats_inserted: int = 0
    skipped: int = 0
    unresolved: list[dict] = field(default_factory=list)
    last_match_id: int | None = None
    blocked_at: int | None = None
    blocked_reason: str | None = None


def ensure_backfill_season(conn: sqlite3.Connection, cup_id: int) -> Season:
    apply_team_seed(conn)
    name, start, end = BACKFILL_SEASONS[cup_id]
    conn.execute(
        "INSERT OR IGNORE INTO seasons (cup_id, name, start_date, end_date) VALUES (?, ?, ?, ?)",
        (cup_id, name, start, end),
    )
    for (team_id, gender), (team_name, short_name) in EXT_TEAM_SEASON_NAMES.get(cup_id, {}).items():
        conn.execute(
            "INSERT OR IGNORE INTO team_season_names (team_id, gender, cup_id, team_name, short_name) "
            "VALUES (?, ?, ?, ?, ?)",
            (team_id, gender, cup_id, team_name, short_name),
        )
    return get_season(conn, cup_id)


def fetch_match_page(cup_id: int, match_id: int) -> str | None:
    """抓 Match.ashx 一頁；無此賽事回 None；403 或驗證挑戰拋 BackfillBlocked。每次請求前隨機等 1–2 秒。"""
    time.sleep(random.uniform(1.0, 2.0))
    r = requests.get(
        f"{EXT_BASE}/_handler/Match.ashx",
        params={"CupID": cup_id, "MatchID": match_id, "SetNum": 0},
        headers=EXT_HEADERS, timeout=15,
    )
    if r.status_code == 403:
        raise BackfillBlocked(f"HTTP 403（MatchID={match_id}）")
    body = r.text or ""
    challenged = r.headers.get("cf-mitigated", "").lower() == "challenge" or r.status_code == 503
    if not challenged and r.status_code in (404, 500):
        return None  # 查無該場或外部系統應用錯誤頁；頁面雖帶 Cloudflare 注入腳本，不是阻擋式驗證
    if challenged or any(marker in body for marker in BLOCK_MARKERS):
        raise BackfillBlocked(f"驗證挑戰（MatchID={match_id}，HTTP {r.status_code}）")
    r.raise_for_status()
    r.encoding = "utf-8"
    text = r.text
    first = BeautifulSoup(text, "html.parser").find("h3")
    if first is None or "組" not in first.get_text(" ", strip=True):
        return None
    return text


def parse_match_ids(spec: str) -> list[int]:
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


def _parse_side(tr, cup_id: int, ext_map: dict) -> TeamSide:
    name_td = tr.find("td", class_="TeamName")
    name = name_td.get_text(strip=True)
    link = name_td.find("a", href=True)
    ext_m = re.search(r"TeamID=(\d+)", link["href"]) if link else None
    if ext_m is None:
        raise RuntimeError(f"cup_id={cup_id} 隊伍 {name} 找不到外部 TeamID")
    ext_team_id = int(ext_m.group(1))
    if ext_team_id not in ext_map:
        raise RuntimeError(f"cup_id={cup_id} 外部 TeamID={ext_team_id}（{name}）不在 EXT_TEAM_MAPS，請先補對照")
    raw_sets = [td.get_text(strip=True) for td in tr.find_all("td", class_="Score")]
    finals = [td.get_text(strip=True) for td in tr.find_all("td", class_="Final")]
    return TeamSide(
        name=name, ext_team_id=ext_team_id, team_id=ext_map[ext_team_id][0], sets=raw_sets,
        total=safe_int(finals[0]) if finals else None,
        sets_won=safe_int(finals[1]) if len(finals) > 1 else None,
    )


def _parse_players(soup, sides: dict[str, TeamSide], match_id_hint: str) -> None:
    for h3 in soup.find_all("h3")[1:]:
        text = h3.get_text(strip=True)
        if "：" not in text:
            continue
        team_name = text.split("：", 1)[0].strip()
        side = sides.get(team_name)
        if side is None:
            raise RuntimeError(f"{match_id_hint} 球員表隊名 {team_name} 對不上各局得分表的隊伍")
        table = h3.find_next("table")
        if table is None:
            continue
        for tr in table.find_all("tr")[2:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if not cells or cells[0] in ("全隊合計", "") or len(cells) < 15:
                continue
            position_raw = cells[2]
            if position_raw and position_raw not in MATCH_POSITION_MAP:
                logger.warning("未知位置用語：%s（球員 %s），記為 None", position_raw, cells[1])
            row = {
                "jersey_number": safe_int(cells[0]),
                "name": cells[1],
                "position": MATCH_POSITION_MAP.get(position_raw),
            }
            row.update({f: safe_int(v) for f, v in zip(STAT_FIELDS, cells[3:15])})
            side.players.append(row)


def parse_match_page(html: str, cup_id: int, season: Season) -> MatchPage | None:
    """純函式：把 Match.ashx 頁面解析成 MatchPage；明星賽或無法解析日期回 None。"""
    soup = BeautifulSoup(html, "html.parser")
    title_h3 = soup.find("h3")
    if title_h3 is None:
        return None
    title = title_h3.get_text(" ", strip=True)
    if "明星賽" in title:
        logger.info("明星賽不回填，略過：%s", title)
        return None
    if "男子組" in title:
        gender = "M"
    elif "女子組" in title:
        gender = "F"
    else:
        return None

    date_m = re.search(r"(\d{1,2})月(\d{1,2})日(?:\s+(\d{1,2}:\d{2}))?", title)
    match_date = resolve_date_in_season(int(date_m.group(1)), int(date_m.group(2)), season) if date_m else None
    if match_date is None:
        logger.warning("無法解析日期，略過：%s", title)
        return None
    week_m = re.search(r"第(\d+)週", title)
    week = int(week_m.group(1)) if week_m else None
    stage_m = re.search(STAGE_PATTERN, title)
    stage = ROUND_STAGE_ALIAS.get(stage_m.group(1), stage_m.group(1)) if stage_m else "例行賽"
    venue_m = re.search(r"(?:第\d+週|" + STAGE_PATTERN[1:-1] + r")\(([^)]*)\)", title)  # 季後賽標題沒有週次，場館接在階段後
    no_m = re.search(r"編號：[男女]?(\d+)", title)  # cup 18 起的舊季寫「編號：男1」
    official_no = int(no_m.group(1)) if no_m else None

    result = soup.select_one("div.MatchResult")
    rows = result.find_all("tr") if result else []
    rows = [tr for tr in rows if tr.find("td", class_="TeamName")]
    if len(rows) < 2:
        raise RuntimeError(f"cup_id={cup_id} 找不到兩隊各局得分表：{title}")
    ext_map = ext_team_map_for(cup_id)
    home, away = (_parse_side(tr, cup_id, ext_map) for tr in rows[:2])
    for side in (home, away):
        if ext_map[side.ext_team_id][1] != gender:
            raise RuntimeError(f"外部 TeamID={side.ext_team_id} 的組別與標題 {gender} 不一致：{title}")
    # 未打的局兩隊皆為 00（invisible）：記為 None
    for k in range(5):
        h, a = home.sets[k], away.sets[k]
        if h in ("00", "") and a in ("00", ""):
            home.sets[k] = away.sets[k] = None
        else:
            home.sets[k], away.sets[k] = safe_int(h), safe_int(a)

    sides = {home.name: home, away.name: away}
    _parse_players(soup, sides, title)
    return MatchPage(
        cup_id=cup_id, gender=gender, match_date=match_date,
        start_time=date_m.group(3), venue=venue_m.group(1) if venue_m else None,
        official_no=official_no, week=week, stage=stage,
        round_name=f"{stage} Week {week}" if week is not None else None,
        game_label=f"Game {official_no}" if official_no is not None else None,
        is_golden_set=1 if "黃金" in title else 0, title=title, home=home, away=away,
    )


def _insert_match(conn: sqlite3.Connection, season: Season, page: MatchPage, game_id: int) -> bool:
    """matches 以 (game_id, gender) 判斷存在，存在就不動（補缺不清表）；回傳是否新增。"""
    if conn.execute("SELECT 1 FROM matches WHERE game_id = ? AND gender = ?", (game_id, page.gender)).fetchone():
        return False
    h, a = page.home, page.away
    conn.execute(
        """INSERT INTO matches (game_id, gender, cup_id, match_date, start_time, venue, round_name, game_label,
               is_golden_set, home_team, away_team, home_team_id, away_team_id,
               home_set1, home_set2, home_set3, home_set4, home_set5, home_total,
               away_set1, away_set2, away_set3, away_set4, away_set5, away_total,
               home_sets_won, away_sets_won)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (game_id, page.gender, season.cup_id, page.match_date, page.start_time, page.venue, page.round_name,
         page.game_label, page.is_golden_set, h.name, a.name, h.team_id, a.team_id,
         *h.sets, h.total, *a.sets, a.total, h.sets_won, a.sets_won),
    )
    return True


def _insert_registrations_and_stats(
    conn: sqlite3.Connection, season: Season, resolver: PlayerResolver, page: MatchPage,
    week_label: str, week_start: str,
) -> tuple[int, int]:
    """寫週登錄與單場數據，回傳 (週登錄 upsert 數, 新增統計列數)。"""
    registrations = stats = 0
    for side, other in ((page.home, page.away), (page.away, page.home)):
        for p in side.players:
            pid = resolver.resolve(side.team_id, page.gender, p["name"], p["jersey_number"])
            if pid is None:
                continue
            reg_row = {
                "team_id": side.team_id, "team_gender": page.gender,
                "jersey_number": p["jersey_number"], "position": p["position"],
            }
            upsert_roster_registration(conn, pid, reg_row, week_label, week_start, "match_page", season.cup_id)
            registrations += 1
            record = {"match_date": page.match_date, "is_golden_set": page.is_golden_set}
            if not filter_new_records(conn, pid, [record]):
                continue
            rid = find_registration_id(conn, pid, side.team_id, page.gender, season.cup_id, week_label)
            insert_stat_rows(conn, [(
                rid, page.match_date, other.name, None,
                p["attack_total"], p["attack_points"], p["block_points"],
                p["serve_total"], p["serve_points"],
                p["receive_total"], p["receive_excellent"],
                p["dig_total"], p["dig_excellent"],
                p["set_total"], p["set_excellent"], p["total_points"],
                page.is_golden_set,
            )])
            stats += 1
    return registrations, stats


def import_match(
    conn: sqlite3.Connection, season: Season, resolver: PlayerResolver, page: MatchPage, match_id: int,
) -> tuple[bool, dict]:
    """寫入一場（matches、週登錄、單場數據），回傳 (是否新增 matches 列, 統計 dict)。

    game_id = cup_id * 1000 + 外部 MatchID：加賽季前綴，避免與官網 game_id 及其他回填賽季衝突。
    週次以 matches.round_name（同日期、同賽季）為準：標題有週次時由本場寫入，季後賽標題無週次時
    可事先從官網單場頁補 round_name 再重跑。查無就只寫 matches，並把該場列入 stats['unresolved']。
    """
    game_id = season.cup_id * 1000 + match_id
    inserted = _insert_match(conn, season, page, game_id)
    stats = {"registrations": 0, "stats": 0, "unresolved": None}
    week = None
    try:
        week = resolve_week_label(conn, page.match_date, page.title, season.cup_id)
    except WeekLabelUnresolved as e:
        logger.warning("%s", e)
    if week is None:
        stats["unresolved"] = {
            "kind": "week_unresolved", "cup_id": season.cup_id, "match_id": match_id,
            "game_id": game_id, "title": page.title,
        }
    else:
        stats["registrations"], stats["stats"] = _insert_registrations_and_stats(
            conn, season, resolver, page, *week,
        )
    conn.commit()
    return inserted, stats


def _is_unplayed(page: MatchPage) -> bool:
    """兩隊各局皆無得分（系列賽提前結束的未比賽場次）：不寫入，與官網「未比賽」一致。"""
    if any(s is not None for side in (page.home, page.away) for s in side.sets):
        return False
    logger.info("未比賽，略過：%s", page.title)
    return True


def run_backfill(
    conn: sqlite3.Connection, cup_id: int, match_ids: list[int] | None = None, gap_limit: int = 20,
) -> BackfillResult:
    """逐場回填。被擋時不拋出：已收集的結果保留，blocked_at 記被擋的 MatchID。"""
    season = ensure_backfill_season(conn, cup_id)
    conn.commit()
    resolver = PlayerResolver(conn, cup_id)
    result = BackfillResult()

    def process(match_id: int) -> bool:
        """處理一個 MatchID，回傳該號是否有賽事頁。"""
        html = fetch_match_page(cup_id, match_id)
        if html is None:
            return False
        result.pages += 1
        page = parse_match_page(html, cup_id, season)
        if page is None or _is_unplayed(page):
            result.skipped += 1
        else:
            inserted, stats = import_match(conn, season, resolver, page, match_id)
            result.new_matches += int(inserted)
            result.registrations += stats["registrations"]
            result.stats_inserted += stats["stats"]
            if stats["unresolved"]:
                result.unresolved.append(stats["unresolved"])
                result.skipped += 1
        result.last_match_id = match_id
        return True

    current = None
    try:
        if match_ids is not None:
            for current in match_ids:
                process(current)
        else:
            gap, current = 0, 0
            while gap < gap_limit:
                current += 1
                gap = 0 if process(current) else gap + 1
    except BackfillBlocked as e:
        result.blocked_at, result.blocked_reason = current, str(e)
    resolver.log_reports()
    return result


def build_review_list(conn: sqlite3.Connection, cup_id: int) -> list[dict]:
    """只讀 DB 計算待人工確認清單（不含 week_unresolved，那類由匯入階段收集）；重跑結果相同。"""
    players = {
        pid: (name, gender)
        for pid, name, gender in conn.execute("SELECT player_id, name, gender FROM players")
    }
    cup_teams: dict[int, set[int]] = {}
    for pid, team_id in conn.execute(
        "SELECT DISTINCT player_id, team_id FROM roster_registrations WHERE cup_id = ?", (cup_id,)
    ):
        cup_teams.setdefault(pid, set()).add(team_id)
    other_teams: dict[int, dict[int, set[int]]] = {}
    for pid, other_cup, team_id in conn.execute(
        "SELECT DISTINCT player_id, cup_id, team_id FROM roster_registrations WHERE cup_id != ?", (cup_id,)
    ):
        other_teams.setdefault(pid, {}).setdefault(other_cup, set()).add(team_id)
    by_norm: dict[str, list[int]] = {}
    for pid, (name, _g) in players.items():
        by_norm.setdefault(normalize_name(name), []).append(pid)

    items: list[dict] = []

    def add(kind: str, pid: int, evidence) -> None:
        name, gender = players[pid]
        items.append({
            "kind": kind, "cup_id": cup_id, "player_id": pid, "name": name, "gender": gender,
            "team_ids": sorted(cup_teams[pid]), "evidence": evidence,
        })

    for pid in sorted(cup_teams):
        name, gender = players[pid]
        if pid not in other_teams:
            add("new_player", pid, "其他賽季沒有任何週登錄，本次回填才出現的身分")
        if len(cup_teams[pid]) > 1:
            add("same_name_multiple_teams_in_season", pid, {"team_ids": sorted(cup_teams[pid])})
        others = [
            {"player_id": o, "gender": players[o][1]}
            for o in by_norm[normalize_name(name)] if o != pid and players[o][1] != gender
        ]
        if others:
            add("same_name_other_group", pid, others)
        history = other_teams.get(pid, {})
        differing = {c: sorted(t) for c, t in history.items() if t - cup_teams[pid]}
        if differing:
            add("same_name_other_team_in_history", pid, {str(c): t for c, t in sorted(differing.items())})
    items.sort(key=lambda i: (i["kind"], i["name"], i["player_id"]))
    return items


def write_review(path: Path, items: list[dict]) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for i in items:
        counts[i["kind"]] = counts.get(i["kind"], 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="歷史賽季一次性回填（ADR 0010；只在隔離 DB 跑）")
    parser.add_argument("--cup-id", type=int, required=True)
    parser.add_argument("--db", required=True, help="隔離 DB 路徑（需已套用 schema；team_seed 由本工具套用）；不提供預設正式 DB")
    parser.add_argument("--review-out", required=True, help="待人工確認清單 JSON 輸出路徑")
    parser.add_argument("--match-ids", help="外部 MatchID，如 1-150 或 1,2,101；省略則從 1 起逐號掃")
    parser.add_argument("--gap-limit", type=int, default=20, help="連續幾個空號就停（掃描模式）")
    args = parser.parse_args(argv)

    reject_external_path_for_official_cup(args.cup_id)
    if args.cup_id not in BACKFILL_SEASONS or args.cup_id not in EXT_TEAM_MAPS:
        raise SystemExit(
            f"cup_id={args.cup_id} 尚未定義回填：需同時在 constants.BACKFILL_SEASONS 與 EXT_TEAM_MAPS 新增（ADR 0010）。"
        )
    db_path = Path(args.db)
    conn = get_connection(db_path=db_path)
    try:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'matches'").fetchone():
            raise SystemExit(f"{db_path} 尚未套用 schema.sql，請先初始化隔離 DB。")
        explicit = parse_match_ids(args.match_ids) if args.match_ids else None
        result = run_backfill(conn, args.cup_id, explicit, args.gap_limit)
        counts = write_review(Path(args.review_out), build_review_list(conn, args.cup_id) + result.unresolved)
    finally:
        conn.close()

    if result.blocked_at is not None:
        print(
            f"已處理到 MatchID {result.last_match_id}、在 MatchID {result.blocked_at} 被擋（{result.blocked_reason}）。"
            "已 commit 的場次保留；請稍後人工確認後再重跑（補缺不清表）。"
        )
        return 2
    print("===== 回填摘要 =====")
    print(f"抓取頁數：{result.pages}")
    print(f"新增賽事：{result.new_matches}")
    print(f"週登錄 upsert：{result.registrations}")
    print(f"新增統計列：{result.stats_inserted}")
    print(f"略過（明星賽／未比賽／週次不明）：{result.skipped}")
    print(f"待人工確認：{counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
