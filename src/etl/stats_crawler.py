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

from src.utils.db_config import DB_PATH, get_connection
from src.utils.logger import get_logger
from src.utils.constants import (
    EXT_BASE, EXT_CUP_ID as CUP_ID, EXT_HEADERS as HEADERS,
    SEASON_YEAR_MAP, DEFAULT_YEAR, EXT_TEAM_MAP,
    MATCH_POSITION_MAP, OPP_SHORT_TO_TEAM,
)

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"

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
    """確保 player_match_stats 表存在（讀 schema.sql，冪等，不清空既有資料）。"""
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    logger.info("player_match_stats 表已確認存在（DDL 來源：schema.sql）")


def build_name_to_pid(conn: sqlite3.Connection) -> dict[tuple[str, str], int]:
    """建立 {(正規化姓名, gender): player_id} 的查找表。
    同名不同性別是不同球員；同名同性別仍會碰撞（已知限制，罕見）。"""
    rows = conn.execute("SELECT player_id, name, gender FROM players").fetchall()
    return {(normalize_name(name), gender): pid for pid, name, gender in rows}


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

    # 偵測黃金決勝局：同日期同對手出現兩筆時，局數較少的為黃金局
    seen: dict[tuple, int] = {}  # (match_date, opponent) -> index
    for i, r in enumerate(records):
        key = (r["match_date"], r["opponent"])
        if key in seen:
            prev_i = seen[key]
            # 局數較少的標記為黃金局
            if (records[prev_i]["sets_played"] or 0) <= (r["sets_played"] or 0):
                records[prev_i]["is_golden_set"] = 1
            else:
                r["is_golden_set"] = 1
        else:
            seen[key] = i

    for r in records:
        r.setdefault("is_golden_set", 0)

    return records


def get_existing_keys(conn: sqlite3.Connection, player_id: int) -> set[tuple]:
    """取得某球員已存在的 (match_date, is_golden_set) 集合（用於去重比對）。
    Phase 2 起 player_match_stats 無 player_id 欄，改經 JOIN roster_registrations 查詢；
    回傳的鍵集合語意不變。"""
    rows = conn.execute(
        """SELECT s.match_date, s.is_golden_set
           FROM player_match_stats s
           JOIN roster_registrations r ON s.registration_id = r.registration_id
           WHERE r.player_id = ?""",
        (player_id,),
    ).fetchall()
    return {(r[0], r[1]) for r in rows}


def filter_new_records(
    conn: sqlite3.Connection, player_id: int, records: list[dict]
) -> list[dict]:
    """過濾出某球員尚未存在的紀錄（去重鍵：match_date + is_golden_set）。

    全量、增量模式皆呼叫此函式，避免重跑造成重複列（無 UNIQUE 約束擋不住）。
    """
    existing = get_existing_keys(conn, player_id)
    return [
        r for r in records
        if (r["match_date"], r["is_golden_set"]) not in existing
    ]


def main(incremental: bool = False):
    """
    主流程：抓取所有球員逐場數據並寫入 DB。

    Parameters
    ----------
    incremental : bool
        True = 增量模式，只新增尚未存在的比賽紀錄（不清除既有資料）。
        False = 全量模式，掃描全部場次，但一樣只插入缺少的紀錄
                （去重鍵：match_date + is_golden_set，逐球員判斷），不清空既有資料。
    """
    conn = get_connection()
    init_stats_table(conn)

    if not incremental:
        logger.warning(
            "全量模式：掃描全部場次，但去重機制與增量模式相同——"
            "只插入尚未存在的紀錄（match_date + is_golden_set，逐球員判斷），不會清空既有資料。"
        )
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

            db_team_id, gender = EXT_TEAM_MAP[ext_team_id]

            # 正規化比對
            player_id = name_map.get((norm_name, gender))

            # Late Arriving Dimension：查無此人則動態新增（僅身分層欄位，
            # team_id/背號/位置一律由 roster_registrations 維護）
            if player_id is None:
                logger.info(
                    "[動態新增] 發現新球員: %s，自動寫入 players 表（僅身分層欄位）。",
                    name,
                )
                cursor = conn.execute(
                    "INSERT INTO players (name, gender) VALUES (?, ?)",
                    (name, gender),
                )
                conn.commit()
                player_id = cursor.lastrowid
                name_map[(norm_name, gender)] = player_id
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

            # 去重：不論全量或增量，皆只插入尚未存在的紀錄（含黃金局區分）
            new_records = filter_new_records(conn, player_id, records)
            total_skipped += len(records) - len(new_records)
            records = new_records
            if not records:
                continue

            # 逐筆解析 registration_id（Phase 2：player_match_stats 改掛
            # registration_id，不同筆記錄可能落在不同週次，須逐筆反查）
            rows_to_insert = [
                (
                    resolve_registration_for_stats(conn, player_id, db_team_id, gender, r["match_date"]),
                    r["match_date"], r["opponent"], r["sets_played"],
                    r["attack_total"], r["attack_points"], r["block_points"],
                    r["serve_total"], r["serve_points"],
                    r["receive_total"], r["receive_excellent"],
                    r["dig_total"], r["dig_excellent"],
                    r["set_total"], r["set_excellent"], r["total_points"],
                    r["is_golden_set"],
                )
                for r in records
            ]

            # 批次寫入
            conn.executemany(
                """INSERT INTO player_match_stats
                   (registration_id, match_date, opponent, sets_played,
                    attack_total, attack_points, block_points,
                    serve_total, serve_points,
                    receive_total, receive_excellent,
                    dig_total, dig_excellent,
                    set_total, set_excellent, total_points,
                    is_golden_set)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows_to_insert,
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
    print(f"跳過（已存在）：{total_skipped} 筆")
    print(f"動態新增球員數：{total_new_players}")
    print(f"players 表總人數：{total_players}")

    print(f"\n===== 前 3 筆資料 =====")
    df = pd.read_sql_query(
        """SELECT s.stat_id, p.name, s.match_date, s.opponent,
                  s.sets_played, s.attack_points, s.block_points,
                  s.serve_points, s.total_points
           FROM player_match_stats s
           JOIN roster_registrations r ON s.registration_id = r.registration_id
           JOIN players p ON r.player_id = p.player_id
           LIMIT 3""",
        conn,
    )
    print(df.to_string(index=False))

    conn.close()
    logger.info("事實表載入完成：%s", DB_PATH)


# ── 出賽名單爬蟲（Phase 2：週次登錄資料來源） ──────────────────

def fetch_match_list(cup_id: int = CUP_ID) -> list[dict]:
    """
    透過 Match.aspx 的下拉選單取得所有 MatchID 清單。
    MatchID 本身不連續，不可用逐一 range 掃描列舉，必須用此函式。
    """
    url = f"{EXT_BASE}/Match.aspx"
    r = requests.get(url, params={"CupID": cup_id}, headers=HEADERS, timeout=15)
    r.raise_for_status()
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "html.parser")
    sel = soup.find("select", id="divSelect")
    if not sel:
        return []
    result = []
    for opt in sel.find_all("option"):
        value = opt.get("value")
        if not value:
            continue
        result.append({"match_id": int(value), "label": opt.get_text(strip=True)})
    return result


def _parse_match_title(title_text: str) -> tuple[str, str] | None:
    """從標題文字解析 (match_date, raw_round_text)。解析失敗回傳 None。"""
    date_m = re.search(r"(\d{1,2})月(\d{1,2})日", title_text)
    if not date_m:
        return None
    month, day = int(date_m.group(1)), int(date_m.group(2))
    year = SEASON_YEAR_MAP.get(month, DEFAULT_YEAR)
    match_date = f"{year}-{month:02d}-{day:02d}"

    round_m = re.search(r"(第\d+週|挑戰賽|總決賽|季後賽|明星賽)", title_text)
    raw_round_text = round_m.group(1) if round_m else "未知賽別"
    return match_date, raw_round_text


def fetch_match_roster(cup_id: int, match_id: int) -> list[dict] | None:
    """
    抓取單場出賽名單，回傳每位球員一筆 dict：
    {'match_date', 'title_text', 'team_id', 'team_gender',
     'jersey_number', 'name', 'position'}
    頁面無效（如錯誤頁、無比賽資料）回傳 None。

    欄位順序注意（與 fetch_player_stats() 不同，不可混用）：
    每列扣除背號/姓名/位置後，共 12 個數值欄位，順序固定為
    [攻擊得, 攻擊總, 攔網得, 發球得, 發球總,
     接發好, 接發總, 防守好, 防守總, 舉球好, 舉球總, 總得分]
    """
    url = f"{EXT_BASE}/_handler/Match.ashx"
    r = requests.get(
        url, params={"CupID": cup_id, "MatchID": match_id},
        headers=HEADERS, timeout=15,
    )
    r.raise_for_status()
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "html.parser")

    title_h3 = soup.find("h3")
    if title_h3 is None:
        return None
    title_text = title_h3.get_text(" ", strip=True)
    if "組" not in title_text:
        return None  # 錯誤頁不含「組」字

    parsed = _parse_match_title(title_text)
    if parsed is None:
        logger.warning("[MatchID=%d] 無法解析日期，跳過：%s", match_id, title_text)
        return None
    match_date, _raw_round_text = parsed

    team_h3s = soup.find_all("h3")[1:]
    rows = []
    for team_h3 in team_h3s:
        team_text = team_h3.get_text(strip=True)
        if "：" not in team_text:
            continue
        team_name = team_text.split("：", 1)[0].strip()
        team_info = OPP_SHORT_TO_TEAM.get(team_name)
        if team_info is None:
            logger.warning("[MatchID=%d] 無法辨識隊名：%s，跳過該隊", match_id, team_name)
            continue
        team_id, team_gender = team_info

        table = team_h3.find_next("table")
        if table is None:
            continue

        for tr in table.find_all("tr")[2:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if not cells or cells[0] in ("全隊合計", ""):
                continue
            if len(cells) < 15:
                continue

            position_raw = cells[2]
            if position_raw and position_raw not in MATCH_POSITION_MAP:
                logger.warning(
                    "[MatchID=%d] 未知位置用語：%s（球員 %s），記為 None",
                    match_id, position_raw, cells[1]
                )
            rows.append({
                "match_date": match_date,
                "title_text": title_text,
                "team_id": team_id,
                "team_gender": team_gender,
                "jersey_number": safe_int(cells[0]),
                "name": cells[1],
                "position": MATCH_POSITION_MAP.get(position_raw),
            })

    return rows


def resolve_week_label(conn: sqlite3.Connection, match_date: str, title_text: str) -> tuple[str, str]:
    """
    回傳 (week_label, week_start_date)。
    優先用 matches.round_name 反查（權威來源）；查無則退化用標題文字，
    並記錄警告（此為已知限制，非本次遷移的資料涵蓋範圍）。
    """
    row = conn.execute(
        "SELECT round_name FROM matches WHERE match_date = ? ORDER BY round_name LIMIT 1",
        (match_date,),
    ).fetchone()
    if row and row[0]:
        week_label = row[0]
        start_row = conn.execute(
            "SELECT MIN(match_date) FROM matches WHERE round_name = ? "
            "AND ABS(julianday(match_date) - julianday(?)) < 200",
            (week_label, match_date),
        ).fetchone()
        week_start_date = start_row[0] if start_row and start_row[0] else match_date
        return week_label, week_start_date

    logger.warning(
        "match_date=%s 在 matches 表查無 round_name，退化用 match_date 當 week_label：%s",
        match_date, title_text,
    )
    return f"未比對-{match_date}", match_date


def upsert_roster_registration(
    conn: sqlite3.Connection, player_id: int, row: dict,
    week_label: str, week_start_date: str, source: str = "match_page",
    cup_id: int = CUP_ID,
) -> None:
    """upsert 一筆 roster_registrations。source 預設 'match_page'（真實出賽名單）；
    統計寫入路徑查無登錄時會傳入 source='backfill'，補一筆背號/位置皆 NULL 的登錄。
    cup_id 為賽季限定鍵：不同賽季的同名週次是不同登錄，不互相覆寫。"""
    conn.execute(
        """
        INSERT INTO roster_registrations
            (player_id, team_id, gender, cup_id, week_label, week_start_date, jersey_number, position, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (player_id, team_id, gender, cup_id, week_label) DO UPDATE SET
            jersey_number = excluded.jersey_number,
            position = excluded.position,
            week_start_date = excluded.week_start_date,
            source = excluded.source
        """,
        (player_id, row["team_id"], row["team_gender"], cup_id, week_label,
         week_start_date, row["jersey_number"], row["position"], source),
    )


def resolve_registration_for_stats(
    conn: sqlite3.Connection, player_id: int, team_id: int, gender: str, match_date: str,
    cup_id: int = CUP_ID,
) -> int:
    """
    統計寫入路徑（Player.ashx 逐場數值統計，無背號/位置資訊）解析 registration_id。
    解析順序：match_date -> resolve_week_label() 取 week_label -> 以含 cup_id 的
    五元鍵查 roster_registrations -> 查無則以 source='backfill' 補一筆
    （背號/位置皆 NULL，只標記不插補，不得用其他資料推測填入）。
    """
    week_label, week_start_date = resolve_week_label(conn, match_date, "")

    row = conn.execute(
        """SELECT registration_id FROM roster_registrations
           WHERE player_id = ? AND team_id = ? AND gender = ? AND cup_id = ? AND week_label = ?""",
        (player_id, team_id, gender, cup_id, week_label),
    ).fetchone()
    if row:
        return row[0]

    upsert_roster_registration(
        conn, player_id,
        {"team_id": team_id, "team_gender": gender, "jersey_number": None, "position": None},
        week_label, week_start_date, source="backfill", cup_id=cup_id,
    )
    row = conn.execute(
        """SELECT registration_id FROM roster_registrations
           WHERE player_id = ? AND team_id = ? AND gender = ? AND cup_id = ? AND week_label = ?""",
        (player_id, team_id, gender, cup_id, week_label),
    ).fetchone()
    return row[0]


def crawl_all_rosters(conn: sqlite3.Connection, cup_id: int = CUP_ID) -> dict:
    """批次爬取全部場次出賽名單並 upsert 進 roster_registrations。"""
    name_map = build_name_to_pid(conn)
    match_list = fetch_match_list(cup_id)
    stats = {"matches_scanned": 0, "matches_skipped": 0, "registrations_upserted": 0, "new_players": 0}

    for m in match_list:
        try:
            roster_rows = fetch_match_roster(cup_id, m["match_id"])
        except requests.RequestException as e:
            logger.warning(
                "[MatchID=%d] 抓取失敗（%s），跳過該場",
                m["match_id"], e
            )
            stats["matches_skipped"] += 1
            continue

        if not roster_rows:
            stats["matches_skipped"] += 1
            continue
        stats["matches_scanned"] += 1

        match_date = roster_rows[0]["match_date"]
        title_text = roster_rows[0]["title_text"]
        week_label, week_start_date = resolve_week_label(conn, match_date, title_text)

        for row in roster_rows:
            norm = normalize_name(row["name"])
            player_id = name_map.get((norm, row["team_gender"]))
            if player_id is None:
                cursor = conn.execute(
                    "INSERT INTO players (name, gender) VALUES (?, ?)",
                    (row["name"], row["team_gender"]),
                )
                player_id = cursor.lastrowid
                name_map[(norm, row["team_gender"])] = player_id
                stats["new_players"] += 1

            upsert_roster_registration(conn, player_id, row, week_label, week_start_date, cup_id=cup_id)
            stats["registrations_upserted"] += 1

        conn.commit()
        time.sleep(0.5)

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TVL 球員逐場數據爬蟲")
    parser.add_argument(
        "--incremental", "-i", action="store_true",
        help="增量模式：僅新增尚未存在的比賽紀錄，不清除既有資料",
    )
    parser.add_argument(
        "--rosters", action="store_true",
        help="改跑出賽名單爬蟲（crawl_all_rosters），寫入 roster_registrations，"
             "跳過技術統計爬蟲；請先跑過 match_crawler",
    )
    args = parser.parse_args()
    if args.rosters:
        _conn = get_connection()
        try:
            _stats = crawl_all_rosters(_conn)
            print(_stats)
        finally:
            _conn.close()
    else:
        main(incremental=args.incremental)
