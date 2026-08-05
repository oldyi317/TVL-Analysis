"""
TVL 球員逐場數據爬蟲模組
從外部數據系統 (114.35.229.141) 抓取球員逐場統計，
透過球員姓名與本地 DB 關聯後 upsert 至 player_match_stats 事實表（依 season 隔離，冪等）。
"""

import re
import time

import requests
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.etl.db_loader import init_db
from src.utils.db_config import get_engine
from src.utils.logger import get_logger
from src.utils.constants import (
    EXT_BASE, EXT_CUP_ID as CUP_ID, EXT_HEADERS as HEADERS,
    SEASON, EXT_TEAM_MAP, season_year_for_month,
)

logger = get_logger(__name__)


def normalize_name(name: str) -> str:
    """正規化姓名：去除全形/半形空白、轉小寫、去除不間斷空白。"""
    return re.sub(r"[\s　\xa0]+", "", name).lower()


def safe_int(val: str) -> int | None:
    """安全轉換整數，失敗回傳 None。"""
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def parse_match_date(raw: str) -> str | None:
    """
    從 '311/01' 格式中萃取日期並轉為 YYYY-MM-DD。
    前面的數字是場次編號，後面 MM/DD 是日期，年份依 SEASON 設定推算。
    """
    m = re.search(r"(\d{1,2})/(\d{2})$", raw)
    if not m:
        return None
    month = int(m.group(1))
    day = int(m.group(2))
    year = season_year_for_month(month)
    return f"{year}-{month:02d}-{day:02d}"


def build_name_to_pid(engine: Engine, season: str = SEASON) -> dict[str, int]:
    """建立當前賽季 {正規化姓名: player_id} 的查找表（各賽季名單獨立）。"""
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT player_id, name FROM players WHERE season = :season"),
            {"season": season},
        ).fetchall()
    return {normalize_name(name): pid for pid, name in rows}


def fetch_player_list(team_id: int) -> list[dict]:
    """從外部系統取得某隊的球員清單。回傳 [{'ext_player_id': int, 'name': str}, ...]"""
    url = f"{EXT_BASE}/_handler/PlayerList.ashx"
    r = requests.get(
        url, params={"CupID": CUP_ID, "TeamID": team_id},
        headers=HEADERS, timeout=15,
    )
    soup = BeautifulSoup(r.text, "html.parser")
    players = []
    for opt in soup.find_all("option"):
        text_ = opt.get_text(strip=True)  # e.g. "No.2-黃宇晨"
        ext_id = opt.get("value")
        name = text_.split("-", 1)[1] if "-" in text_ else text_
        players.append({"ext_player_id": int(ext_id), "name": name})
    return players


def fetch_player_stats(team_id: int, ext_player_id: int) -> list[dict]:
    """抓取單一球員的逐場數據表，回傳字典列表。跳過表頭行與最後的「累計」行。"""
    url = f"{EXT_BASE}/_handler/Player.ashx"
    r = requests.get(
        url,
        params={"CupID": CUP_ID, "PlayerID": ext_player_id, "TeamID": team_id},
        headers=HEADERS, timeout=15,
    )
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    rows = table.find_all("tr")
    records = []
    for row in rows[2:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
        if not cells or cells[0] == "累計":
            continue
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
    seen: dict[tuple, int] = {}
    for i, r in enumerate(records):
        key = (r["match_date"], r["opponent"])
        if key in seen:
            prev_i = seen[key]
            if (records[prev_i]["sets_played"] or 0) <= (r["sets_played"] or 0):
                records[prev_i]["is_golden_set"] = 1
            else:
                r["is_golden_set"] = 1
        else:
            seen[key] = i

    for r in records:
        r.setdefault("is_golden_set", 0)

    return records


def upsert_stats(engine: Engine, player_id: int, records: list[dict], season: str) -> None:
    """
    將單一球員的逐場數據 upsert 至 player_match_stats（唯一鍵含 season，永不觸碰其他賽季）。

    Normalizes NULL keys for idempotency:
    - Skips records with match_date=None (無法判定比賽日期，無用於下游分析)
    - Coalesces opponent=None to "" (確保 UNIQUE 鍵冪等碰撞)
    """
    skipped = 0
    rows = []
    for r in records:
        if r["match_date"] is None:
            logger.warning(
                "跳過無法解析日期的統計紀錄 (player_id=%d, opponent=%s)",
                player_id, r.get("opponent"),
            )
            skipped += 1
            continue

        rows.append({
            "player_id": player_id,
            "season": season,
            "match_date": r["match_date"],
            "opponent": r["opponent"] or "",  # Coalesce NULL to "" for unique key idempotency
            "sets_played": r["sets_played"],
            "attack_total": r["attack_total"],
            "attack_points": r["attack_points"],
            "block_points": r["block_points"],
            "serve_total": r["serve_total"],
            "serve_points": r["serve_points"],
            "receive_total": r["receive_total"],
            "receive_excellent": r["receive_excellent"],
            "dig_total": r["dig_total"],
            "dig_excellent": r["dig_excellent"],
            "set_total": r["set_total"],
            "set_excellent": r["set_excellent"],
            "total_points": r["total_points"],
            "is_golden_set": r["is_golden_set"],
        })

    if skipped > 0:
        logger.info("player_id=%d: 跳過 %d 筆無效日期紀錄", player_id, skipped)

    if not rows:
        return

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO player_match_stats
                (player_id, season, match_date, opponent, sets_played,
                 attack_total, attack_points, block_points,
                 serve_total, serve_points,
                 receive_total, receive_excellent,
                 dig_total, dig_excellent,
                 set_total, set_excellent, total_points, is_golden_set)
            VALUES
                (:player_id, :season, :match_date, :opponent, :sets_played,
                 :attack_total, :attack_points, :block_points,
                 :serve_total, :serve_points,
                 :receive_total, :receive_excellent,
                 :dig_total, :dig_excellent,
                 :set_total, :set_excellent, :total_points, :is_golden_set)
            ON CONFLICT (player_id, season, match_date, opponent, is_golden_set) DO UPDATE SET
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
        """), rows)


def get_or_create_player(
    engine: Engine, name: str, ext_team_id: int, season: str = SEASON
) -> int | None:
    """
    取得或動態建立球員。

    若球員已在本賽季名單中，返回既有 player_id。
    若查無此人，檢查 ext_team_id 是否已知；若已知則動態建立並返回新 player_id。
    若 ext_team_id 未知則返回 None。
    """
    norm_name = normalize_name(name)
    name_map = build_name_to_pid(engine, season)
    if norm_name in name_map:
        return name_map[norm_name]

    if ext_team_id not in EXT_TEAM_MAP:
        logger.error("ext_team_id=%d 未知，無法建立球員 %s", ext_team_id, name)
        return None

    db_team_id, gender = EXT_TEAM_MAP[ext_team_id]
    logger.info("[動態新增] 發現新球員: %s，自動寫入 players 表。", name)
    with engine.begin() as conn:
        player_id = conn.execute(
            text(
                "INSERT INTO players (name, team_id, gender, season) "
                "VALUES (:name, :team_id, :gender, :season) RETURNING player_id"
            ),
            {"name": name, "team_id": db_team_id, "gender": gender, "season": season},
        ).scalar_one()
    return player_id


def _scalar(engine: Engine, sql: str):
    with engine.begin() as conn:
        return conn.execute(text(sql)).scalar_one()


def main(incremental: bool = False):
    """
    主流程：抓取所有球員逐場數據並 upsert 至 DB。

    Parameters
    ----------
    incremental : bool
        僅影響記錄檔訊息；全量與增量模式皆一律 upsert（冪等），
        不再有「全量先砍表重建」與「增量只新增缺少紀錄」的行為差異。
    """
    engine = get_engine()
    init_db(engine)  # CREATE TABLE IF NOT EXISTS，永不清空既有資料

    mode_label = "增量" if incremental else "全量"
    logger.info("%s模式：upsert 所有球員逐場數據（不再砍表重建）", mode_label)

    name_map = build_name_to_pid(engine, SEASON)

    total_upserted = 0
    total_new_players = 0

    for ext_team_id in range(1, 10):
        players = fetch_player_list(ext_team_id)
        logger.info("TeamID=%d: %d 位球員", ext_team_id, len(players))

        for p in players:
            ext_pid = p["ext_player_id"]
            name = p["name"]

            player_id = get_or_create_player(engine, name, ext_team_id, SEASON)
            if player_id is None:
                logger.warning("無法取得/建立球員 %s (ext_team_id=%d)，略過", name, ext_team_id)
                continue

            # 重新初始化 name_map 以包含新增的球員
            norm_name = normalize_name(name)
            if norm_name not in name_map:
                name_map[norm_name] = player_id
                total_new_players += 1

            try:
                records = fetch_player_stats(ext_team_id, ext_pid)
            except Exception as e:
                logger.error("抓取球員 [%s] 數據失敗: %s", name, e)
                continue

            if not records:
                continue

            upsert_stats(engine, player_id, records, SEASON)
            total_upserted += len(records)

            time.sleep(0.5)

    total_rows = _scalar(engine, "SELECT COUNT(*) FROM player_match_stats")
    total_players = _scalar(engine, "SELECT COUNT(*) FROM players")

    print(f"\n===== {mode_label} upsert 完成 =====")
    print(f"player_match_stats 總筆數：{total_rows}")
    print(f"本次 upsert：{total_upserted} 筆")
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
        engine,
    )
    print(df.to_string(index=False))

    logger.info("事實表載入完成")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TVL 球員逐場數據爬蟲")
    parser.add_argument(
        "--incremental", "-i", action="store_true",
        help="增量模式（僅影響記錄檔訊息，實際皆為 upsert）",
    )
    args = parser.parse_args()
    main(incremental=args.incremental)
