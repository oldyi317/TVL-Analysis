"""
每周戰報資料彙整模組
從 SQLite 撈取指定日期範圍內的比賽數據，產生結構化摘要供 Claude API 使用。
"""

import sqlite3
from datetime import datetime, timedelta

import pandas as pd

from pathlib import Path

try:
    from src.utils.db_config import get_connection
except ModuleNotFoundError:
    def get_connection(foreign_keys=True):
        _db = Path(__file__).resolve().parents[2] / "data" / "db" / "tvl_database.db"
        return sqlite3.connect(_db)


def _conn():
    return get_connection(foreign_keys=False)


def get_match_weeks() -> list[tuple[str, str]]:
    """
    回傳所有比賽周次的 (week_start, week_end) 列表。
    以 ISO 周次分組，方便使用者選擇。
    """
    conn = _conn()
    try:
        dates = pd.read_sql(
            "SELECT DISTINCT match_date FROM player_match_stats ORDER BY match_date",
            conn,
        )["match_date"].tolist()
    finally:
        conn.close()

    if not dates:
        return []

    weeks: dict[tuple[int, int], list[str]] = {}
    for d in dates:
        dt = datetime.strptime(d, "%Y-%m-%d")
        iso_year, iso_week, _ = dt.isocalendar()
        key = (iso_year, iso_week)
        weeks.setdefault(key, []).append(d)

    result = []
    for (_y, _w), ds in sorted(weeks.items()):
        result.append((min(ds), max(ds)))
    return result


def gather_weekly_data(
    date_from: str, date_to: str, gender_filter: str | None = None
) -> dict:
    """
    彙整指定日期範圍內的所有比賽數據，回傳結構化 dict。

    Parameters
    ----------
    date_from : 起始日期 (YYYY-MM-DD)
    date_to : 結束日期 (YYYY-MM-DD)
    gender_filter : "M", "F", or None (全部)

    Returns
    -------
    dict with keys: "period", "matches"
    """
    conn = _conn()
    try:
        gender_clause = "AND p.gender = ?" if gender_filter else ""
        params: tuple = (date_from, date_to)
        if gender_filter:
            params = (date_from, date_to, gender_filter)

        # 撈取該期間所有球員單場數據
        raw = pd.read_sql(
            f"""
            SELECT s.match_date, s.opponent,
                   p.player_id, p.name, p.position, p.gender,
                   t.team_id, t.team_name,
                   s.sets_played,
                   s.attack_total, s.attack_points,
                   s.block_points,
                   s.serve_total, s.serve_points,
                   s.receive_total, s.receive_excellent,
                   s.dig_total, s.dig_excellent,
                   s.set_total, s.set_excellent,
                   s.total_points,
                   s.is_golden_set
            FROM player_match_stats s
            JOIN players p ON s.player_id = p.player_id
            JOIN teams   t ON p.team_id = t.team_id AND p.gender = t.gender
            WHERE s.match_date BETWEEN ? AND ?
            {gender_clause}
            ORDER BY s.match_date, t.team_name
            """,
            conn,
            params=params,
        )

        # 撈取賽季累計（用來對比本周表現）
        season_agg = pd.read_sql(
            f"""
            SELECT p.player_id, p.name, p.position, p.gender,
                   t.team_name,
                   COUNT(*) AS season_games,
                   SUM(s.sets_played) AS season_sets,
                   SUM(s.attack_points) AS season_atk_pts,
                   SUM(s.attack_total) AS season_atk_tot,
                   SUM(s.block_points) AS season_blk_pts,
                   SUM(s.serve_points) AS season_srv_pts,
                   SUM(s.serve_total) AS season_srv_tot,
                   SUM(s.receive_excellent) AS season_rcv_exc,
                   SUM(s.receive_total) AS season_rcv_tot,
                   SUM(s.dig_excellent) AS season_dig_exc,
                   SUM(s.dig_total) AS season_dig_tot,
                   SUM(s.set_excellent) AS season_set_exc,
                   SUM(s.set_total) AS season_set_tot,
                   SUM(s.total_points) AS season_total_pts
            FROM player_match_stats s
            JOIN players p ON s.player_id = p.player_id
            JOIN teams   t ON p.team_id = t.team_id AND p.gender = t.gender
            WHERE s.match_date <= ? AND s.is_golden_set = 0
            {gender_clause}
            GROUP BY p.player_id
            HAVING COUNT(*) >= 2
            """,
            conn,
            params=(date_to,) + ((gender_filter,) if gender_filter else ()),
        )
    finally:
        conn.close()

    if raw.empty:
        return {"period": f"{date_from} ~ {date_to}", "matches": []}

    # 建立賽季平均 lookup
    season_lookup = {}
    for _, row in season_agg.iterrows():
        pid = row["player_id"]
        g = row["season_games"]
        atk_tot = row["season_atk_tot"] or 0
        season_lookup[pid] = {
            "season_games": int(g),
            "season_ppg": round(row["season_total_pts"] / g, 1) if g else 0,
            "season_asr": round(
                row["season_atk_pts"] / atk_tot * 100, 1
            ) if atk_tot > 0 else None,
        }

    def _safe_pct(num, den):
        return round(num / den * 100, 1) if den and den > 0 else None

    # 分離正規賽與黃金決勝局
    raw_regular = raw[raw["is_golden_set"] == 0]
    raw_golden = raw[raw["is_golden_set"] == 1]

    # 建立黃金局 lookup: (date, team_name, opponent) -> golden set data
    golden_lookup: dict[tuple, dict] = {}
    for (date, team_name, opponent), grp in raw_golden.groupby(
        ["match_date", "team_name", "opponent"]
    ):
        gs_team_stats = {
            "total_points": int(grp["total_points"].sum()),
            "attack_points": int(grp["attack_points"].sum()),
            "attack_total": int(grp["attack_total"].sum()),
            "block_points": int(grp["block_points"].sum()),
            "serve_points": int(grp["serve_points"].sum()),
        }
        gs_players = []
        for _, p in grp.sort_values("total_points", ascending=False).iterrows():
            if p["total_points"] == 0:
                continue
            gs_players.append({
                "name": p["name"],
                "position": p["position"],
                "total_points": int(p["total_points"]),
                "attack_points": int(p["attack_points"]),
                "block_points": int(p["block_points"]),
                "serve_points": int(p["serve_points"]),
            })
        golden_lookup[(date, team_name, opponent)] = {
            "team_stats": gs_team_stats,
            "players": gs_players,
        }

    # 逐場比賽彙整（僅正規賽）
    matches = []
    for (date, team_name, opponent), grp in raw_regular.groupby(
        ["match_date", "team_name", "opponent"]
    ):
        gender = grp.iloc[0]["gender"]
        gender_label = "男子組" if gender == "M" else "女子組"

        # 團隊總計
        team_stats = {
            "total_points": int(grp["total_points"].sum()),
            "attack_points": int(grp["attack_points"].sum()),
            "attack_total": int(grp["attack_total"].sum()),
            "attack_rate": _safe_pct(grp["attack_points"].sum(), grp["attack_total"].sum()),
            "block_points": int(grp["block_points"].sum()),
            "serve_points": int(grp["serve_points"].sum()),
            "serve_total": int(grp["serve_total"].sum()),
        }

        # 球員個人表現（僅列出有上場的）
        players = []
        for _, p in grp.sort_values("total_points", ascending=False).iterrows():
            if p["sets_played"] == 0:
                continue
            pid = int(p["player_id"])
            pdata = {
                "name": p["name"],
                "position": p["position"],
                "sets_played": int(p["sets_played"]),
                "total_points": int(p["total_points"]),
                "attack_points": int(p["attack_points"]),
                "attack_total": int(p["attack_total"]),
                "attack_rate": _safe_pct(p["attack_points"], p["attack_total"]),
                "block_points": int(p["block_points"]),
                "serve_points": int(p["serve_points"]),
                "receive_excellent": int(p["receive_excellent"]),
                "receive_total": int(p["receive_total"]),
                "receive_rate": _safe_pct(p["receive_excellent"], p["receive_total"]),
                "dig_excellent": int(p["dig_excellent"]),
                "dig_total": int(p["dig_total"]),
            }
            # 附加賽季平均對比
            if pid in season_lookup:
                sl = season_lookup[pid]
                pdata["season_ppg"] = sl["season_ppg"]
                pdata["season_asr"] = sl["season_asr"]
                pdata["vs_season_ppg"] = round(
                    p["total_points"] - sl["season_ppg"], 1
                )
            players.append(pdata)

        match_entry = {
            "date": date,
            "gender": gender_label,
            "team_name": team_name,
            "opponent": opponent,
            "team_stats": team_stats,
            "players": players,
        }

        # 附加黃金決勝局資料
        gs_key = (date, team_name, opponent)
        if gs_key in golden_lookup:
            match_entry["golden_set"] = golden_lookup[gs_key]

        matches.append(match_entry)

    return {
        "period": f"{date_from} ~ {date_to}",
        "matches": matches,
    }
