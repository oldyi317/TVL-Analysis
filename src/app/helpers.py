"""
TVL 儀表板共用函式
提供 DB 查詢、外部系統資料擷取、進階指標計算等共用功能。
"""

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

try:
    from src.utils.constants import (
        EXT_BASE, EXT_CUP_ID, EXT_HEADERS, SEASON_YEAR_MAP, OPP_SHORT_TO_TEAM,
    )
except ModuleNotFoundError:
    EXT_BASE = "http://114.35.229.141"
    EXT_CUP_ID = 21
    EXT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    SEASON_YEAR_MAP = {11: 2025, 12: 2025}
    OPP_SHORT_TO_TEAM = {}

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "tvl_database.db"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "match_predictor.pkl"


# ── DB 查詢 ──────────────────────────────────────────────────

@st.cache_data
def load_data(query: str, params: tuple = ()) -> pd.DataFrame:
    """連線 SQLite，執行查詢並回傳 DataFrame。連線使用後立即關閉。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


# ── 數值工具 ─────────────────────────────────────────────────

def safe_pct(numerator: float, denominator: float) -> float:
    """安全計算百分比，分母為 0 時回傳 0.0。"""
    return (numerator / denominator * 100) if denominator > 0 else 0.0


def vec_pct(num, den):
    """向量化百分比計算：分母為 0 時回傳 0.0。"""
    return np.where(den > 0, num / den * 100, 0.0)


# ── 外部系統資料擷取 ──────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_match_index() -> list[dict]:
    """從外部系統抓取所有比賽場次索引（MatchID, 日期, 隊伍）。"""
    try:
        r = requests.get(
            f"{EXT_BASE}/Match.aspx?CupID={EXT_CUP_ID}",
            headers=EXT_HEADERS, timeout=15,
        )
        soup = BeautifulSoup(r.text, "html.parser")
        sel = soup.find("select", id="divSelect")
        if not sel:
            return []

        pat = re.compile(r"[：](.+?)\s+vs\s+(.+?)\s+\((\d+)月(\d+)日\)")
        matches = []
        for opt in sel.find_all("option"):
            m = pat.search(opt.get_text(strip=True))
            if not m:
                continue
            month, day = int(m.group(3)), int(m.group(4))
            year = SEASON_YEAR_MAP.get(month, 2026)
            matches.append({
                "match_id": opt["value"],
                "date": f"{year}-{month:02d}-{day:02d}",
                "team_a": m.group(1).strip(),
                "team_b": m.group(2).strip(),
            })
        return matches
    except Exception:
        return []


@st.cache_data(ttl=3600)
def fetch_set_scores(match_id: str) -> list[dict] | None:
    """從外部系統取得單場比賽的局比分。回傳兩隊各局分數。"""
    try:
        r = requests.get(
            f"{EXT_BASE}/_handler/Match.ashx",
            params={"CupID": EXT_CUP_ID, "MatchID": match_id, "SetNum": 0},
            headers=EXT_HEADERS, timeout=15,
        )
        soup = BeautifulSoup(r.text, "html.parser")
        tables = soup.find_all("table")
        if len(tables) < 2:
            return None

        rows = tables[1].find_all("tr")
        result = []
        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
            if len(cells) >= 8:
                sets = cells[1:6]
                result.append({
                    "team": cells[0],
                    "sets": [int(s) if s and s != "00" else None for s in sets],
                    "total_pts": int(cells[6]) if cells[6].isdigit() else 0,
                    "sets_won": int(cells[7]) if cells[7].isdigit() else 0,
                })
        return result if result else None
    except Exception:
        return None


def find_match_id(match_index: list[dict], date: str, opponent: str) -> str | None:
    """從比賽索引中找到對應的 MatchID。"""
    for m in match_index:
        if m["date"] != date:
            continue
        if opponent in m["team_a"] or opponent in m["team_b"]:
            return m["match_id"]
        if m["team_a"] in opponent or m["team_b"] in opponent:
            return m["match_id"]
    return None


# ── 聯盟聚合數據 ──────────────────────────────────────────────

# 散佈圖軸可選指標對照表
AXIS_OPTIONS = {
    "總攻擊次數":          ("atk_tot",      "總攻擊次數"),
    "攻擊成功率 (ASR)":    ("asr",          "攻擊成功率 (%)"),
    "接發球總數":          ("rcv_tot",      "接發球總數"),
    "接發球到位率 (GP%)":  ("gp_pct",       "接發球到位率 (%)"),
    "防守起球總數":        ("dig_tot",      "防守起球總數"),
    "防守起球率 (DIG%)":   ("dig_pct",      "防守起球率 (%)"),
    "總防守負擔":          ("def_load",     "總防守負擔 (接發+防守)"),
    "綜合防守到位率 (DEF%)": ("def_pct",    "綜合防守到位率 (%)"),
    "總舉球次數":          ("set_tot",      "總舉球次數"),
    "舉球到位率 (SET%)":   ("set_pct",      "舉球到位率 (%)"),
    "局均攔網得分 (BLK/Set)": ("blk_per_set", "局均攔網得分"),
}
AXIS_NAMES = list(AXIS_OPTIONS.keys())

# 依位置的智慧預設 (x_index, y_index)
POS_DEFAULTS: dict[str, tuple[int, int]] = {
    "OH": (AXIS_NAMES.index("總攻擊次數"),          AXIS_NAMES.index("攻擊成功率 (ASR)")),
    "OP": (AXIS_NAMES.index("總攻擊次數"),          AXIS_NAMES.index("攻擊成功率 (ASR)")),
    "MB": (AXIS_NAMES.index("局均攔網得分 (BLK/Set)"), AXIS_NAMES.index("攻擊成功率 (ASR)")),
    "S":  (AXIS_NAMES.index("總舉球次數"),          AXIS_NAMES.index("舉球到位率 (SET%)")),
    "L":  (AXIS_NAMES.index("總防守負擔"),          AXIS_NAMES.index("綜合防守到位率 (DEF%)")),
}


@st.cache_data
def get_league_aggregated_stats(gender_code: str) -> pd.DataFrame:
    """
    撈取該組別所有球員的聚合統計數據，JOIN players + teams 取得姓名/球隊/位置。
    僅保留總局數 >= 5 的球員，排除極端值。
    """
    raw = load_data(
        """
        SELECT p.player_id,
               p.name,
               p.position,
               t.team_name,
               SUM(s.sets_played)       AS total_sets,
               SUM(s.attack_points)     AS atk_pts,
               SUM(s.attack_total)      AS atk_tot,
               SUM(s.block_points)      AS blk_pts,
               SUM(s.serve_points)      AS srv_pts,
               SUM(s.serve_total)       AS srv_tot,
               SUM(s.receive_excellent) AS rcv_exc,
               SUM(s.receive_total)     AS rcv_tot,
               SUM(s.dig_excellent)     AS dig_exc,
               SUM(s.dig_total)         AS dig_tot,
               SUM(s.set_excellent)     AS set_exc,
               SUM(s.set_total)         AS set_tot,
               SUM(s.total_points)      AS total_points,
               COUNT(*)                 AS n_games
        FROM player_match_stats s
        JOIN players p ON s.player_id = p.player_id
        JOIN teams   t ON p.team_id = t.team_id AND p.gender = t.gender
        WHERE p.gender = ?
        GROUP BY p.player_id
        HAVING SUM(s.sets_played) >= 5
        """,
        (gender_code,),
    )
    # 計算進階比率指標（向量化）
    raw["asr"] = vec_pct(raw["atk_pts"], raw["atk_tot"])
    raw["gp_pct"] = vec_pct(raw["rcv_exc"], raw["rcv_tot"])
    raw["ace_pct"] = vec_pct(raw["srv_pts"], raw["srv_tot"])
    raw["dig_pct"] = vec_pct(raw["dig_exc"], raw["dig_tot"])
    raw["ppg"] = np.where(raw["n_games"] > 0, raw["total_points"] / raw["n_games"], 0.0)
    raw["set_pct"] = vec_pct(raw["set_exc"], raw["set_tot"])
    raw["blk_per_set"] = np.where(raw["total_sets"] > 0, raw["blk_pts"] / raw["total_sets"], 0.0)
    raw["def_load"] = raw["rcv_tot"] + raw["dig_tot"]
    raw["def_pct"] = vec_pct(raw["rcv_exc"] + raw["dig_exc"], raw["rcv_tot"] + raw["dig_tot"])

    # 同位置 PR 值（百分位排名 0~100）
    pr_cols = ["asr", "gp_pct", "ace_pct", "dig_pct", "set_pct", "blk_per_set", "def_pct"]
    for col in pr_cols:
        raw[f"{col}_pr"] = (
            raw.groupby("position")[col]
            .rank(pct=True)
            .mul(100)
            .round(1)
        )

    return raw


def enrich_box_score(df: pd.DataFrame) -> pd.DataFrame:
    """為 box score DataFrame 加上單場進階指標。"""
    out = df.copy()
    out["ASR%"] = vec_pct(out["attack_points"], out["attack_total"])
    out["GP%"] = vec_pct(out["receive_excellent"], out["receive_total"])
    out["DIG%"] = vec_pct(out["dig_excellent"], out["dig_total"])
    out["SET%"] = vec_pct(out["set_excellent"], out["set_total"])
    return out
