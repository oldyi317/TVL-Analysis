"""
TVL 企業排球聯賽進階數據分析儀表板
使用 Streamlit + Plotly，基於 Proxy Metrics 呈現球員進階數據。
包含：個人深度分析 + 聯盟分佈與同位置 PR 值比較。
"""

import sqlite3
from pathlib import Path

import re as _re

# ── 在 matplotlib 匯入前清除舊字型快取，避免抓到缺少 CJK 字型的舊快取 ──
def _purge_mpl_font_cache():
    import os, glob
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "matplotlib")
    if os.path.isdir(cache_dir):
        for f in glob.glob(os.path.join(cache_dir, "fontlist-*")):
            try:
                os.remove(f)
            except OSError:
                pass
_purge_mpl_font_cache()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests as _requests
import streamlit as st
from bs4 import BeautifulSoup as _BS

st.set_page_config(
    layout="wide",
    page_title="TVL 進階數據分析儀表板",
    page_icon="🏐",
)

# ── 全域中文字型設定（matplotlib） ────────────────────────────
@st.cache_resource
def _init_matplotlib_fonts():
    """找到系統 Noto CJK 字型檔，註冊並回傳 (字型家族名稱, 字型檔路徑)。"""
    import matplotlib.font_manager as fm
    import os

    # 搜尋系統已安裝的 Noto CJK 字型檔
    font_path = None
    for root, _dirs, files in os.walk("/usr/share/fonts"):
        for f in files:
            low = f.lower()
            if "notosanscjk" in low.replace("-", "").replace("_", ""):
                font_path = os.path.join(root, f)
                break
        if font_path:
            break

    if font_path:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        return prop.get_name(), font_path
    return None, None

_CJK_FONT_NAME, _CJK_FONT_PATH = _init_matplotlib_fonts()
CJK_FONT_STACK = (
    [_CJK_FONT_NAME] if _CJK_FONT_NAME else []
) + ["Noto Sans CJK TC", "Microsoft JhengHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["font.sans-serif"] = CJK_FONT_STACK
matplotlib.rcParams["axes.unicode_minus"] = False

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "tvl_database.db"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "match_predictor.pkl"


# ── 工具函式 ──────────────────────────────────────────────────

@st.cache_data
def load_data(query: str, params: tuple = ()) -> pd.DataFrame:
    """連線 SQLite，執行查詢並回傳 DataFrame。連線使用後立即關閉。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


@st.cache_resource
def load_model_and_explainer():
    """載入 ML 模型與 SHAP Explainer，使用 cache_resource 避免重複載入。"""
    import joblib
    import shap

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    explainer = shap.TreeExplainer(model)
    return artifact, model, explainer


def safe_pct(numerator: float, denominator: float) -> float:
    """安全計算百分比，分母為 0 時回傳 0.0。"""
    return (numerator / denominator * 100) if denominator > 0 else 0.0


EXT_BASE = "http://114.35.229.141"
EXT_CUP_ID = 21
EXT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
SEASON_YEAR_MAP = {11: 2025, 12: 2025}


@st.cache_data(ttl=3600)
def fetch_match_index() -> list[dict]:
    """從外部系統抓取所有比賽場次索引（MatchID, 日期, 隊伍）。"""
    try:
        r = _requests.get(
            f"{EXT_BASE}/Match.aspx?CupID={EXT_CUP_ID}",
            headers=EXT_HEADERS, timeout=15,
        )
        soup = _BS(r.text, "html.parser")
        sel = soup.find("select", id="divSelect")
        if not sel:
            return []

        pat = _re.compile(r"[：](.+?)\s+vs\s+(.+?)\s+\((\d+)月(\d+)日\)")
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
        r = _requests.get(
            f"{EXT_BASE}/_handler/Match.ashx",
            params={"CupID": EXT_CUP_ID, "MatchID": match_id, "SetNum": 0},
            headers=EXT_HEADERS, timeout=15,
        )
        soup = _BS(r.text, "html.parser")
        tables = soup.find_all("table")
        if len(tables) < 2:
            return None

        rows = tables[1].find_all("tr")
        result = []
        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
            # [team_name, s1, s2, s3, s4, s5, total_pts, sets_won]
            if len(cells) >= 8:
                sets = cells[1:6]  # 最多 5 局
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
        # 對手簡稱可能出現在 team_a 或 team_b
        if opponent in m["team_a"] or opponent in m["team_b"]:
            return m["match_id"]
        if m["team_a"] in opponent or m["team_b"] in opponent:
            return m["match_id"]
    return None


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
    # 計算進階比率指標
    raw["asr"] = raw.apply(lambda r: safe_pct(r["atk_pts"], r["atk_tot"]), axis=1)
    raw["gp_pct"] = raw.apply(lambda r: safe_pct(r["rcv_exc"], r["rcv_tot"]), axis=1)
    raw["ace_pct"] = raw.apply(lambda r: safe_pct(r["srv_pts"], r["srv_tot"]), axis=1)
    raw["dig_pct"] = raw.apply(lambda r: safe_pct(r["dig_exc"], r["dig_tot"]), axis=1)
    raw["ppg"] = raw.apply(
        lambda r: r["total_points"] / r["n_games"] if r["n_games"] > 0 else 0, axis=1
    )
    # 舉球到位率
    raw["set_pct"] = raw.apply(lambda r: safe_pct(r["set_exc"], r["set_tot"]), axis=1)
    # 局均攔網得分
    raw["blk_per_set"] = raw.apply(
        lambda r: r["blk_pts"] / r["total_sets"] if r["total_sets"] > 0 else 0, axis=1
    )
    # 總防守負擔 = 接發總數 + 防守總數
    raw["def_load"] = raw["rcv_tot"] + raw["dig_tot"]
    # 綜合防守到位率
    raw["def_pct"] = raw.apply(
        lambda r: safe_pct(r["rcv_exc"] + r["dig_exc"], r["rcv_tot"] + r["dig_tot"]),
        axis=1,
    )

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


# 散佈圖軸可選指標對照表：顯示名稱 -> (DataFrame 欄位名, 軸標籤)
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
# 外部系統對手簡稱 -> 本地 DB (team_id, gender)
OPP_SHORT_TO_TEAM: dict[str, tuple[int, str]] = {
    "屏東台電":   (1, "M"),
    "雲林美津濃": (2, "M"),
    "臺北國北獅": (4, "M"),
    "桃園臺產":   (5, "M"),
    "獅子王":     (7, "M"),
    "高雄台電":   (4, "F"),
    "臺北鯨華":   (3, "F"),
    "新北中纖":   (5, "F"),
    "義力營造":   (7, "F"),
}

POS_DEFAULTS: dict[str, tuple[int, int]] = {
    "OH": (AXIS_NAMES.index("總攻擊次數"),          AXIS_NAMES.index("攻擊成功率 (ASR)")),
    "OP": (AXIS_NAMES.index("總攻擊次數"),          AXIS_NAMES.index("攻擊成功率 (ASR)")),
    "MB": (AXIS_NAMES.index("局均攔網得分 (BLK/Set)"), AXIS_NAMES.index("攻擊成功率 (ASR)")),
    "S":  (AXIS_NAMES.index("總舉球次數"),          AXIS_NAMES.index("舉球到位率 (SET%)")),
    "L":  (AXIS_NAMES.index("總防守負擔"),          AXIS_NAMES.index("綜合防守到位率 (DEF%)")),
}


# ── 側邊欄篩選器（三層連動） ──────────────────────────────────

st.sidebar.title("🏐 TVL 進階數據儀表板")
st.sidebar.markdown("---")

gender = st.sidebar.selectbox("選擇組別", ["男子組", "女子組"])
gender_code = "M" if gender == "男子組" else "F"

teams_df = load_data(
    "SELECT team_id, team_name FROM teams WHERE gender = ? ORDER BY team_id",
    (gender_code,),
)
if teams_df.empty:
    st.warning("該組別目前沒有球隊資料。")
    st.stop()

team_name = st.sidebar.selectbox("選擇球隊", teams_df["team_name"].tolist())
team_id = int(teams_df.loc[teams_df["team_name"] == team_name, "team_id"].iloc[0])

players_df = load_data(
    "SELECT player_id, jersey_number, name, position FROM players "
    "WHERE team_id = ? AND gender = ? ORDER BY jersey_number",
    (team_id, gender_code),
)
if players_df.empty:
    st.warning("該球隊目前沒有球員資料。")
    st.stop()

player_display = players_df.apply(
    lambda r: f"#{int(r['jersey_number'])} {r['name']}"
    if pd.notna(r["jersey_number"])
    else r["name"],
    axis=1,
).tolist()

selected_display = st.sidebar.selectbox("選擇球員", player_display)
selected_idx = player_display.index(selected_display)
player_id = int(players_df.iloc[selected_idx]["player_id"])
player_name = players_df.iloc[selected_idx]["name"]
player_position = players_df.iloc[selected_idx].get("position", None)

# ── 頁面標題 ──────────────────────────────────────────────────

pos_display = f"（{player_position}）" if player_position else ""
st.title(f"🏐 {player_name}{pos_display}　{gender}・{team_name}")
st.markdown("---")

# ── 分頁結構 ──────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "球員個人深度", "聯盟 PR 值與分佈", "逐場趨勢", "單場 Box Score", "賽果預測",
    "每周戰報",
])

# ══════════════════════════════════════════════════════════════
# Tab 1：球員個人深度分析（KPI + 雷達圖 + 逐場趨勢）
# ══════════════════════════════════════════════════════════════

with tab1:
    stats_df = load_data(
        "SELECT * FROM player_match_stats WHERE player_id = ? ORDER BY match_date",
        (player_id,),
    )

    if stats_df.empty:
        st.info("該球員目前沒有比賽數據紀錄。")
        st.stop()

    # ── 聚合計算 ──────────────────────────────────────────────
    s = stats_df
    sum_atk_pts = s["attack_points"].sum()
    sum_atk_tot = s["attack_total"].sum()
    sum_rcv_exc = s["receive_excellent"].sum()
    sum_rcv_tot = s["receive_total"].sum()
    sum_srv_pts = s["serve_points"].sum()
    sum_srv_tot = s["serve_total"].sum()
    sum_dig_exc = s["dig_excellent"].sum()
    sum_dig_tot = s["dig_total"].sum()
    sum_set_exc = s["set_excellent"].sum()
    sum_set_tot = s["set_total"].sum()
    sum_blk_pts = s["block_points"].sum()
    total_sets = int(s["sets_played"].sum())
    total_points = int(s["total_points"].sum())
    n_games = len(s)

    asr = safe_pct(sum_atk_pts, sum_atk_tot)
    gp = safe_pct(sum_rcv_exc, sum_rcv_tot)
    ace = safe_pct(sum_srv_pts, sum_srv_tot)
    dig_rate = safe_pct(sum_dig_exc, sum_dig_tot)
    set_rate = safe_pct(sum_set_exc, sum_set_tot)
    def_rate = safe_pct(sum_rcv_exc + sum_dig_exc, sum_rcv_tot + sum_dig_tot)
    blk_per_set = sum_blk_pts / total_sets if total_sets > 0 else 0
    ppg = total_points / n_games if n_games > 0 else 0

    # ── 全聯盟平均（供 KPI delta 對照） ────────────────────────
    _lg_all = load_data(
        """
        SELECT SUM(s.attack_points) AS atk_pts, SUM(s.attack_total) AS atk_tot,
               SUM(s.receive_excellent) AS rcv_exc, SUM(s.receive_total) AS rcv_tot,
               SUM(s.serve_points) AS srv_pts, SUM(s.serve_total) AS srv_tot,
               SUM(s.dig_excellent) AS dig_exc, SUM(s.dig_total) AS dig_tot,
               SUM(s.set_excellent) AS set_exc, SUM(s.set_total) AS set_tot,
               SUM(s.block_points) AS blk_pts, SUM(s.sets_played) AS tot_sets,
               SUM(s.total_points) AS tot_pts, COUNT(*) AS n_games
        FROM player_match_stats s
        JOIN players p ON s.player_id = p.player_id
        WHERE p.gender = ?
        """,
        (gender_code,),
    ).iloc[0]
    _la_asr = safe_pct(_lg_all["atk_pts"], _lg_all["atk_tot"])
    _la_gp = safe_pct(_lg_all["rcv_exc"], _lg_all["rcv_tot"])
    _la_ace = safe_pct(_lg_all["srv_pts"], _lg_all["srv_tot"])
    _la_dig = safe_pct(_lg_all["dig_exc"], _lg_all["dig_tot"])
    _la_set = safe_pct(_lg_all["set_exc"], _lg_all["set_tot"])
    _la_blk = (_lg_all["blk_pts"] / _lg_all["tot_sets"]
               if _lg_all["tot_sets"] and _lg_all["tot_sets"] > 0 else 0)
    _la_ppg = (_lg_all["tot_pts"] / _lg_all["n_games"]
               if _lg_all["n_games"] and _lg_all["n_games"] > 0 else 0)

    # ── 樣本數門檻：比率型指標分母 < 10 → N/A ─────────────────
    MIN_DENOM = 10

    def _rate_display(val: float, denom: float) -> str:
        """分母不足門檻時回傳 'N/A'，否則回傳格式化百分比。"""
        return f"{val:.1f}%" if denom >= MIN_DENOM else "N/A"

    def _rate_delta(val: float, denom: float, league_val: float) -> str | None:
        """分母不足門檻時不顯示 delta。"""
        if denom < MIN_DENOM:
            return None
        diff = val - league_val
        return f"{diff:+.1f}% vs 聯盟均"

    def _rate_safe(val: float, denom: float) -> float:
        """分母不足門檻時回傳 0（供雷達圖使用，避免極端值）。"""
        return val if denom >= MIN_DENOM else 0.0

    # ── 依位置動態 KPI 卡片（含 delta） ───────────────────────
    POS_KPI_MAP = {
        "OH": [
            ("攻擊成功率 (ASR)", _rate_display(asr, sum_atk_tot), _rate_delta(asr, sum_atk_tot, _la_asr)),
            ("接發到位率 (GP%)", _rate_display(gp, sum_rcv_tot), _rate_delta(gp, sum_rcv_tot, _la_gp)),
            ("發球破壞率 (ACE%)", _rate_display(ace, sum_srv_tot), _rate_delta(ace, sum_srv_tot, _la_ace)),
            ("防守起球率 (DIG%)", _rate_display(dig_rate, sum_dig_tot), _rate_delta(dig_rate, sum_dig_tot, _la_dig)),
        ],
        "OP": [
            ("攻擊成功率 (ASR)", _rate_display(asr, sum_atk_tot), _rate_delta(asr, sum_atk_tot, _la_asr)),
            ("局均攔網 (BLK/Set)", f"{blk_per_set:.2f}", f"{blk_per_set - _la_blk:+.2f} vs 聯盟均"),
            ("發球破壞率 (ACE%)", _rate_display(ace, sum_srv_tot), _rate_delta(ace, sum_srv_tot, _la_ace)),
            ("防守起球率 (DIG%)", _rate_display(dig_rate, sum_dig_tot), _rate_delta(dig_rate, sum_dig_tot, _la_dig)),
        ],
        "MB": [
            ("攻擊成功率 (ASR)", _rate_display(asr, sum_atk_tot), _rate_delta(asr, sum_atk_tot, _la_asr)),
            ("局均攔網 (BLK/Set)", f"{blk_per_set:.2f}", f"{blk_per_set - _la_blk:+.2f} vs 聯盟均"),
            ("發球破壞率 (ACE%)", _rate_display(ace, sum_srv_tot), _rate_delta(ace, sum_srv_tot, _la_ace)),
            ("防守起球率 (DIG%)", _rate_display(dig_rate, sum_dig_tot), _rate_delta(dig_rate, sum_dig_tot, _la_dig)),
        ],
        "S": [
            ("舉球到位率 (SET%)", _rate_display(set_rate, sum_set_tot), _rate_delta(set_rate, sum_set_tot, _la_set)),
            ("攻擊成功率 (ASR)", _rate_display(asr, sum_atk_tot), _rate_delta(asr, sum_atk_tot, _la_asr)),
            ("發球破壞率 (ACE%)", _rate_display(ace, sum_srv_tot), _rate_delta(ace, sum_srv_tot, _la_ace)),
            ("防守起球率 (DIG%)", _rate_display(dig_rate, sum_dig_tot), _rate_delta(dig_rate, sum_dig_tot, _la_dig)),
        ],
        "L": [
            ("接發到位率 (GP%)", _rate_display(gp, sum_rcv_tot), _rate_delta(gp, sum_rcv_tot, _la_gp)),
            ("防守起球率 (DIG%)", _rate_display(dig_rate, sum_dig_tot), _rate_delta(dig_rate, sum_dig_tot, _la_dig)),
            ("綜合防守到位率 (DEF%)", _rate_display(def_rate, sum_rcv_tot + sum_dig_tot), _rate_delta(def_rate, sum_rcv_tot + sum_dig_tot, safe_pct((_lg_all["rcv_exc"] or 0) + (_lg_all["dig_exc"] or 0), (_lg_all["rcv_tot"] or 0) + (_lg_all["dig_tot"] or 0)))),
            ("發球破壞率 (ACE%)", _rate_display(ace, sum_srv_tot), _rate_delta(ace, sum_srv_tot, _la_ace)),
        ],
    }
    default_kpi = [
        ("攻擊成功率 (ASR)", _rate_display(asr, sum_atk_tot), _rate_delta(asr, sum_atk_tot, _la_asr)),
        ("接發到位率 (GP%)", _rate_display(gp, sum_rcv_tot), _rate_delta(gp, sum_rcv_tot, _la_gp)),
        ("發球破壞率 (ACE%)", _rate_display(ace, sum_srv_tot), _rate_delta(ace, sum_srv_tot, _la_ace)),
        ("防守起球率 (DIG%)", _rate_display(dig_rate, sum_dig_tot), _rate_delta(dig_rate, sum_dig_tot, _la_dig)),
    ]
    kpi_items = POS_KPI_MAP.get(player_position, default_kpi)

    kpi_cols = st.columns(4)
    for col, (label, val, delta) in zip(kpi_cols, kpi_items):
        col.metric(label, val, delta=delta)

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("出賽場次", f"{n_games} 場")
    k6.metric("總出賽局數", f"{total_sets} 局")
    k7.metric("總得分", f"{total_points} 分")
    k8.metric("場均得分", f"{ppg:.1f} 分", delta=f"{ppg - _la_ppg:+.1f} 分 vs 聯盟均")

    st.markdown("---")

    # ── 組別同位置平均（供雷達圖對照） ────────────────────────
    pos_filter = "AND p.position = ?" if player_position else ""
    pos_params = (gender_code, player_position) if player_position else (gender_code,)

    league_agg = load_data(
        f"""
        SELECT SUM(s.attack_points) AS atk_pts, SUM(s.attack_total) AS atk_tot,
               SUM(s.receive_excellent) AS rcv_exc, SUM(s.receive_total) AS rcv_tot,
               SUM(s.serve_points) AS srv_pts, SUM(s.serve_total) AS srv_tot,
               SUM(s.dig_excellent) AS dig_exc, SUM(s.dig_total) AS dig_tot,
               SUM(s.set_excellent) AS set_exc, SUM(s.set_total) AS set_tot,
               SUM(s.block_points) AS blk_pts, SUM(s.sets_played) AS tot_sets,
               SUM(s.total_points) AS tot_pts, COUNT(*) AS n_games
        FROM player_match_stats s
        JOIN players p ON s.player_id = p.player_id
        WHERE p.gender = ? {pos_filter}
        """,
        pos_params,
    )
    lg = league_agg.iloc[0]
    lg_asr = safe_pct(lg["atk_pts"], lg["atk_tot"])
    lg_gp = safe_pct(lg["rcv_exc"], lg["rcv_tot"])
    lg_ace = safe_pct(lg["srv_pts"], lg["srv_tot"])
    lg_dig = safe_pct(lg["dig_exc"], lg["dig_tot"])
    lg_set = safe_pct(lg["set_exc"], lg["set_tot"])
    lg_def = safe_pct(
        (lg["rcv_exc"] or 0) + (lg["dig_exc"] or 0),
        (lg["rcv_tot"] or 0) + (lg["dig_tot"] or 0),
    )
    lg_blk = lg["blk_pts"] / lg["tot_sets"] if lg["tot_sets"] and lg["tot_sets"] > 0 else 0
    lg_ppg = lg["tot_pts"] / lg["n_games"] if lg["n_games"] and lg["n_games"] > 0 else 0

    # ── 依位置動態雷達圖維度（比率指標套用樣本數門檻） ────────
    _s_asr = _rate_safe(asr, sum_atk_tot)
    _s_gp = _rate_safe(gp, sum_rcv_tot)
    _s_ace = _rate_safe(ace, sum_srv_tot)
    _s_dig = _rate_safe(dig_rate, sum_dig_tot)
    _s_set = _rate_safe(set_rate, sum_set_tot)
    _s_def = _rate_safe(def_rate, sum_rcv_tot + sum_dig_tot)

    POS_RADAR_MAP = {
        "OH": [
            ("攻擊成功率", _s_asr, lg_asr, True),
            ("接發到位率", _s_gp, lg_gp, True),
            ("防守起球率", _s_dig, lg_dig, True),
            ("發球破壞率", _s_ace, lg_ace, True),
            ("場均得分", ppg, lg_ppg, False),
        ],
        "OP": [
            ("攻擊成功率", _s_asr, lg_asr, True),
            ("局均攔網", blk_per_set, lg_blk, False),
            ("發球破壞率", _s_ace, lg_ace, True),
            ("防守起球率", _s_dig, lg_dig, True),
            ("場均得分", ppg, lg_ppg, False),
        ],
        "MB": [
            ("攻擊成功率", _s_asr, lg_asr, True),
            ("局均攔網", blk_per_set, lg_blk, False),
            ("發球破壞率", _s_ace, lg_ace, True),
            ("防守起球率", _s_dig, lg_dig, True),
            ("場均得分", ppg, lg_ppg, False),
        ],
        "S": [
            ("舉球到位率", _s_set, lg_set, True),
            ("攻擊成功率", _s_asr, lg_asr, True),
            ("防守起球率", _s_dig, lg_dig, True),
            ("發球破壞率", _s_ace, lg_ace, True),
            ("場均得分", ppg, lg_ppg, False),
        ],
        "L": [
            ("接發到位率", _s_gp, lg_gp, True),
            ("防守起球率", _s_dig, lg_dig, True),
            ("綜合防守到位率", _s_def, lg_def, True),
            ("發球破壞率", _s_ace, lg_ace, True),
            ("場均得分", ppg, lg_ppg, False),
        ],
    }
    default_radar = [
        ("攻擊成功率", _s_asr, lg_asr, True),
        ("接發到位率", _s_gp, lg_gp, True),
        ("發球破壞率", _s_ace, lg_ace, True),
        ("防守起球率", _s_dig, lg_dig, True),
        ("場均得分", ppg, lg_ppg, False),
    ]
    radar_dims = POS_RADAR_MAP.get(player_position, default_radar)
    categories = [d[0] for d in radar_dims]
    player_vals = [d[1] for d in radar_dims]
    league_vals = [d[2] for d in radar_dims]
    is_pct = [d[3] for d in radar_dims]

    # ── 圖表區 ────────────────────────────────────────────────
    chart_left, chart_right = st.columns(2)

    # 雷達圖
    with chart_left:
        avg_label = f"{gender} 同位置平均" if player_position else f"{gender}平均"
        st.subheader("多維度戰力雷達圖")

        player_norm, league_norm = [], []
        for pv, lv in zip(player_vals, league_vals):
            hi = max(pv, lv, 0.001)
            player_norm.append(pv / hi * 100)
            league_norm.append(lv / hi * 100)

        player_hover = [
            f"{c}: {v:.1f}{'%' if p else ''}"
            for c, v, p in zip(categories, player_vals, is_pct)
        ]
        league_hover = [
            f"{c}: {v:.1f}{'%' if p else ''}"
            for c, v, p in zip(categories, league_vals, is_pct)
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=player_norm + [player_norm[0]],
            theta=categories + [categories[0]],
            fill="toself", name=player_name,
            text=player_hover + [player_hover[0]],
            hovertemplate="%{text}<extra></extra>",
            line=dict(color="#FF6B35", width=2.5),
            fillcolor="rgba(255, 107, 53, 0.25)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=league_norm + [league_norm[0]],
            theta=categories + [categories[0]],
            fill="toself", name=avg_label,
            text=league_hover + [league_hover[0]],
            hovertemplate="%{text}<extra></extra>",
            line=dict(color="#636EFA", width=1.5, dash="dash"),
            fillcolor="rgba(99, 110, 250, 0.10)",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 105], showticklabels=False)),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            height=480, margin=dict(l=60, r=60, t=40, b=60),
        )
        st.plotly_chart(fig_radar, width="stretch")

    # ── 依位置動態逐場趨勢圖 ─────────────────────────────────
    # 各位置使用最有意義的效率指標作為折線
    POS_TREND_MAP = {
        "OH": ("match_asr", "攻擊成功率 (%)", lambda r: safe_pct(r["attack_points"], r["attack_total"])),
        "OP": ("match_asr", "攻擊成功率 (%)", lambda r: safe_pct(r["attack_points"], r["attack_total"])),
        "MB": ("match_asr", "攻擊成功率 (%)", lambda r: safe_pct(r["attack_points"], r["attack_total"])),
        "S":  ("match_set_pct", "舉球到位率 (%)", lambda r: safe_pct(r["set_excellent"], r["set_total"])),
        "L":  ("match_def_pct", "綜合防守到位率 (%)",
               lambda r: safe_pct(r["receive_excellent"] + r["dig_excellent"],
                                  r["receive_total"] + r["dig_total"])),
    }
    default_trend = ("match_asr", "攻擊成功率 (%)",
                     lambda r: safe_pct(r["attack_points"], r["attack_total"]))
    trend_col, trend_label, trend_fn = POS_TREND_MAP.get(player_position, default_trend)

    # 各位置使用的長條圖量值
    POS_BAR_MAP = {
        "OH": ("total_points", "總得分"),
        "OP": ("total_points", "總得分"),
        "MB": ("block_points", "攔網得分"),
        "S":  ("set_excellent", "舉球好球"),
        "L":  ("dig_excellent", "防守好球"),
    }
    bar_col, bar_label = POS_BAR_MAP.get(player_position, ("total_points", "總得分"))

    with chart_right:
        st.subheader(f"逐場趨勢：{trend_label} + {bar_label}")

        plot_df = stats_df.copy()
        plot_df[trend_col] = plot_df.apply(trend_fn, axis=1)
        plot_df["label"] = plot_df["match_date"].str[5:] + " vs " + plot_df["opponent"]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=plot_df["label"], y=plot_df[bar_col],
            name=bar_label, marker_color="rgba(99, 110, 250, 0.45)",
            yaxis="y2", hovertemplate=f"{bar_label}: %{{y}}<extra></extra>",
        ))
        fig_trend.add_trace(go.Scatter(
            x=plot_df["label"], y=plot_df[trend_col],
            name=trend_label, mode="lines+markers",
            marker=dict(size=7, color="#FF6B35"),
            line=dict(color="#FF6B35", width=2),
            hovertemplate=f"{trend_label}: %{{y:.1f}}%<extra></extra>",
        ))
        fig_trend.update_layout(
            xaxis=dict(title="比賽", tickangle=-45),
            yaxis=dict(title=trend_label, side="left", rangemode="tozero"),
            yaxis2=dict(title=bar_label, side="right", overlaying="y", rangemode="tozero"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=480, margin=dict(l=50, r=50, t=50, b=80), bargap=0.3,
        )
        st.plotly_chart(fig_trend, width="stretch")

    # 完整比賽紀錄
    with st.expander("📊 完整比賽紀錄", expanded=False):
        display_cols = {
            "match_date": "日期", "opponent": "對手", "sets_played": "局數",
            "attack_points": "攻擊得分", "attack_total": "攻擊總數",
            "block_points": "攔網得分",
            "serve_points": "發球得分", "serve_total": "發球總數",
            "receive_excellent": "接發好球", "receive_total": "接發總數",
            "dig_excellent": "防守好球", "dig_total": "防守總數",
            "set_excellent": "舉球好球", "set_total": "舉球總數",
            "total_points": "總得分",
        }
        st.dataframe(
            stats_df[list(display_cols.keys())].rename(columns=display_cols),
            width="stretch", hide_index=True,
        )

# ══════════════════════════════════════════════════════════════
# Tab 2：聯盟分佈與 PR 值比較
# ══════════════════════════════════════════════════════════════

with tab2:
    league_all = get_league_aggregated_stats(gender_code)

    if league_all.empty:
        st.info("目前無足夠的聯盟數據。")
        st.stop()

    # ── 所選球員的 PR 值卡片 ──────────────────────────────────
    me = league_all[league_all["player_id"] == player_id]

    if not me.empty:
        me_row = me.iloc[0]
        pos_label = me_row["position"] if me_row["position"] else "未知"
        st.subheader(f"{player_name} 的同位置 PR 值（位置：{pos_label}）")

        # 依位置顯示有意義的 PR 指標（自由球員不攔網/舉球，舉球員不接發）
        # (顯示名稱, PR 欄位, 分母欄位 or None)
        POS_PR_ITEMS = {
            "OH": [
                ("攻擊成功率", "asr_pr", "atk_tot"),
                ("接發到位率", "gp_pct_pr", "rcv_tot"),
                ("防守起球率", "dig_pct_pr", "dig_tot"),
                ("發球破壞率", "ace_pct_pr", "srv_tot"),
                ("局均攔網", "blk_per_set_pr", None),
                ("綜合防守", "def_pct_pr", None),
            ],
            "OP": [
                ("攻擊成功率", "asr_pr", "atk_tot"),
                ("局均攔網", "blk_per_set_pr", None),
                ("防守起球率", "dig_pct_pr", "dig_tot"),
                ("發球破壞率", "ace_pct_pr", "srv_tot"),
                ("綜合防守", "def_pct_pr", None),
            ],
            "MB": [
                ("攻擊成功率", "asr_pr", "atk_tot"),
                ("局均攔網", "blk_per_set_pr", None),
                ("防守起球率", "dig_pct_pr", "dig_tot"),
                ("發球破壞率", "ace_pct_pr", "srv_tot"),
                ("綜合防守", "def_pct_pr", None),
            ],
            "S": [
                ("舉球到位率", "set_pct_pr", "set_tot"),
                ("攻擊成功率", "asr_pr", "atk_tot"),
                ("防守起球率", "dig_pct_pr", "dig_tot"),
                ("發球破壞率", "ace_pct_pr", "srv_tot"),
                ("綜合防守", "def_pct_pr", None),
            ],
            "L": [
                ("接發到位率", "gp_pct_pr", "rcv_tot"),
                ("防守起球率", "dig_pct_pr", "dig_tot"),
                ("綜合防守", "def_pct_pr", None),
            ],
        }
        default_pr_items = [
            ("攻擊成功率", "asr_pr", "atk_tot"),
            ("接發到位率", "gp_pct_pr", "rcv_tot"),
            ("防守起球率", "dig_pct_pr", "dig_tot"),
            ("舉球到位率", "set_pct_pr", "set_tot"),
            ("局均攔網", "blk_per_set_pr", None),
            ("綜合防守", "def_pct_pr", None),
        ]
        pr_items = POS_PR_ITEMS.get(pos_label, default_pr_items)
        pr_cols = st.columns(len(pr_items))
        for col, (label, key, denom_key) in zip(pr_cols, pr_items):
            val = me_row.get(key, 0)
            # 比率型指標：分母 < 10 時顯示 N/A
            if denom_key and me_row.get(denom_key, 0) < 10:
                col.metric(f"{label} PR", "N/A",
                           help=f"樣本數不足（{denom_key} < 10），不計入排名")
            else:
                col.metric(f"{label} PR", f"{val:.0f}",
                           help=f"在所有 {pos_label} 中的百分位排名（0~100）")
        st.markdown("---")

    # ── 位置篩選 ──────────────────────────────────────────────
    positions = sorted(league_all["position"].dropna().unique().tolist())
    if not positions:
        st.warning("球員位置資料不足，無法繪製散佈圖。")
        st.stop()

    # 預設選到當前球員的位置
    me_pos = me.iloc[0]["position"] if not me.empty and me.iloc[0]["position"] in positions else None
    default_pos_idx = positions.index(me_pos) if me_pos else 0

    selected_pos = st.selectbox("選擇位置進行分析", positions, index=default_pos_idx)
    pos_df = league_all[league_all["position"] == selected_pos].copy()

    if len(pos_df) < 2:
        st.info(f"位置 {selected_pos} 僅有 {len(pos_df)} 位球員，資料不足。")
        st.stop()

    # ── 動態 X/Y 軸選擇（智慧預設） ──────────────────────────
    def_x_idx, def_y_idx = POS_DEFAULTS.get(selected_pos, (0, 1))

    ax_left, ax_right = st.columns(2)
    with ax_left:
        x_choice = st.selectbox("X 軸指標", AXIS_NAMES, index=def_x_idx)
    with ax_right:
        y_choice = st.selectbox("Y 軸指標", AXIS_NAMES, index=def_y_idx)

    x_col, x_label = AXIS_OPTIONS[x_choice]
    y_col, y_label = AXIS_OPTIONS[y_choice]

    # ── 火力與效率象限散佈圖 ──────────────────────────────────
    st.subheader(f"🔥 {selected_pos} 象限分析：{x_choice} vs {y_choice}")

    pos_df["is_selected"] = pos_df["player_id"] == player_id

    med_x = pos_df[x_col].median()
    med_y = pos_df[y_col].median()

    # 動態 hover：顯示該位置最相關的指標
    hover_data_cfg = {
        x_col: ":.1f", y_col: ":.1f",
        "total_points": ":.0f", "n_games": True,
        "team_name": True, "is_selected": False,
    }

    fig_scatter = px.scatter(
        pos_df,
        x=x_col, y=y_col,
        size="total_points",
        color="team_name",
        hover_name="name",
        hover_data=hover_data_cfg,
        labels={
            x_col: x_label, y_col: y_label,
            "total_points": "總得分", "team_name": "球隊",
            "n_games": "出賽場次",
        },
        size_max=35,
    )

    # 中位數虛線
    fig_scatter.add_hline(
        y=med_y, line_dash="dash", line_color="gray", line_width=1,
        annotation_text=f"中位數 {med_y:.1f}",
        annotation_position="top left", annotation_font_size=11,
    )
    fig_scatter.add_vline(
        x=med_x, line_dash="dash", line_color="gray", line_width=1,
        annotation_text=f"中位數 {med_x:.1f}",
        annotation_position="top right", annotation_font_size=11,
    )

    # 象限標籤
    x_range = pos_df[x_col].max() - pos_df[x_col].min()
    y_range = pos_df[y_col].max() - pos_df[y_col].min()
    if x_range > 0 and y_range > 0:
        quadrant_labels = [
            ("低量高質", med_x - x_range * 0.3, med_y + y_range * 0.3, "#2ECC71"),
            ("★ 頂尖球員", med_x + x_range * 0.3, med_y + y_range * 0.3, "#E74C3C"),
            ("低量低質", med_x - x_range * 0.3, med_y - y_range * 0.3, "#95A5A6"),
            ("高量低質", med_x + x_range * 0.3, med_y - y_range * 0.3, "#F39C12"),
        ]
        for text, qx, qy, color in quadrant_labels:
            fig_scatter.add_annotation(
                x=qx, y=qy, text=text, showarrow=False,
                font=dict(size=11, color=color), opacity=0.6,
            )

    # 標記所選球員
    if not me.empty and me.iloc[0]["position"] == selected_pos:
        me_r = me.iloc[0]
        fig_scatter.add_annotation(
            x=me_r[x_col], y=me_r[y_col],
            text=f"　★ {player_name}", showarrow=False,
            font=dict(size=13, color="#FF6B35", family="Arial Black"),
            xanchor="left",
        )

    fig_scatter.update_layout(
        height=560,
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(title="球隊"),
    )
    st.plotly_chart(fig_scatter, width="stretch")

    # ── 同位置排行表 ─────────────────────────────────────────
    with st.expander(f"📋 {selected_pos} 完整排行數據", expanded=False):
        rank_cols = {
            "name": "球員", "team_name": "球隊",
            "n_games": "出賽", "total_sets": "局數",
            "atk_tot": "攻擊總數", "atk_pts": "攻擊得分",
            "asr": "ASR%", "asr_pr": "ASR PR",
            "rcv_tot": "接發總數", "rcv_exc": "接發好球",
            "gp_pct": "GP%", "gp_pct_pr": "GP PR",
            "dig_tot": "防守總數", "dig_exc": "防守好球",
            "dig_pct": "DIG%", "dig_pct_pr": "DIG PR",
            "set_tot": "舉球總數", "set_exc": "舉球好球",
            "set_pct": "SET%", "set_pct_pr": "SET PR",
            "blk_pts": "攔網得分", "blk_per_set": "BLK/Set",
            "blk_per_set_pr": "BLK PR",
            "def_load": "防守負擔", "def_pct": "DEF%",
            "def_pct_pr": "DEF PR",
            "total_points": "總得分", "ppg": "場均得分",
        }
        available = {k: v for k, v in rank_cols.items() if k in pos_df.columns}
        show_df = (
            pos_df[list(available.keys())]
            .rename(columns=available)
            .sort_values("總得分", ascending=False)
        )
        st.dataframe(show_df, width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════
# Tab 3：逐場賽事明細與對戰分析
# ══════════════════════════════════════════════════════════════

with tab3:
    match_df = load_data(
        "SELECT * FROM player_match_stats WHERE player_id = ? ORDER BY match_date",
        (player_id,),
    )

    if match_df.empty:
        st.info("該球員目前沒有比賽數據紀錄。")
        st.stop()

    # ── 計算逐場進階指標 ──────────────────────────────────────
    md = match_df.copy()
    md["單場ASR"] = md.apply(
        lambda r: safe_pct(r["attack_points"], r["attack_total"]), axis=1
    )
    md["單場GP%"] = md.apply(
        lambda r: safe_pct(r["receive_excellent"], r["receive_total"]), axis=1
    )
    md["單場DIG%"] = md.apply(
        lambda r: safe_pct(r["dig_excellent"], r["dig_total"]), axis=1
    )
    md["單場SET%"] = md.apply(
        lambda r: safe_pct(r["set_excellent"], r["set_total"]), axis=1
    )
    md["單場DEF%"] = md.apply(
        lambda r: safe_pct(
            r["receive_excellent"] + r["dig_excellent"],
            r["receive_total"] + r["dig_total"],
        ),
        axis=1,
    )

    # ── 熱力資料表 ────────────────────────────────────────────
    st.subheader("📋 逐場數據明細（條件格式化）")

    heat_cols = {
        "match_date": "日期",
        "opponent": "對手",
        "sets_played": "局數",
        "attack_total": "攻擊次數",
        "attack_points": "攻擊得分",
        "單場ASR": "ASR%",
        "block_points": "攔網得分",
        "serve_total": "發球次數",
        "serve_points": "發球得分",
        "receive_total": "接發總數",
        "receive_excellent": "接發好球",
        "單場GP%": "GP%",
        "dig_total": "防守總數",
        "dig_excellent": "防守好球",
        "單場DIG%": "DIG%",
        "set_total": "舉球總數",
        "set_excellent": "舉球好球",
        "單場SET%": "SET%",
        "total_points": "總得分",
    }
    heat_df = md[list(heat_cols.keys())].rename(columns=heat_cols)

    # 條件格式化：對數值欄位套用漸層色彩
    gradient_cols = ["總得分", "攻擊得分", "ASR%", "攔網得分", "發球得分",
                     "GP%", "DIG%", "SET%"]
    # 只對存在且有變異的欄位上色
    valid_gradient = [
        c for c in gradient_cols
        if c in heat_df.columns and heat_df[c].nunique() > 1
    ]

    styled = heat_df.style.format(
        {c: "{:.1f}" for c in ["ASR%", "GP%", "DIG%", "SET%"] if c in heat_df.columns}
    )
    if valid_gradient:
        styled = styled.background_gradient(
            cmap="YlGnBu", subset=valid_gradient
        )

    st.dataframe(styled, width="stretch", hide_index=True, height=450)

    st.download_button(
        label="📥 下載逐場數據 CSV",
        data=heat_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{player_name}_逐場數據.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ── 對戰對手分佈圖 ───────────────────────────────────────
    st.subheader("🥊 對戰對手績效分佈")

    # 依位置提供不同的預設指標選項
    metric_options = {
        "單場總得分": "total_points",
        "單場攻擊成功率 (ASR%)": "單場ASR",
        "單場接發到位率 (GP%)": "單場GP%",
        "單場防守起球率 (DIG%)": "單場DIG%",
        "單場舉球到位率 (SET%)": "單場SET%",
        "單場綜合防守到位率 (DEF%)": "單場DEF%",
        "單場攻擊得分": "attack_points",
        "單場攔網得分": "block_points",
        "單場發球得分": "serve_points",
    }
    # 依位置設定預設選項
    POS_DEFAULT_METRIC = {
        "OH": "單場總得分", "OP": "單場總得分",
        "MB": "單場攔網得分", "S": "單場舉球到位率 (SET%)",
        "L": "單場綜合防守到位率 (DEF%)",
    }
    default_metric = POS_DEFAULT_METRIC.get(player_position, "單場總得分")
    metric_names = list(metric_options.keys())
    default_idx = metric_names.index(default_metric) if default_metric in metric_names else 0

    selected_metric = st.selectbox("選擇分析指標", metric_names, index=default_idx)
    metric_col = metric_options[selected_metric]

    # 按對手出場次數排序
    opp_order = (
        md.groupby("opponent").size()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig_box = px.box(
        md,
        x="opponent",
        y=metric_col,
        points="all",
        color="opponent",
        category_orders={"opponent": opp_order},
        labels={"opponent": "對手", metric_col: selected_metric},
    )
    fig_box.update_layout(
        showlegend=False,
        height=480,
        margin=dict(l=50, r=50, t=30, b=50),
        xaxis_title="對手",
        yaxis_title=selected_metric,
    )
    # hover 加上日期資訊
    fig_box.update_traces(
        hovertemplate=(
            f"對手: %{{x}}<br>{selected_metric}: %{{y:.1f}}<extra></extra>"
        ),
    )
    st.plotly_chart(fig_box, width="stretch")

    # ── 對手績效摘要表 ───────────────────────────────────────
    with st.expander("📊 對戰對手績效摘要", expanded=False):
        opp_summary = (
            md.groupby("opponent")
            .agg(
                出賽場次=("sets_played", "count"),
                總得分=("total_points", "sum"),
                場均得分=("total_points", "mean"),
                攻擊得分=("attack_points", "sum"),
                平均ASR=("單場ASR", "mean"),
                攔網得分=("block_points", "sum"),
                平均GP=("單場GP%", "mean"),
                平均DIG=("單場DIG%", "mean"),
            )
            .round(1)
            .sort_values("總得分", ascending=False)
            .rename_axis("對手")
            .reset_index()
        )
        opp_summary.columns = [
            "對手", "出賽場次", "總得分", "場均得分",
            "攻擊得分", "平均ASR%", "攔網得分", "平均GP%", "平均DIG%",
        ]
        st.dataframe(opp_summary, width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════
# Tab 4：單場 Box Score 與對戰比較
# ══════════════════════════════════════════════════════════════

with tab4:

    # ── 篩選器：性別 → 球隊 → 比賽場次 ──────────────────────
    f1, f2, f3 = st.columns(3)

    with f1:
        bs_gender = st.selectbox("組別", ["男子組", "女子組"], key="bs_gender")
    bs_gender_code = "M" if bs_gender == "男子組" else "F"

    bs_teams = load_data(
        "SELECT team_id, team_name FROM teams WHERE gender = ? ORDER BY team_id",
        (bs_gender_code,),
    )
    if bs_teams.empty:
        st.warning("該組別無球隊資料。")
        st.stop()

    with f2:
        bs_team_name = st.selectbox("選擇球隊 (Team A)", bs_teams["team_name"].tolist(),
                                    key="bs_team")
    bs_team_id = int(
        bs_teams.loc[bs_teams["team_name"] == bs_team_name, "team_id"].iloc[0]
    )

    # 找出該隊所有比賽場次
    matches_df = load_data(
        """
        SELECT DISTINCT s.match_date, s.opponent
        FROM player_match_stats s
        JOIN players p ON s.player_id = p.player_id
        WHERE p.team_id = ? AND p.gender = ?
        ORDER BY s.match_date
        """,
        (bs_team_id, bs_gender_code),
    )
    if matches_df.empty:
        st.info("該球隊尚無比賽紀錄。")
        st.stop()

    match_labels = (
        matches_df["match_date"] + " vs " + matches_df["opponent"]
    ).tolist()

    with f3:
        selected_match = st.selectbox("選擇比賽", match_labels, key="bs_match")

    sel_idx = match_labels.index(selected_match)
    sel_date = matches_df.iloc[sel_idx]["match_date"]
    sel_opponent = matches_df.iloc[sel_idx]["opponent"]

    st.markdown("---")

    # ── 賽果比分卡 ────────────────────────────────────────────
    match_index = fetch_match_index()
    mid = find_match_id(match_index, sel_date, sel_opponent)

    if mid:
        scores = fetch_set_scores(mid)
        if scores and len(scores) == 2:
            t_a, t_b = scores[0], scores[1]
            # 判斷局數（過濾 None / 0 的空局）
            played_sets = sum(1 for s in t_a["sets"] if s is not None and s > 0)

            sc1, sc2, sc3 = st.columns([2, 3, 2])
            with sc1:
                st.markdown(
                    f"<h2 style='text-align:right; margin:0;'>{t_a['team']}</h2>",
                    unsafe_allow_html=True,
                )
            with sc2:
                set_strs_a = "　".join(
                    str(s) for s in t_a["sets"][:played_sets] if s is not None
                )
                set_strs_b = "　".join(
                    str(s) for s in t_b["sets"][:played_sets] if s is not None
                )
                winner = "🏆" if t_a["sets_won"] > t_b["sets_won"] else ""
                loser = "🏆" if t_b["sets_won"] > t_a["sets_won"] else ""
                st.markdown(
                    f"""
                    <div style='text-align:center; font-size:1.1em; line-height:2;'>
                        <b style='font-size:2em;'>{t_a['sets_won']}</b>
                        <span style='font-size:1.5em; color:gray;'> : </span>
                        <b style='font-size:2em;'>{t_b['sets_won']}</b>
                        <br>
                        <span style='color:gray; font-size:0.85em;'>
                            {set_strs_a}<br>{set_strs_b}
                        </span>
                        <br>
                        <span style='color:gray; font-size:0.8em;'>
                            總分 {t_a['total_pts']} : {t_b['total_pts']}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with sc3:
                st.markdown(
                    f"<h2 style='text-align:left; margin:0;'>{t_b['team']}</h2>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")

    # ── 撈取 Team A 單場數據 ─────────────────────────────────
    team_a_df = load_data(
        """
        SELECT p.name, p.position, s.sets_played,
               s.attack_points, s.attack_total,
               s.block_points,
               s.serve_points, s.serve_total,
               s.receive_excellent, s.receive_total,
               s.dig_excellent, s.dig_total,
               s.set_excellent, s.set_total,
               s.total_points
        FROM player_match_stats s
        JOIN players p ON s.player_id = p.player_id
        WHERE p.team_id = ? AND p.gender = ?
          AND s.match_date = ? AND s.opponent = ?
        ORDER BY s.total_points DESC
        """,
        (bs_team_id, bs_gender_code, sel_date, sel_opponent),
    )

    # ── 撈取 Team B 單場數據 ─────────────────────────────────
    opp_info = OPP_SHORT_TO_TEAM.get(sel_opponent)
    team_b_df = pd.DataFrame()

    if opp_info:
        opp_team_id, opp_gender = opp_info
        # Team B 的對手名稱：需找出 Team A 在對手眼中的簡稱
        # 方式：直接查 Team B 在同日的紀錄
        team_b_df = load_data(
            """
            SELECT p.name, p.position, s.sets_played,
                   s.attack_points, s.attack_total,
                   s.block_points,
                   s.serve_points, s.serve_total,
                   s.receive_excellent, s.receive_total,
                   s.dig_excellent, s.dig_total,
                   s.set_excellent, s.set_total,
                   s.total_points
            FROM player_match_stats s
            JOIN players p ON s.player_id = p.player_id
            WHERE p.team_id = ? AND p.gender = ?
              AND s.match_date = ?
            ORDER BY s.total_points DESC
            """,
            (opp_team_id, opp_gender, sel_date),
        )

    # ── 計算進階指標的輔助函式 ────────────────────────────────
    def enrich_box_score(df: pd.DataFrame) -> pd.DataFrame:
        """為 box score DataFrame 加上單場進階指標。"""
        out = df.copy()
        out["ASR%"] = out.apply(
            lambda r: safe_pct(r["attack_points"], r["attack_total"]), axis=1
        )
        out["GP%"] = out.apply(
            lambda r: safe_pct(r["receive_excellent"], r["receive_total"]), axis=1
        )
        out["DIG%"] = out.apply(
            lambda r: safe_pct(r["dig_excellent"], r["dig_total"]), axis=1
        )
        out["SET%"] = out.apply(
            lambda r: safe_pct(r["set_excellent"], r["set_total"]), axis=1
        )
        return out

    def format_box_score(df: pd.DataFrame) -> pd.DataFrame:
        """將 box score 整理為顯示用的 DataFrame。"""
        cols = {
            "name": "姓名", "position": "位置", "sets_played": "局數",
            "total_points": "總得分",
            "attack_points": "攻擊得", "attack_total": "攻擊總", "ASR%": "ASR%",
            "block_points": "攔網得",
            "serve_points": "發球得", "serve_total": "發球總",
            "receive_excellent": "接發好", "receive_total": "接發總", "GP%": "GP%",
            "dig_excellent": "防守好", "dig_total": "防守總", "DIG%": "DIG%",
        }
        return df[[c for c in cols if c in df.columns]].rename(
            columns=cols
        )

    def style_box_score(df: pd.DataFrame):
        """套用條件格式化。"""
        pct_cols = [c for c in ["ASR%", "GP%", "DIG%"] if c in df.columns]
        styled = df.style.format(
            {c: "{:.1f}" for c in pct_cols}
        )
        grad_cols = [c for c in ["總得分", "ASR%"] if c in df.columns and df[c].nunique() > 1]
        if grad_cols:
            styled = styled.background_gradient(cmap="Blues", subset=grad_cols)
        return styled

    # ── 雙方 Box Score 並排 ───────────────────────────────────
    st.subheader(f"📊 {sel_date}　{bs_team_name} vs {sel_opponent}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"**{bs_team_name}**")
        if team_a_df.empty:
            st.info("Team A 無該場數據。")
        else:
            a_enriched = enrich_box_score(team_a_df)
            a_display = format_box_score(a_enriched)
            st.dataframe(
                style_box_score(a_display),
                width="stretch", hide_index=True, height=450,
            )
            a_total = int(team_a_df["total_points"].sum())
            st.caption(f"團隊總得分：**{a_total}**")
            st.download_button(
                label="📥 下載 CSV",
                data=a_display.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{bs_team_name}_{sel_date}_box_score.csv",
                mime="text/csv",
                key="dl_team_a",
            )

    with col_b:
        st.markdown(f"**{sel_opponent}**")
        if team_b_df.empty:
            st.info("對手數據無法取得（可能為跨組別或資料缺失）。")
        else:
            b_enriched = enrich_box_score(team_b_df)
            b_display = format_box_score(b_enriched)
            st.dataframe(
                style_box_score(b_display),
                width="stretch", hide_index=True, height=450,
            )
            b_total = int(team_b_df["total_points"].sum())
            st.caption(f"團隊總得分：**{b_total}**")
            st.download_button(
                label="📥 下載 CSV",
                data=b_display.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{sel_opponent}_{sel_date}_box_score.csv",
                mime="text/csv",
                key="dl_team_b",
            )

    st.markdown("---")

    # ── 雙方排行榜比較 ─────────────────────────────────────────
    # 可選排行指標
    RANK_METRICS = {
        "總得分":       ("total_points", False),
        "攻擊得分":     ("attack_points", False),
        "攻擊成功率 (ASR%)": ("ASR%", True),
        "攔網得分":     ("block_points", False),
        "發球得分":     ("serve_points", False),
        "接發好球":     ("receive_excellent", False),
        "接發到位率 (GP%)": ("GP%", True),
        "防守好球":     ("dig_excellent", False),
        "防守起球率 (DIG%)": ("DIG%", True),
        "舉球好球":     ("set_excellent", False),
        "舉球到位率 (SET%)": ("SET%", True),
    }

    has_both = not team_a_df.empty and not team_b_df.empty
    has_any = not team_a_df.empty or not team_b_df.empty

    if has_any:
        rank_label = st.selectbox(
            "選擇排行指標", list(RANK_METRICS.keys()), key="bs_rank_metric"
        )
        rank_col, is_pct_metric = RANK_METRICS[rank_label]

        # 準備數據
        parts = []
        if not team_a_df.empty:
            a_e = enrich_box_score(team_a_df).copy()
            a_e["team"] = bs_team_name
            parts.append(a_e)
        if not team_b_df.empty:
            b_e = enrich_box_score(team_b_df).copy()
            b_e["team"] = sel_opponent
            parts.append(b_e)
        combined = pd.concat(parts, ignore_index=True)

        # 比率指標：過濾分母為 0 的球員（避免 0/0 = 0% 混入排行）
        if is_pct_metric:
            denom_map = {
                "ASR%": "attack_total", "GP%": "receive_total",
                "DIG%": "dig_total", "SET%": "set_total",
            }
            denom_col = denom_map.get(rank_col)
            if denom_col and denom_col in combined.columns:
                combined = combined[combined[denom_col] > 0]

        if combined.empty:
            st.info("所選指標無有效數據。")
        else:
            top_n = combined.nlargest(10, rank_col)
            top_n = top_n.sort_values(rank_col, ascending=True)

            title_suffix = "（雙方 Top 10）" if has_both else f"（{parts[0].iloc[0]['team']}）"
            st.subheader(f"🏆 {rank_label}排行 {title_suffix}")

            fig_rank = go.Figure()
            team_colors = {bs_team_name: "#636EFA", sel_opponent: "#EF553B"}

            for team_label in top_n["team"].unique():
                sub = top_n[top_n["team"] == team_label]
                color = team_colors.get(team_label, "#636EFA")

                fmt = ".1f%" if is_pct_metric else ".0f"
                text_vals = sub[rank_col].apply(
                    lambda v: f"{v:.1f}%" if is_pct_metric else f"{int(v)}"
                )

                fig_rank.add_trace(go.Bar(
                    y=sub["name"],
                    x=sub[rank_col],
                    name=team_label,
                    orientation="h",
                    marker_color=color,
                    text=text_vals,
                    textposition="auto",
                    hovertemplate=(
                        "%{y}<br>"
                        f"{rank_label}: %{{x:.1f}}<br>"
                        "總得分: %{customdata[0]}<br>"
                        "攻擊: %{customdata[1]}/%{customdata[2]} (ASR %{customdata[3]:.1f}%)"
                        "<extra></extra>"
                    ),
                    customdata=sub[
                        ["total_points", "attack_points", "attack_total", "ASR%"]
                    ].values,
                ))

            fig_rank.update_layout(
                barmode="group",
                height=max(350, len(top_n) * 35 + 100),
                margin=dict(l=100, r=50, t=30, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5),
                xaxis_title=rank_label,
                yaxis_title="",
            )
            st.plotly_chart(fig_rank, width="stretch")

# ══════════════════════════════════════════════════════════════
# Tab 5：賽果預測 (ML Match Prediction)
# ══════════════════════════════════════════════════════════════

with tab5:
    # ── 檢查模型檔案是否存在 ──────────────────────────────────
    if not MODEL_PATH.exists():
        st.warning(
            "找不到預測模型檔案。請先執行 "
            "`notebooks/02_ml_match_prediction.ipynb` 產生 "
            "`src/models/match_predictor.pkl`。"
        )
        st.stop()

    artifact, model, explainer = load_model_and_explainer()
    model_feature_cols = artifact.get("feature_cols", [])
    model_name = artifact.get("model_name", "XGBoost")
    model_version = artifact.get("version", "v1")

    st.subheader("ML 賽果預測模擬器")
    st.caption(f"模型：{model_name}　|　特徵數：{len(model_feature_cols)}")

    # ── 根據模型版本動態建立滑桿 ──────────────────────────────
    # v1 模型：5 個單場效率指標
    # v2 模型：11 個滾動歷史特徵
    V1_SLIDER_CFG = [
        ("ASR",         "攻擊成功率 (ASR %)",     30.0, 65.0, 42.0, 0.5),
        ("GP_pct",      "接發到位率 (GP %)",       20.0, 80.0, 50.0, 0.5),
        ("DIG_pct",     "防守起球率 (DIG %)",      10.0, 75.0, 32.0, 0.5),
        ("BLK_per_set", "局均攔網 (BLK/Set)",       0.0,  5.0,  1.8, 0.1),
        ("ACE_pct",     "發球破壞率 (ACE %)",       0.0, 18.0,  4.0, 0.5),
    ]

    V2_SLIDER_CFG = [
        ("ASR_roll3",         "近3場 攻擊率 (%)",       25.0, 65.0, 42.0, 0.5),
        ("ASR_roll5",         "近5場 攻擊率 (%)",       25.0, 65.0, 42.0, 0.5),
        ("GP_pct_roll3",      "近3場 接發率 (%)",       15.0, 85.0, 50.0, 0.5),
        ("GP_pct_roll5",      "近5場 接發率 (%)",       15.0, 85.0, 50.0, 0.5),
        ("DIG_pct_roll3",     "近3場 防守率 (%)",        5.0, 75.0, 32.0, 0.5),
        ("DIG_pct_roll5",     "近5場 防守率 (%)",        5.0, 75.0, 32.0, 0.5),
        ("BLK_per_set_roll3", "近3場 局均攔網",          0.0,  5.0,  1.8, 0.1),
        ("BLK_per_set_roll5", "近5場 局均攔網",          0.0,  5.0,  1.8, 0.1),
        ("ACE_pct_roll3",     "近3場 發球率 (%)",        0.0, 18.0,  4.0, 0.5),
        ("ACE_pct_roll5",     "近5場 發球率 (%)",        0.0, 18.0,  4.0, 0.5),
        ("win_streak",        "連勝/連敗 (正=連勝)",    -8.0,  8.0,  0.0, 1.0),
    ]

    # 依模型特徵數自動選擇 slider 配置
    slider_cfg_map = {5: V1_SLIDER_CFG, 11: V2_SLIDER_CFG}
    slider_cfg = slider_cfg_map.get(len(model_feature_cols))

    if slider_cfg is None:
        st.error(f"模型特徵數 ({len(model_feature_cols)}) 不在預期範圍內。")
        st.stop()

    # ── 建立滑桿 UI ──────────────────────────────────────────
    st.markdown("#### 調整球隊數據指標")

    # 分兩欄排列滑桿，避免太長
    slider_values = {}
    n_cols = 2
    rows = [slider_cfg[i:i + n_cols] for i in range(0, len(slider_cfg), n_cols)]

    for row_items in rows:
        cols = st.columns(n_cols)
        for col, item in zip(cols, row_items):
            key, label, mn, mx, default, step = item
            slider_values[key] = col.slider(
                label, min_value=mn, max_value=mx,
                value=default, step=step, key=f"pred_{key}",
            )

    # ── 組裝特徵向量並預測 ────────────────────────────────────
    feature_vector = [slider_values[col] for col in model_feature_cols]
    X_input = np.array([feature_vector])

    proba = model.predict_proba(X_input)[0]
    win_prob = proba[1]
    lose_prob = proba[0]

    # ── 勝率顯示區 ──────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3 = st.columns([1, 2, 1])

    with m2:
        # 勝率色彩：> 60% 綠色、< 40% 紅色、中間橘色
        if win_prob >= 0.6:
            color = "#2ECC71"
            verdict = "預測：勝"
        elif win_prob <= 0.4:
            color = "#E74C3C"
            verdict = "預測：敗"
        else:
            color = "#F39C12"
            verdict = "預測：五五波"

        st.markdown(
            f"""
            <div style="text-align:center; padding: 1.5em 0;">
                <p style="font-size: 1.1em; color: gray; margin-bottom: 0.2em;">
                    模型預測勝率
                </p>
                <p style="font-size: 4em; font-weight: bold; color: {color};
                          margin: 0; line-height: 1.1;">
                    {win_prob:.1%}
                </p>
                <p style="font-size: 1.3em; color: {color};">
                    {verdict}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── SHAP 戰術診斷 ────────────────────────────────────────
    if st.button("產生戰術診斷圖 (SHAP Waterfall)", type="primary"):
        import shap as _shap

        # 特徵中文名稱
        FEAT_LABELS_MAP = {
            "ASR": "攻擊成功率", "GP_pct": "接發到位率",
            "DIG_pct": "防守起球率", "BLK_per_set": "局均攔網",
            "ACE_pct": "發球破壞率",
            "ASR_roll3": "近3場攻擊率", "ASR_roll5": "近5場攻擊率",
            "GP_pct_roll3": "近3場接發率", "GP_pct_roll5": "近5場接發率",
            "DIG_pct_roll3": "近3場防守率", "DIG_pct_roll5": "近5場防守率",
            "BLK_per_set_roll3": "近3場局均攔網", "BLK_per_set_roll5": "近5場局均攔網",
            "ACE_pct_roll3": "近3場發球率", "ACE_pct_roll5": "近5場發球率",
            "win_streak": "連勝/連敗",
        }
        cn_labels = [FEAT_LABELS_MAP.get(c, c) for c in model_feature_cols]

        # 計算 SHAP values
        shap_values = explainer.shap_values(X_input)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # 正類 (勝) 的 SHAP
        else:
            sv = shap_values[0]

        # 建立 SHAP Explanation 物件
        explanation = _shap.Explanation(
            values=sv,
            base_values=(
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            ),
            data=X_input[0],
            feature_names=cn_labels,
        )

        # 繪製 Waterfall Plot
        _shap.plots.waterfall(explanation, show=False)
        fig_shap = plt.gcf()  # SHAP 內部建立的 figure
        fig_shap.set_size_inches(10, max(4, len(cn_labels) * 0.5 + 1))

        # 強制所有文字元素使用中文字型（SHAP 內部不走 rcParams）
        from matplotlib.font_manager import FontProperties
        _cjk_fp = (FontProperties(fname=_CJK_FONT_PATH)
                    if _CJK_FONT_PATH else FontProperties(family=CJK_FONT_STACK))
        for text_obj in fig_shap.findobj(matplotlib.text.Text):
            text_obj.set_fontproperties(_cjk_fp)

        plt.title("SHAP 戰術診斷：各指標對勝率的影響", fontsize=14,
                  fontproperties=_cjk_fp)
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close(fig_shap)

        # 文字摘要：正/負貢獻排名
        sorted_idx = np.argsort(np.abs(sv))[::-1]
        st.markdown("##### 影響力排名")
        for rank, idx in enumerate(sorted_idx[:3], 1):
            direction = "提升勝率" if sv[idx] > 0 else "降低勝率"
            st.markdown(
                f"**{rank}. {cn_labels[idx]}** = {X_input[0][idx]:.1f}"
                f"{direction} (SHAP: {sv[idx]:+.3f})"
            )

# ══════════════════════════════════════════════════════════════
# Tab 6：每周戰報（Claude API 生成）
# ══════════════════════════════════════════════════════════════

with tab6:
    import json as _json

    from dotenv import load_dotenv as _load_dotenv

    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.etl.weekly_report import gather_weekly_data, get_match_weeks

    _load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    import os as _os

    # 優先從 Streamlit Secrets 讀取，其次從環境變數（.env）
    _api_key = st.secrets.get("ANTHROPIC_API_KEY", "") or _os.environ.get("ANTHROPIC_API_KEY", "")

    st.subheader("每周戰報產生器")
    st.caption("根據比賽數據，透過 Claude AI 自動產生結構化中文戰報。")

    # ── 檢查 API Key ──────────────────────────────────────────
    if not _api_key:
        st.warning(
            "尚未設定 `ANTHROPIC_API_KEY`。請在專案根目錄的 `.env` 檔案中加入：\n\n"
            "```\nANTHROPIC_API_KEY=sk-ant-...\n```"
        )

    # ── 周次選擇器 ────────────────────────────────────────────
    weeks = get_match_weeks()
    if not weeks:
        st.info("資料庫中尚無比賽紀錄。")
        st.stop()

    week_labels = [
        f"第 {i+1} 周：{w[0]} ~ {w[1]}" for i, w in enumerate(weeks)
    ]
    # 預設選最新一周
    wr_col1, wr_col2 = st.columns([3, 1])
    with wr_col1:
        selected_week_label = st.selectbox(
            "選擇比賽周次", week_labels, index=len(week_labels) - 1,
            key="wr_week",
        )
    week_idx = week_labels.index(selected_week_label)
    date_from, date_to = weeks[week_idx]

    with wr_col2:
        wr_gender = st.selectbox(
            "組別", ["全部", "男子組", "女子組"], key="wr_gender"
        )
    wr_gender_code = {"男子組": "M", "女子組": "F"}.get(wr_gender)

    # ── 撈取資料並預覽 ────────────────────────────────────────
    weekly_data = gather_weekly_data(date_from, date_to, wr_gender_code)

    # 附加局數比資訊（從外部系統取得）
    match_index = fetch_match_index()
    for m in weekly_data.get("matches", []):
        mid = find_match_id(match_index, m["date"], m["opponent"])
        if mid:
            scores = fetch_set_scores(mid)
            if scores and len(scores) == 2:
                # 找出哪一邊是本隊
                t_a, t_b = scores[0], scores[1]
                m["set_score"] = f"{t_a['sets_won']}:{t_b['sets_won']}"
                m["set_details"] = [
                    {"team": t_a["team"], "sets_won": t_a["sets_won"],
                     "set_points": [s for s in t_a["sets"] if s is not None]},
                    {"team": t_b["team"], "sets_won": t_b["sets_won"],
                     "set_points": [s for s in t_b["sets"] if s is not None]},
                ]

    n_matches = len(weekly_data["matches"])

    if n_matches == 0:
        st.info("該周次無符合條件的比賽。")
        st.stop()

    st.markdown(f"該周共有 **{n_matches}** 場比賽紀錄。")

    # 原始數據預覽
    with st.expander("查看原始比賽數據摘要", expanded=False):
        for m in weekly_data["matches"]:
            st.markdown(
                f"**{m['date']}　{m['gender']}　{m['team_name']} vs {m['opponent']}**"
            )
            ts = m["team_stats"]
            st.markdown(
                f"- 團隊總得分 {ts['total_points']}　"
                f"攻擊 {ts['attack_points']}/{ts['attack_total']}"
                f"（{ts['attack_rate']}%）　"
                f"攔網 {ts['block_points']}　發球 {ts['serve_points']}"
            )
            if m["players"]:
                top = m["players"][0]
                st.markdown(
                    f"- 最佳得分：{top['name']}（{top['position']}）"
                    f" {top['total_points']} 分"
                )
            st.markdown("---")

    # ── 產生戰報按鈕 ──────────────────────────────────────────
    REPORT_SYSTEM_PROMPT = """\
你是一位專業的排球賽事記者，專門報導台灣企業排球聯賽（TVL）。
請根據提供的比賽數據撰寫該周的戰報。

撰寫規範：
1. 使用繁體中文，語氣專業但生動，適合球迷閱讀。
2. 比賽結果必須以「局數比」(set_score) 呈現（例如「3:1」），不要用總得分表示勝負。各局比分 (set_details) 可在摘要中附帶說明。
3. 每場比賽包含：比賽結果摘要、關鍵球員點評（引用具體數據）、戰術觀察。
4. 若球員本場表現明顯高於或低於賽季平均，請特別指出。
5. 在最後加上「本周數據亮點」段落，列出該周最佳表現者。
6. 使用 Markdown 格式，包含標題層級和粗體強調。
7. 攻擊成功率 (ASR) = 攻擊得分 / 攻擊總數 × 100%。
8. 不要編造任何數據中沒有的資訊。"""

    def _build_user_prompt(data: dict) -> str:
        return (
            f"以下是 {data['period']} 的 TVL 企業排球聯賽比賽數據，"
            f"共 {len(data['matches'])} 場比賽。\n"
            f"請根據這些數據撰寫本周戰報。\n\n"
            f"```json\n{_json.dumps(data, ensure_ascii=False, indent=2)}\n```"
        )

    if not _api_key:
        st.button("產生 AI 戰報", disabled=True, help="請先設定 ANTHROPIC_API_KEY")
    elif st.button("產生 AI 戰報", type="primary", key="wr_generate"):
        with st.spinner("正在透過 Claude API 產生戰報，請稍候..."):
            try:
                import anthropic

                client = anthropic.Anthropic(api_key=_api_key)
                message = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=16384,
                    system=REPORT_SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": _build_user_prompt(weekly_data)}
                    ],
                )
                report_text = message.content[0].text

                # 顯示戰報
                st.markdown("---")
                st.markdown(report_text)

                # 儲存到 session state 供下載
                st.session_state["weekly_report_text"] = report_text
                st.session_state["weekly_report_period"] = weekly_data["period"]

                # Token 用量
                st.caption(
                    f"Token 用量：輸入 {message.usage.input_tokens} / "
                    f"輸出 {message.usage.output_tokens}"
                )
            except Exception as e:
                st.error(f"API 呼叫失敗：{e}")

    # ── 下載按鈕（戰報產生後顯示） ────────────────────────────
    if "weekly_report_text" in st.session_state:
        period = st.session_state.get("weekly_report_period", "")
        safe_period = period.replace(" ", "").replace("~", "_")
        st.download_button(
            label="下載戰報 (.md)",
            data=st.session_state["weekly_report_text"].encode("utf-8"),
            file_name=f"TVL_戰報_{safe_period}.md",
            mime="text/markdown",
            key="wr_download",
        )
