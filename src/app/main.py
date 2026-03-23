"""
TVL 企業排球聯賽進階數據分析儀表板
使用 Streamlit + Plotly，基於 Proxy Metrics 呈現球員進階數據。
包含：個人深度分析 + 聯盟分佈與同位置 PR 值比較。
"""

import sys
from pathlib import Path

# 確保專案根目錄在 sys.path 中（Streamlit Cloud 需要）
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import streamlit as st

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

# ── 共用函式（從 helpers 匯入） ──────────────────────────────
from src.app.helpers import load_data, inject_mobile_css

# ── 注入手機 RWD CSS ─────────────────────────────────────────
inject_mobile_css()

# ── Tab 模組（延遲匯入各分頁） ───────────────────────────────
from src.app.tabs import player_deep, league_pr, match_trend, box_score, prediction, weekly_report_tab


# ── 側邊欄篩選器（三層連動） ──────────────────────────────────

st.sidebar.title("TVL 進階數據儀表板")
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

player_display = [
    f"#{int(r['jersey_number'])} {r['name']}" if pd.notna(r["jersey_number"]) else r["name"]
    for _, r in players_df.iterrows()
]

selected_display = st.sidebar.selectbox("選擇球員", player_display)
selected_idx = player_display.index(selected_display)
player_id = int(players_df.iloc[selected_idx]["player_id"])
player_name = players_df.iloc[selected_idx]["name"]
player_position = players_df.iloc[selected_idx].get("position", None)

# ── 頁面標題 ──────────────────────────────────────────────────

pos_display = f"（{player_position}）" if player_position else ""
st.title(f"{player_name}{pos_display}　{gender}・{team_name}")
st.markdown("---")

# ── 共用 Context（傳遞給各 Tab 模組） ────────────────────────

ctx = {
    "player_id": player_id,
    "player_name": player_name,
    "player_position": player_position,
    "gender_code": gender_code,
    "gender": gender,
    "team_name": team_name,
    "team_id": team_id,
}

# ── 分頁結構 ──────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "球員個人深度", "聯盟 PR 值與分佈", "逐場趨勢", "單場 Box Score", "賽果預測",
    "每周戰報",
])

with tab1:
    player_deep.render(ctx)

with tab2:
    league_pr.render(ctx)

with tab3:
    match_trend.render(ctx)

with tab4:
    box_score.render(ctx)

with tab5:
    prediction.render(ctx, cjk_font_path=_CJK_FONT_PATH, cjk_font_stack=CJK_FONT_STACK)

with tab6:
    weekly_report_tab.render(ctx)
