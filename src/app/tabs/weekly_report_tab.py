"""
Tab 6：每周戰報（Gemini AI 生成）
提供周次選擇、性別篩選，以視覺化卡片呈現該周比賽摘要，並透過 Gemini API 產生專業戰報。
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.app.helpers import load_data
from src.etl.weekly_report import gather_weekly_data, get_match_weeks

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

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


def _get_gemini_key() -> str | None:
    for key_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        try:
            return st.secrets[key_name]
        except (KeyError, FileNotFoundError):
            val = os.getenv(key_name)
            if val:
                return val
    return None


GEMINI_MODELS = ["gemini-3-flash-preview", "gemini-2.0-flash", "gemini-2.0-flash-lite"]


def _call_gemini(api_key: str, user_prompt: str) -> str:
    """呼叫 Gemini API 產生戰報，自動重試與 fallback 模型。"""
    import time
    from google import genai

    client = genai.Client(api_key=api_key)

    status_placeholder = st.empty()

    for model in GEMINI_MODELS:
        status_placeholder.caption(f"嘗試模型：{model}...")
        for attempt in range(2):  # 每個模型最多試 2 次
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config={
                        "system_instruction": REPORT_SYSTEM_PROMPT,
                        "max_output_tokens": 8192,
                        "temperature": 0.7,
                    },
                )
                status_placeholder.caption(f"使用模型：{model}")
                return response.text
            except Exception as e:
                if "429" in str(e) and attempt == 0:
                    status_placeholder.caption(f"模型 {model} 速率限制，30 秒後重試...")
                    time.sleep(30)
                    continue
                if "429" in str(e):
                    status_placeholder.caption(f"模型 {model} 額度已滿，切換下一個...")
                    break  # 這個模型額度用完，換下一個
                raise  # 其他錯誤直接拋出

    status_placeholder.empty()
    raise RuntimeError("所有 Gemini 模型額度已用完，請稍後再試。")


def _attach_set_scores(weekly_data: dict) -> dict:
    """
    從本地 matches 表附加各局比分與黃金決勝局比分。
    """
    dates = list({m["date"] for m in weekly_data.get("matches", [])})
    if not dates:
        return weekly_data

    placeholders = ",".join(["?"] * len(dates))
    matches_db = load_data(
        f"""SELECT match_date, home_team, away_team,
                   home_set1, home_set2, home_set3, home_set4, home_set5,
                   away_set1, away_set2, away_set3, away_set4, away_set5,
                   home_total, away_total,
                   home_sets_won, away_sets_won, is_golden_set
            FROM matches
            WHERE match_date IN ({placeholders})""",
        tuple(dates),
    )
    if matches_db.empty:
        return weekly_data

    # 建立 lookup: (date, frozenset(teams)) -> list of rows
    from collections import defaultdict
    lookup: dict[tuple, list] = defaultdict(list)
    for _, row in matches_db.iterrows():
        key = (row["match_date"], frozenset([row["home_team"], row["away_team"]]))
        lookup[key].append(row)

    for match in weekly_data.get("matches", []):
        team_name = match.get("team_name", "")
        opponent = match.get("opponent", "")
        key = (match["date"], frozenset([team_name, opponent]))
        rows = lookup.get(key, [])

        for row in rows:
            home = row["home_team"]
            away = row["away_team"]

            if row["is_golden_set"] == 1:
                # 黃金決勝局實際比分
                h_score = int(row["home_set1"]) if pd.notna(row["home_set1"]) else None
                a_score = int(row["away_set1"]) if pd.notna(row["away_set1"]) else None
                if h_score is None or a_score is None:
                    continue
                if "golden_set" not in match:
                    match["golden_set"] = {}
                # 對齊隊名方向（match 的 team_name 對應 home 或 away）
                if team_name == home or home in team_name or team_name in home:
                    match["golden_set"]["score"] = f"{h_score}:{a_score}"
                else:
                    match["golden_set"]["score"] = f"{a_score}:{h_score}"
                match["golden_set"]["score_detail"] = {home: h_score, away: a_score}
            else:
                # 正規賽局比分
                set_cols = ["home_set1", "home_set2", "home_set3", "home_set4", "home_set5"]
                home_pts = [int(row[c]) for c in set_cols if pd.notna(row[c])]
                away_pts = [int(row[c.replace("home_", "away_")]) for c in set_cols if pd.notna(row[c])]

                h_won = int(row["home_sets_won"])
                a_won = int(row["away_sets_won"])

                # 對齊隊名方向
                if team_name == home or home in team_name or team_name in home:
                    match["set_score"] = f"{h_won}:{a_won}"
                    match["set_details"] = [
                        {"team": home, "sets_won": h_won, "set_points": home_pts},
                        {"team": away, "sets_won": a_won, "set_points": away_pts},
                    ]
                else:
                    match["set_score"] = f"{a_won}:{h_won}"
                    match["set_details"] = [
                        {"team": away, "sets_won": a_won, "set_points": away_pts},
                        {"team": home, "sets_won": h_won, "set_points": home_pts},
                    ]

    return weekly_data


def _group_matches(matches: list[dict]) -> list[dict]:
    """
    將同一場比賽的兩隊資料合併為一筆。
    用 (date, frozenset({team_name, opponent})) 作為 key 去重。
    """
    seen = {}
    for m in matches:
        key = (m["date"], frozenset([m["team_name"], m["opponent"]]))
        if key not in seen:
            seen[key] = {"team_a": m, "team_b": None}
        else:
            seen[key]["team_b"] = m
    return list(seen.values())


def _build_player_df(players: list[dict]) -> pd.DataFrame:
    rows = []
    for p in players[:8]:
        row = {
            "球員": p["name"],
            "位置": p.get("position") or "-",
            "局數": p["sets_played"],
            "得分": p["total_points"],
            "攻擊": f"{p['attack_points']}/{p['attack_total']}",
            "攔網": p["block_points"],
            "發球": p["serve_points"],
        }
        if "vs_season_ppg" in p:
            row["vs 賽季均"] = f"{p['vs_season_ppg']:+.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def _render_team_side(m: dict):
    ts = m["team_stats"]
    atk_rate = ts.get("attack_rate")

    c1, c2, c3 = st.columns(3)
    c1.metric("總得分", ts["total_points"])
    c2.metric("攻擊", f"{ts['attack_points']}/{ts['attack_total']}",
              delta=f"ASR {atk_rate}%" if atk_rate else None)
    c3.metric("攔網 / 發球", f"{ts['block_points']} / {ts['serve_points']}")

    if m["players"]:
        df = _build_player_df(m["players"])
        st.dataframe(df, use_container_width=True, hide_index=True,
                     height=min(36 * len(df) + 38, 320))


def _render_golden_set(a: dict, b: dict | None):
    """渲染黃金決勝局區塊。"""
    gs_a = a.get("golden_set")
    gs_b = b.get("golden_set") if b else None
    if not gs_a and not gs_b:
        return

    # 顯示實際比分（優先從外部系統取得）
    score_text = ""
    if gs_a and "score" in gs_a:
        score_text = f"**{a['team_name']}** {gs_a['score']} **{a['opponent']}**"
    elif gs_b and "score" in gs_b:
        score_text = f"**{b['team_name']}** {gs_b['score']} **{b['opponent']}**"

    if score_text:
        st.markdown(f"##### Golden Set 黃金決勝局　{score_text}")
    else:
        st.markdown("##### Golden Set 黃金決勝局")

    col_a, col_b = st.columns(2)

    for col, side, team_name in [
        (col_a, gs_a, a["team_name"]),
        (col_b, gs_b, b["team_name"] if b else None),
    ]:
        if not side or not team_name:
            continue
        with col:
            # 顯示實際局分（從外部系統）或球員得分加總（fallback）
            score_detail = side.get("score_detail", {})
            actual_score = None
            for ext_name, pts in score_detail.items():
                if ext_name in team_name or team_name in ext_name:
                    actual_score = pts
                    break
            if actual_score is not None:
                st.metric(f"{team_name}", f"{actual_score} 分")
            else:
                ts = side.get("team_stats", {})
                st.metric(f"{team_name}", f'{ts.get("total_points", "?")} 分*')

            if side.get("players"):
                rows = []
                for p in side["players"]:
                    rows.append({
                        "球員": p["name"],
                        "得分": p["total_points"],
                        "攻擊": p["attack_points"],
                        "攔網": p["block_points"],
                        "發球": p["serve_points"],
                    })
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True, hide_index=True,
                    height=min(36 * len(rows) + 38, 200),
                )


def _render_match_card(group: dict):
    a = group["team_a"]
    b = group["team_b"]
    date_short = a["date"][5:]

    set_score = a.get("set_score", "")
    set_str = ""
    if "set_details" in a and isinstance(a["set_details"], list) and len(a["set_details"]) == 2:
        d_a, d_b = a["set_details"][0], a["set_details"][1]
        set_str = "　".join(
            f"**{sa}**:{sb}" if sa > sb else f"{sa}:**{sb}**"
            for sa, sb in zip(d_a.get("set_points", []), d_b.get("set_points", []))
        )

    has_golden = a.get("golden_set") or (b and b.get("golden_set"))

    st.markdown(
        f"#### {a['gender']}　{date_short}　{a['team_name']} vs {a['opponent']}"
        + (f"　**{set_score}**" if set_score else "")
        + ("　:trophy: Golden Set" if has_golden else "")
    )
    if set_str:
        st.caption(set_str)

    if b:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**{a['team_name']}**")
            _render_team_side(a)
        with col_b:
            st.markdown(f"**{b['team_name']}**")
            _render_team_side(b)
    else:
        st.markdown(f"**{a['team_name']}**")
        _render_team_side(a)

    if has_golden:
        _render_golden_set(a, b)

    st.markdown("---")


def render(ctx):
    st.subheader("每周戰報")
    st.caption("根據比賽數據，透過 Gemini AI 自動產生結構化中文戰報。")

    # ── 周次選擇器 ────────────────────────────────────────────
    weeks = get_match_weeks()
    if not weeks:
        st.info("資料庫中尚無比賽紀錄。")
        st.stop()

    week_labels = [
        f"第 {i+1} 周：{w[0]} ~ {w[1]}" for i, w in enumerate(weeks)
    ]

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

    # ── 撈取資料 ──────────────────────────────────────────────
    weekly_data = gather_weekly_data(date_from, date_to, wr_gender_code)

    weekly_data = _attach_set_scores(weekly_data)

    all_matches = weekly_data.get("matches", [])
    if not all_matches:
        st.info("該周次無符合條件的比賽。")
        st.stop()

    grouped = _group_matches(all_matches)

    st.markdown(f"該周共有 **{len(grouped)}** 場比賽。")
    st.markdown("---")

    for g in grouped:
        _render_match_card(g)

    # ── 產生 AI 戰報 ──────────────────────────────────────────
    gemini_key = _get_gemini_key()

    if not gemini_key:
        st.info(
            "如需 AI 戰報功能，請在 `.env` 或 Streamlit Secrets 中設定 "
            "`GOOGLE_API_KEY`（從 [Google AI Studio](https://aistudio.google.com/) 免費取得）。"
        )
    elif st.button("產生 AI 戰報", type="primary", key="wr_generate"):
        data_json = json.dumps(weekly_data, ensure_ascii=False, indent=2)
        user_prompt = (
            f"以下是 {weekly_data['period']} 的 TVL 企業排球聯賽比賽數據，"
            f"共 {len(grouped)} 場比賽。\n"
            f"請根據這些數據撰寫本周戰報。\n\n"
            f"```json\n{data_json}\n```"
        )
        with st.spinner("正在透過 Gemini API 產生戰報，請稍候..."):
            try:
                report_text = _call_gemini(gemini_key, user_prompt)
                st.session_state["weekly_report_text"] = report_text
                st.session_state["weekly_report_period"] = weekly_data["period"]
            except Exception as e:
                st.error(f"API 呼叫失敗：{e}")

    # ── 顯示已生成的戰報 ──────────────────────────────────────
    if "weekly_report_text" in st.session_state:
        st.markdown("---")
        st.markdown("### AI 戰報")
        st.markdown(st.session_state["weekly_report_text"])

        period = st.session_state.get("weekly_report_period", "")
        safe_period = period.replace(" ", "").replace("~", "_")
        st.download_button(
            label="下載戰報 (.md)",
            data=st.session_state["weekly_report_text"].encode("utf-8"),
            file_name=f"TVL_戰報_{safe_period}.md",
            mime="text/markdown",
            key="wr_download",
        )
