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

from src.app.helpers import fetch_match_index, fetch_set_scores, find_match_id
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
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("GOOGLE_API_KEY")


GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]


def _call_gemini(api_key: str, user_prompt: str) -> str:
    """呼叫 Gemini API 產生戰報，自動重試與 fallback 模型。"""
    import time
    from google import genai

    client = genai.Client(api_key=api_key)

    for model in GEMINI_MODELS:
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
                return response.text
            except Exception as e:
                if "429" in str(e) and attempt == 0:
                    time.sleep(30)  # 速率限制，等 30 秒重試
                    continue
                if "429" in str(e):
                    break  # 這個模型額度用完，換下一個
                raise  # 其他錯誤直接拋出

    raise RuntimeError("所有 Gemini 模型額度已用完，請稍後再試。")


def _attach_set_scores(weekly_data: dict, match_index: list[dict]) -> dict:
    for match in weekly_data.get("matches", []):
        mid = find_match_id(match_index, match.get("date", ""), match.get("opponent", ""))
        if not mid:
            continue
        scores = fetch_set_scores(mid)
        if not scores or len(scores) < 2:
            continue
        t_a, t_b = scores[0], scores[1]
        match["set_score"] = f"{t_a['sets_won']}:{t_b['sets_won']}"
        match["set_details"] = [
            {"team": t_a["team"], "sets_won": t_a["sets_won"],
             "set_points": [s for s in t_a["sets"] if s is not None]},
            {"team": t_b["team"], "sets_won": t_b["sets_won"],
             "set_points": [s for s in t_b["sets"] if s is not None]},
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

    st.markdown(
        f"#### {a['gender']}　{date_short}　{a['team_name']} vs {a['opponent']}"
        + (f"　**{set_score}**" if set_score else "")
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

    match_index = fetch_match_index()
    if match_index:
        weekly_data = _attach_set_scores(weekly_data, match_index)

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
