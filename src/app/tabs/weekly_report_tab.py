"""
Tab 6：每周戰報（Claude API 生成）
提供周次選擇、性別篩選，以視覺化卡片呈現該周比賽摘要，並可透過 Claude API 產生專業戰報。
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.app.helpers import fetch_match_index, fetch_set_scores, find_match_id

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

# 延遲匯入 weekly_report 模組（避免循環匯入）
from src.etl.weekly_report import gather_weekly_data, get_match_weeks

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


def _get_api_key() -> str | None:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("ANTHROPIC_API_KEY")


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


def _render_match_card(m: dict):
    """渲染單場比賽的視覺化卡片。"""
    ts = m["team_stats"]
    set_score = m.get("set_score", "")
    date_short = m["date"][5:]  # MM-DD

    # 比賽標題列
    st.markdown(
        f"#### {date_short}　{m['team_name']} vs {m['opponent']}"
        + (f"　**{set_score}**" if set_score else "")
    )

    # 各局比分
    if "set_details" in m and isinstance(m["set_details"], list) and len(m["set_details"]) == 2:
        d_a, d_b = m["set_details"][0], m["set_details"][1]
        set_str = "　".join(
            f"**{a}**:{b}" if a > b else f"{a}:**{b}**"
            for a, b in zip(d_a.get("set_points", []), d_b.get("set_points", []))
        )
        if set_str:
            st.caption(f"{d_a['team']} vs {d_b['team']}　｜　{set_str}")

    # 團隊數據指標（3 欄）
    c1, c2, c3 = st.columns(3)
    c1.metric("團隊總得分", ts["total_points"])
    atk_rate = ts.get("attack_rate")
    c2.metric(
        "攻擊",
        f"{ts['attack_points']}/{ts['attack_total']}",
        delta=f"ASR {atk_rate}%" if atk_rate else None,
    )
    c3.metric("攔網 / 發球", f"{ts['block_points']} / {ts['serve_points']}")

    # 上場球員表格
    if m["players"]:
        rows = []
        for p in m["players"][:8]:  # 最多顯示前 8 名
            row = {
                "球員": p["name"],
                "位置": p.get("position") or "-",
                "局數": p["sets_played"],
                "得分": p["total_points"],
                "攻擊": f"{p['attack_points']}/{p['attack_total']}",
                "攔網": p["block_points"],
                "發球": p["serve_points"],
            }
            # 賽季對比
            if "vs_season_ppg" in p:
                diff = p["vs_season_ppg"]
                row["vs 賽季均"] = f"{diff:+.1f}"
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=min(40 * len(rows) + 38, 350))

    st.markdown("---")


def render(ctx):
    st.subheader("每周戰報")
    st.caption("根據比賽數據，透過 Claude AI 自動產生結構化中文戰報。")

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

    n_matches = len(weekly_data.get("matches", []))

    if n_matches == 0:
        st.info("該周次無符合條件的比賽。")
        st.stop()

    st.markdown(f"該周共有 **{n_matches}** 場比賽紀錄。")
    st.markdown("---")

    # ── 比賽卡片視覺化預覽 ────────────────────────────────────
    for m in weekly_data["matches"]:
        _render_match_card(m)

    # ── 產生 AI 戰報 ──────────────────────────────────────────
    api_key = _get_api_key()

    if not api_key:
        st.info(
            "如需 AI 戰報功能，請在 `.env` 或 Streamlit Secrets 中設定 "
            "`ANTHROPIC_API_KEY`。"
        )
    elif st.button("產生 AI 戰報", type="primary", key="wr_generate"):
        data_json = json.dumps(weekly_data, ensure_ascii=False, indent=2)
        with st.spinner("正在透過 Claude API 產生戰報，請稍候..."):
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                message = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=16384,
                    system=REPORT_SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": (
                            f"以下是 {weekly_data['period']} 的 TVL 企業排球聯賽比賽數據，"
                            f"共 {n_matches} 場比賽。\n"
                            f"請根據這些數據撰寫本周戰報。\n\n"
                            f"```json\n{data_json}\n```"
                        )}
                    ],
                )
                report_text = message.content[0].text
                st.session_state["weekly_report_text"] = report_text
                st.session_state["weekly_report_period"] = weekly_data["period"]

                st.caption(
                    f"Token 用量：輸入 {message.usage.input_tokens} / "
                    f"輸出 {message.usage.output_tokens}"
                )
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
