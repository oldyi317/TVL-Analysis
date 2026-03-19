"""
Tab 6：每周戰報（Claude API 生成）
提供周次選擇、性別篩選，彙整該周比賽數據後呼叫 Claude API 產生專業戰報。
"""

import json
import os
from pathlib import Path

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


def _get_api_key() -> str | None:
    """從 st.secrets 或環境變數取得 Anthropic API key。"""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("ANTHROPIC_API_KEY")


def _attach_set_scores(weekly_data: dict, match_index: list[dict]) -> dict:
    """為每場比賽附加局比分資料（set_score 與 set_details）。"""
    for match in weekly_data.get("matches", []):
        date = match.get("date", "")
        opponent = match.get("opponent", "")
        mid = find_match_id(match_index, date, opponent)
        if not mid:
            continue
        scores = fetch_set_scores(mid)
        if not scores or len(scores) < 2:
            continue

        team_name = match.get("team_name", "")
        # 判斷哪一行是我方、哪一行是對手
        row_a, row_b = scores[0], scores[1]
        if team_name in row_b.get("team", ""):
            row_a, row_b = row_b, row_a

        sets_won_a = row_a.get("sets_won", 0)
        sets_won_b = row_b.get("sets_won", 0)
        match["set_score"] = f"{sets_won_a}:{sets_won_b}"

        # 各局比分明細
        details = []
        sets_a = row_a.get("sets", [])
        sets_b = row_b.get("sets", [])
        for i in range(min(len(sets_a), len(sets_b))):
            if sets_a[i] is not None and sets_b[i] is not None:
                details.append(f"{sets_a[i]}:{sets_b[i]}")
        if details:
            match["set_details"] = ", ".join(details)

    return weekly_data


def _call_claude_api(api_key: str, data_json: str) -> str:
    """呼叫 Claude API 產生戰報，回傳 Markdown 文字。"""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16384,
        system=REPORT_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"以下是本周的比賽數據（JSON 格式），請撰寫戰報：\n\n{data_json}",
            }
        ],
    )
    return message.content[0].text


def render(ctx):
    """渲染每周戰報頁籤。"""
    st.header("每周戰報（Claude API 生成）")

    api_key = _get_api_key()
    if not api_key:
        st.warning("未設定 ANTHROPIC_API_KEY，請在 .env 或 Streamlit Secrets 中設定。")
        return

    # ── 周次選擇 ──
    weeks = get_match_weeks()
    if not weeks:
        st.info("目前沒有可用的比賽周次資料。")
        return

    week_labels = [f"{start} ~ {end}" for start, end in weeks]
    default_idx = len(week_labels) - 1  # 預設最新一周
    selected_idx = st.selectbox(
        "選擇周次",
        range(len(week_labels)),
        index=default_idx,
        format_func=lambda i: week_labels[i],
    )
    date_from, date_to = weeks[selected_idx]

    # ── 性別篩選 ──
    gender_options = {"全部": None, "男子組": "M", "女子組": "F"}
    gender_label = st.radio("組別篩選", list(gender_options.keys()), horizontal=True)
    gender_filter = gender_options[gender_label]

    # ── 彙整數據 ──
    weekly_data = gather_weekly_data(date_from, date_to, gender_filter)

    if not weekly_data.get("matches"):
        st.info("該周次沒有符合條件的比賽數據。")
        return

    # 附加局比分
    match_index = fetch_match_index()
    if match_index:
        weekly_data = _attach_set_scores(weekly_data, match_index)

    data_json = json.dumps(weekly_data, ensure_ascii=False, indent=2)

    # ── 數據預覽 ──
    with st.expander("查看原始數據（JSON）", expanded=False):
        st.code(data_json, language="json")

    # ── 產生戰報 ──
    if st.button("產生 AI 戰報", type="primary", use_container_width=True):
        with st.spinner("正在呼叫 Claude API 產生戰報..."):
            try:
                report = _call_claude_api(api_key, data_json)
                st.session_state["weekly_report"] = report
                st.session_state["weekly_report_period"] = weekly_data["period"]
            except Exception as e:
                st.error(f"呼叫 Claude API 失敗：{e}")
                return

    # ── 顯示戰報 ──
    if "weekly_report" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["weekly_report"])

        # 下載按鈕
        period = st.session_state.get("weekly_report_period", "report")
        filename = f"TVL_戰報_{period.replace(' ~ ', '_')}.md"
        st.download_button(
            label="下載戰報 (Markdown)",
            data=st.session_state["weekly_report"],
            file_name=filename,
            mime="text/markdown",
            use_container_width=True,
        )
