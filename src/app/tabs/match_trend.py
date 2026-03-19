"""
Tab 3：逐場賽事明細與對戰分析（熱力表 + 對手箱型圖）
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app.helpers import load_data, safe_pct, vec_pct, responsive_chart_config, compact_margin


def render(ctx: dict):
    player_id = ctx["player_id"]
    player_name = ctx["player_name"]
    player_position = ctx["player_position"]
    gender_code = ctx["gender_code"]
    gender = ctx["gender"]
    team_name = ctx["team_name"]

    match_df = load_data(
        "SELECT * FROM player_match_stats WHERE player_id = ? ORDER BY match_date",
        (player_id,),
    )

    if match_df.empty:
        st.info("該球員目前沒有比賽數據紀錄。")
        st.stop()

    # ── 計算逐場進階指標（向量化） ────────────────────────────
    md = match_df.copy()
    md["單場ASR"] = vec_pct(md["attack_points"], md["attack_total"])
    md["單場GP%"] = vec_pct(md["receive_excellent"], md["receive_total"])
    md["單場DIG%"] = vec_pct(md["dig_excellent"], md["dig_total"])
    md["單場SET%"] = vec_pct(md["set_excellent"], md["set_total"])
    md["單場DEF%"] = vec_pct(
        md["receive_excellent"] + md["dig_excellent"],
        md["receive_total"] + md["dig_total"],
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
    gradient_cols = [
        "總得分", "攻擊得分", "ASR%", "攔網得分", "發球得分",
        "GP%", "DIG%", "SET%",
    ]
    # 只對存在且有變異的欄位上色
    valid_gradient = [
        c for c in gradient_cols
        if c in heat_df.columns and heat_df[c].nunique() > 1
    ]

    styled = heat_df.style.format(
        {c: "{:.1f}" for c in ["ASR%", "GP%", "DIG%", "SET%"] if c in heat_df.columns}
    )
    if valid_gradient:
        styled = styled.background_gradient(cmap="YlGnBu", subset=valid_gradient)

    st.dataframe(styled, use_container_width=True, hide_index=True, height=400)

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
        "OH": "單場總得分",
        "OP": "單場總得分",
        "MB": "單場攔網得分",
        "S": "單場舉球到位率 (SET%)",
        "L": "單場綜合防守到位率 (DEF%)",
    }
    default_metric = POS_DEFAULT_METRIC.get(player_position, "單場總得分")
    metric_names = list(metric_options.keys())
    default_idx = (
        metric_names.index(default_metric) if default_metric in metric_names else 0
    )

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
        height=420,
        margin=compact_margin(l=30, r=30, t=30, b=40),
        xaxis_title="對手",
        yaxis_title=selected_metric,
    )
    fig_box.update_traces(
        hovertemplate=(
            f"對手: %{{x}}<br>{selected_metric}: %{{y:.1f}}<extra></extra>"
        ),
    )
    st.plotly_chart(fig_box, use_container_width=True, config=responsive_chart_config())

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
        st.dataframe(opp_summary, use_container_width=True, hide_index=True)
