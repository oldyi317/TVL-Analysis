"""Tab 2 – League-wide PR (percentile ranking) & scatter quadrant analysis."""

import streamlit as st
import plotly.express as px

from src.app.helpers import (
    load_data,
    safe_pct,
    get_league_aggregated_stats,
    AXIS_OPTIONS,
    AXIS_NAMES,
    POS_DEFAULTS,
)


def render(ctx: dict) -> None:
    """Render the league PR tab.

    Parameters
    ----------
    ctx : dict
        Must contain keys: player_id, player_name, player_position,
        gender_code, gender, team_name.
    """
    player_id = ctx["player_id"]
    player_name = ctx["player_name"]
    player_position = ctx["player_position"]
    gender_code = ctx["gender_code"]
    gender = ctx["gender"]
    team_name = ctx["team_name"]

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
            if denom_key and me_row.get(denom_key, 0) < 10:
                col.metric(
                    f"{label} PR",
                    "N/A",
                    help=f"樣本數不足（{denom_key} < 10），不計入排名",
                )
            else:
                col.metric(
                    f"{label} PR",
                    f"{val:.0f}",
                    help=f"在所有 {pos_label} 中的百分位排名（0~100）",
                )
        st.markdown("---")

    # ── 位置篩選 ──────────────────────────────────────────────
    positions = sorted(league_all["position"].dropna().unique().tolist())
    if not positions:
        st.warning("球員位置資料不足，無法繪製散佈圖。")
        st.stop()

    me_pos = (
        me.iloc[0]["position"]
        if not me.empty and me.iloc[0]["position"] in positions
        else None
    )
    default_pos_idx = positions.index(me_pos) if me_pos else 0

    selected_pos = st.selectbox("選擇位置進行分析", positions, index=default_pos_idx)
    pos_df = league_all[league_all["position"] == selected_pos].copy()

    if len(pos_df) < 2:
        st.info(f"位置 {selected_pos} 僅有 {len(pos_df)} 位球員，資料不足。")
        st.stop()

    def_x_idx, def_y_idx = POS_DEFAULTS.get(selected_pos, (0, 1))

    ax_left, ax_right = st.columns(2)
    with ax_left:
        x_choice = st.selectbox("X 軸指標", AXIS_NAMES, index=def_x_idx)
    with ax_right:
        y_choice = st.selectbox("Y 軸指標", AXIS_NAMES, index=def_y_idx)

    x_col, x_label = AXIS_OPTIONS[x_choice]
    y_col, y_label = AXIS_OPTIONS[y_choice]

    st.subheader(f"\U0001f525 {selected_pos} 象限分析：{x_choice} vs {y_choice}")

    pos_df["is_selected"] = pos_df["player_id"] == player_id

    med_x = pos_df[x_col].median()
    med_y = pos_df[y_col].median()

    hover_data_cfg = {
        x_col: ":.1f",
        y_col: ":.1f",
        "total_points": ":.0f",
        "n_games": True,
        "team_name": True,
        "is_selected": False,
    }

    # 高對比色盤（適合淺色背景）
    TEAM_COLORS = [
        "#E63946", "#1D3557", "#2A9D8F", "#E9C46A", "#F4A261",
        "#264653", "#A8DADC", "#457B9D", "#6A0572", "#D62828",
    ]

    fig_scatter = px.scatter(
        pos_df,
        x=x_col,
        y=y_col,
        size="total_points",
        color="team_name",
        hover_name="name",
        hover_data=hover_data_cfg,
        labels={
            x_col: x_label,
            y_col: y_label,
            "total_points": "總得分",
            "team_name": "球隊",
            "n_games": "出賽場次",
        },
        size_max=40,
        color_discrete_sequence=TEAM_COLORS,
    )

    # 氣泡加深邊框，更容易辨識
    fig_scatter.update_traces(
        marker=dict(opacity=0.85, line=dict(width=1.5, color="white")),
    )

    # 在每個氣泡旁邊顯示球員名稱
    fig_scatter.update_traces(
        textposition="top center",
        textfont=dict(size=11),
    )
    for trace in fig_scatter.data:
        trace.text = pos_df.loc[pos_df["team_name"] == trace.name, "name"].tolist()
        trace.textposition = "top center"
        trace.mode = "markers+text"

    fig_scatter.add_hline(
        y=med_y,
        line_dash="dash",
        line_color="rgba(0,0,0,0.3)",
        line_width=1.5,
        annotation_text=f"中位數 {med_y:.1f}",
        annotation_position="top left",
        annotation_font_size=12,
        annotation_font_color="#555",
    )
    fig_scatter.add_vline(
        x=med_x,
        line_dash="dash",
        line_color="rgba(0,0,0,0.3)",
        line_width=1.5,
        annotation_text=f"中位數 {med_x:.1f}",
        annotation_position="top right",
        annotation_font_size=12,
        annotation_font_color="#555",
    )

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
                x=qx,
                y=qy,
                text=f"<b>{text}</b>",
                showarrow=False,
                font=dict(size=14, color=color),
                opacity=0.8,
            )

    if not me.empty and me.iloc[0]["position"] == selected_pos:
        me_r = me.iloc[0]
        fig_scatter.add_annotation(
            x=me_r[x_col],
            y=me_r[y_col],
            text=f"  ★ {player_name}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor="#FF6B35",
            font=dict(size=14, color="#FF6B35", family="Arial Black"),
            xanchor="left",
            ax=30,
            ay=-25,
        )

    fig_scatter.update_layout(
        height=620,
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(title="球隊", font=dict(size=13)),
        xaxis=dict(title=dict(font=dict(size=14)), tickfont=dict(size=12)),
        yaxis=dict(title=dict(font=dict(size=14)), tickfont=dict(size=12)),
        plot_bgcolor="rgba(248,249,250,1)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander(f"\U0001f4cb {selected_pos} 完整排行數據", expanded=False):
        rank_cols = {
            "name": "球員",
            "team_name": "球隊",
            "n_games": "出賽",
            "total_sets": "局數",
            "atk_tot": "攻擊總數",
            "atk_pts": "攻擊得分",
            "asr": "ASR%",
            "asr_pr": "ASR PR",
            "rcv_tot": "接發總數",
            "rcv_exc": "接發好球",
            "gp_pct": "GP%",
            "gp_pct_pr": "GP PR",
            "dig_tot": "防守總數",
            "dig_exc": "防守好球",
            "dig_pct": "DIG%",
            "dig_pct_pr": "DIG PR",
            "set_tot": "舉球總數",
            "set_exc": "舉球好球",
            "set_pct": "SET%",
            "set_pct_pr": "SET PR",
            "blk_pts": "攔網得分",
            "blk_per_set": "BLK/Set",
            "blk_per_set_pr": "BLK PR",
            "def_load": "防守負擔",
            "def_pct": "DEF%",
            "def_pct_pr": "DEF PR",
            "total_points": "總得分",
            "ppg": "場均得分",
        }
        available = {k: v for k, v in rank_cols.items() if k in pos_df.columns}
        show_df = (
            pos_df[list(available.keys())]
            .rename(columns=available)
            .sort_values("總得分", ascending=False)
        )
        st.dataframe(show_df, use_container_width=True, hide_index=True)
