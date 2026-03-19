"""
Tab 4：單場 Box Score 與對戰比較
提供單場比賽雙方 Box Score 並列比較、局比分資訊與 Top-10 排名圖。
"""

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.helpers import (
    load_data, enrich_box_score, vec_pct,
    fetch_match_index, fetch_set_scores, find_match_id,
    OPP_SHORT_TO_TEAM, responsive_chart_config, compact_margin,
)


def _format_box_score(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "name": "姓名", "position": "位置", "sets_played": "局數",
        "total_points": "總得分",
        "attack_points": "攻擊得", "attack_total": "攻擊總", "ASR%": "ASR%",
        "block_points": "攔網得",
        "serve_points": "發球得", "serve_total": "發球總",
        "receive_excellent": "接發好", "receive_total": "接發總", "GP%": "GP%",
        "dig_excellent": "防守好", "dig_total": "防守總", "DIG%": "DIG%",
    }
    return df[[c for c in cols if c in df.columns]].rename(columns=cols)


def _style_box_score(df: pd.DataFrame):
    pct_cols = [c for c in ["ASR%", "GP%", "DIG%"] if c in df.columns]
    styled = df.style.format({c: "{:.1f}" for c in pct_cols})
    grad_cols = [c for c in ["總得分", "ASR%"] if c in df.columns and df[c].nunique() > 1]
    if grad_cols:
        styled = styled.background_gradient(cmap="Blues", subset=grad_cols)
    return styled


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


def render(ctx: dict):
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

    match_labels = (matches_df["match_date"] + " vs " + matches_df["opponent"]).tolist()

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
            played_sets = sum(1 for s in t_a["sets"] if s is not None and s > 0)

            sc1, sc2, sc3 = st.columns([2, 3, 2])
            with sc1:
                st.markdown(
                    f"<h2 style='text-align:right; margin:0;'>{html.escape(t_a['team'])}</h2>",
                    unsafe_allow_html=True,
                )
            with sc2:
                set_strs_a = "　".join(
                    str(s) for s in t_a["sets"][:played_sets] if s is not None
                )
                set_strs_b = "　".join(
                    str(s) for s in t_b["sets"][:played_sets] if s is not None
                )
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
                    f"<h2 style='text-align:left; margin:0;'>{html.escape(t_b['team'])}</h2>",
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

    # ── 雙方 Box Score 並排 ───────────────────────────────────
    st.subheader(f"📊 {sel_date}　{bs_team_name} vs {sel_opponent}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"**{bs_team_name}**")
        if team_a_df.empty:
            st.info("Team A 無該場數據。")
        else:
            a_enriched = enrich_box_score(team_a_df)
            a_display = _format_box_score(a_enriched)
            st.dataframe(
                _style_box_score(a_display),
                use_container_width=True, hide_index=True, height=380,
            )
            st.caption(f"團隊總得分：**{int(team_a_df['total_points'].sum())}**")
            st.download_button(
                label="📥 下載 CSV",
                data=a_display.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{bs_team_name}_{sel_date}_box_score.csv",
                mime="text/csv", key="dl_team_a",
            )

    with col_b:
        st.markdown(f"**{sel_opponent}**")
        if team_b_df.empty:
            st.info("對手數據無法取得（可能為跨組別或資料缺失）。")
        else:
            b_enriched = enrich_box_score(team_b_df)
            b_display = _format_box_score(b_enriched)
            st.dataframe(
                _style_box_score(b_display),
                use_container_width=True, hide_index=True, height=380,
            )
            st.caption(f"團隊總得分：**{int(team_b_df['total_points'].sum())}**")
            st.download_button(
                label="📥 下載 CSV",
                data=b_display.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{sel_opponent}_{sel_date}_box_score.csv",
                mime="text/csv", key="dl_team_b",
            )

    st.markdown("---")

    # ── 雙方排行榜比較 ─────────────────────────────────────────
    has_both = not team_a_df.empty and not team_b_df.empty
    has_any = not team_a_df.empty or not team_b_df.empty

    if has_any:
        rank_label = st.selectbox(
            "選擇排行指標", list(RANK_METRICS.keys()), key="bs_rank_metric"
        )
        rank_col, is_pct_metric = RANK_METRICS[rank_label]

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
            top_n = combined.nlargest(10, rank_col).sort_values(rank_col, ascending=True)

            title_suffix = "（雙方 Top 10）" if has_both else f"（{parts[0].iloc[0]['team']}）"
            st.subheader(f"🏆 {rank_label}排行 {title_suffix}")

            fig_rank = go.Figure()
            team_colors = {bs_team_name: "#636EFA", sel_opponent: "#EF553B"}

            for team_label in top_n["team"].unique():
                sub = top_n[top_n["team"] == team_label]
                color = team_colors.get(team_label, "#636EFA")
                text_vals = sub[rank_col].apply(
                    lambda v: f"{v:.1f}%" if is_pct_metric else f"{int(v)}"
                )

                fig_rank.add_trace(go.Bar(
                    y=sub["name"], x=sub[rank_col],
                    name=team_label, orientation="h",
                    marker_color=color, text=text_vals, textposition="auto",
                    hovertemplate=(
                        "%{y}<br>"
                        f"{rank_label}: %{{x:.1f}}<br>"
                        "總得分: %{customdata[0]}<br>"
                        "攻擊: %{customdata[1]}/%{customdata[2]} (ASR %{customdata[3]:.1f}%)"
                        "<extra></extra>"
                    ),
                    customdata=sub[["total_points", "attack_points", "attack_total", "ASR%"]].values,
                ))

            fig_rank.update_layout(
                barmode="group",
                height=max(350, len(top_n) * 35 + 100),
                margin=compact_margin(l=80, r=20, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                xaxis_title=rank_label, yaxis_title="",
            )
            st.plotly_chart(fig_rank, use_container_width=True, config=responsive_chart_config())
