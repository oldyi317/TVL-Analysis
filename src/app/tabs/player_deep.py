"""
Tab 1：球員個人深度分析（KPI + 雷達圖 + 逐場趨勢）
"""

import plotly.graph_objects as go
import streamlit as st

from src.app.helpers import load_data, safe_pct, vec_pct

# 樣本數門檻：比率型指標分母 < 10 → N/A
MIN_DENOM = 10


def _rate_display(val: float, denom: float) -> str:
    return f"{val:.1f}%" if denom >= MIN_DENOM else "N/A"


def _rate_delta(val: float, denom: float, league_val: float) -> str | None:
    if denom < MIN_DENOM:
        return None
    return f"{val - league_val:+.1f}% vs 聯盟均"


def _rate_safe(val: float, denom: float) -> float:
    return val if denom >= MIN_DENOM else 0.0


def _load_league_agg(gender_code: str, pos_filter: str = "", params: tuple = ()):
    """撈取聯盟聚合數據（全組別或特定位置）。"""
    return load_data(
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
        params,
    ).iloc[0]


def _parse_agg(lg):
    """從聯盟聚合 row 計算各項比率。"""
    return {
        "asr": safe_pct(lg["atk_pts"], lg["atk_tot"]),
        "gp": safe_pct(lg["rcv_exc"], lg["rcv_tot"]),
        "ace": safe_pct(lg["srv_pts"], lg["srv_tot"]),
        "dig": safe_pct(lg["dig_exc"], lg["dig_tot"]),
        "set": safe_pct(lg["set_exc"], lg["set_tot"]),
        "def": safe_pct(
            (lg["rcv_exc"] or 0) + (lg["dig_exc"] or 0),
            (lg["rcv_tot"] or 0) + (lg["dig_tot"] or 0),
        ),
        "blk": (lg["blk_pts"] / lg["tot_sets"]
                if lg["tot_sets"] and lg["tot_sets"] > 0 else 0),
        "ppg": (lg["tot_pts"] / lg["n_games"]
                if lg["n_games"] and lg["n_games"] > 0 else 0),
    }


def render(ctx: dict):
    player_id = ctx["player_id"]
    player_name = ctx["player_name"]
    player_position = ctx["player_position"]
    gender_code = ctx["gender_code"]
    gender = ctx["gender"]

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

    # ── 全聯盟平均 ─────────────────────────────────────────────
    la = _parse_agg(_load_league_agg(gender_code, params=(gender_code,)))

    # ── 依位置動態 KPI 卡片 ────────────────────────────────────
    POS_KPI_MAP = {
        "OH": [
            ("攻擊成功率 (ASR)", _rate_display(asr, sum_atk_tot), _rate_delta(asr, sum_atk_tot, la["asr"])),
            ("接發到位率 (GP%)", _rate_display(gp, sum_rcv_tot), _rate_delta(gp, sum_rcv_tot, la["gp"])),
            ("發球破壞率 (ACE%)", _rate_display(ace, sum_srv_tot), _rate_delta(ace, sum_srv_tot, la["ace"])),
            ("防守起球率 (DIG%)", _rate_display(dig_rate, sum_dig_tot), _rate_delta(dig_rate, sum_dig_tot, la["dig"])),
        ],
        "OP": [
            ("攻擊成功率 (ASR)", _rate_display(asr, sum_atk_tot), _rate_delta(asr, sum_atk_tot, la["asr"])),
            ("局均攔網 (BLK/Set)", f"{blk_per_set:.2f}", f"{blk_per_set - la['blk']:+.2f} vs 聯盟均"),
            ("發球破壞率 (ACE%)", _rate_display(ace, sum_srv_tot), _rate_delta(ace, sum_srv_tot, la["ace"])),
            ("防守起球率 (DIG%)", _rate_display(dig_rate, sum_dig_tot), _rate_delta(dig_rate, sum_dig_tot, la["dig"])),
        ],
        "MB": [
            ("攻擊成功率 (ASR)", _rate_display(asr, sum_atk_tot), _rate_delta(asr, sum_atk_tot, la["asr"])),
            ("局均攔網 (BLK/Set)", f"{blk_per_set:.2f}", f"{blk_per_set - la['blk']:+.2f} vs 聯盟均"),
            ("發球破壞率 (ACE%)", _rate_display(ace, sum_srv_tot), _rate_delta(ace, sum_srv_tot, la["ace"])),
            ("防守起球率 (DIG%)", _rate_display(dig_rate, sum_dig_tot), _rate_delta(dig_rate, sum_dig_tot, la["dig"])),
        ],
        "S": [
            ("舉球到位率 (SET%)", _rate_display(set_rate, sum_set_tot), _rate_delta(set_rate, sum_set_tot, la["set"])),
            ("攻擊成功率 (ASR)", _rate_display(asr, sum_atk_tot), _rate_delta(asr, sum_atk_tot, la["asr"])),
            ("發球破壞率 (ACE%)", _rate_display(ace, sum_srv_tot), _rate_delta(ace, sum_srv_tot, la["ace"])),
            ("防守起球率 (DIG%)", _rate_display(dig_rate, sum_dig_tot), _rate_delta(dig_rate, sum_dig_tot, la["dig"])),
        ],
        "L": [
            ("接發到位率 (GP%)", _rate_display(gp, sum_rcv_tot), _rate_delta(gp, sum_rcv_tot, la["gp"])),
            ("防守起球率 (DIG%)", _rate_display(dig_rate, sum_dig_tot), _rate_delta(dig_rate, sum_dig_tot, la["dig"])),
            ("綜合防守到位率 (DEF%)", _rate_display(def_rate, sum_rcv_tot + sum_dig_tot), _rate_delta(def_rate, sum_rcv_tot + sum_dig_tot, la["def"])),
            ("發球破壞率 (ACE%)", _rate_display(ace, sum_srv_tot), _rate_delta(ace, sum_srv_tot, la["ace"])),
        ],
    }
    default_kpi = [
        ("攻擊成功率 (ASR)", _rate_display(asr, sum_atk_tot), _rate_delta(asr, sum_atk_tot, la["asr"])),
        ("接發到位率 (GP%)", _rate_display(gp, sum_rcv_tot), _rate_delta(gp, sum_rcv_tot, la["gp"])),
        ("發球破壞率 (ACE%)", _rate_display(ace, sum_srv_tot), _rate_delta(ace, sum_srv_tot, la["ace"])),
        ("防守起球率 (DIG%)", _rate_display(dig_rate, sum_dig_tot), _rate_delta(dig_rate, sum_dig_tot, la["dig"])),
    ]
    kpi_items = POS_KPI_MAP.get(player_position, default_kpi)

    kpi_cols = st.columns(4)
    for col, (label, val, delta) in zip(kpi_cols, kpi_items):
        col.metric(label, val, delta=delta)

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("出賽場次", f"{n_games} 場")
    k6.metric("總出賽局數", f"{total_sets} 局")
    k7.metric("總得分", f"{total_points} 分")
    k8.metric("場均得分", f"{ppg:.1f} 分", delta=f"{ppg - la['ppg']:+.1f} 分 vs 聯盟均")

    st.markdown("---")

    # ── 同位置平均（供雷達圖對照） ─────────────────────────────
    pos_filter = "AND p.position = ?" if player_position else ""
    pos_params = (gender_code, player_position) if player_position else (gender_code,)
    lg = _parse_agg(_load_league_agg(gender_code, pos_filter, pos_params))

    # ── 雷達圖 ─────────────────────────────────────────────────
    _s_asr = _rate_safe(asr, sum_atk_tot)
    _s_gp = _rate_safe(gp, sum_rcv_tot)
    _s_ace = _rate_safe(ace, sum_srv_tot)
    _s_dig = _rate_safe(dig_rate, sum_dig_tot)
    _s_set = _rate_safe(set_rate, sum_set_tot)
    _s_def = _rate_safe(def_rate, sum_rcv_tot + sum_dig_tot)

    POS_RADAR_MAP = {
        "OH": [("攻擊成功率", _s_asr, lg["asr"], True), ("接發到位率", _s_gp, lg["gp"], True),
               ("防守起球率", _s_dig, lg["dig"], True), ("發球破壞率", _s_ace, lg["ace"], True),
               ("場均得分", ppg, lg["ppg"], False)],
        "OP": [("攻擊成功率", _s_asr, lg["asr"], True), ("局均攔網", blk_per_set, lg["blk"], False),
               ("發球破壞率", _s_ace, lg["ace"], True), ("防守起球率", _s_dig, lg["dig"], True),
               ("場均得分", ppg, lg["ppg"], False)],
        "MB": [("攻擊成功率", _s_asr, lg["asr"], True), ("局均攔網", blk_per_set, lg["blk"], False),
               ("發球破壞率", _s_ace, lg["ace"], True), ("防守起球率", _s_dig, lg["dig"], True),
               ("場均得分", ppg, lg["ppg"], False)],
        "S":  [("舉球到位率", _s_set, lg["set"], True), ("攻擊成功率", _s_asr, lg["asr"], True),
               ("防守起球率", _s_dig, lg["dig"], True), ("發球破壞率", _s_ace, lg["ace"], True),
               ("場均得分", ppg, lg["ppg"], False)],
        "L":  [("接發到位率", _s_gp, lg["gp"], True), ("防守起球率", _s_dig, lg["dig"], True),
               ("綜合防守到位率", _s_def, lg["def"], True), ("發球破壞率", _s_ace, lg["ace"], True),
               ("場均得分", ppg, lg["ppg"], False)],
    }
    default_radar = [
        ("攻擊成功率", _s_asr, lg["asr"], True), ("接發到位率", _s_gp, lg["gp"], True),
        ("發球破壞率", _s_ace, lg["ace"], True), ("防守起球率", _s_dig, lg["dig"], True),
        ("場均得分", ppg, lg["ppg"], False),
    ]
    radar_dims = POS_RADAR_MAP.get(player_position, default_radar)
    categories = [d[0] for d in radar_dims]
    player_vals = [d[1] for d in radar_dims]
    league_vals = [d[2] for d in radar_dims]
    is_pct = [d[3] for d in radar_dims]

    chart_left, chart_right = st.columns(2)

    with chart_left:
        avg_label = f"{gender} 同位置平均" if player_position else f"{gender}平均"
        st.subheader("多維度戰力雷達圖")

        player_norm, league_norm = [], []
        for pv, lv in zip(player_vals, league_vals):
            hi = max(pv, lv, 0.001)
            player_norm.append(pv / hi * 100)
            league_norm.append(lv / hi * 100)

        player_hover = [f"{c}: {v:.1f}{'%' if p else ''}" for c, v, p in zip(categories, player_vals, is_pct)]
        league_hover = [f"{c}: {v:.1f}{'%' if p else ''}" for c, v, p in zip(categories, league_vals, is_pct)]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=player_norm + [player_norm[0]], theta=categories + [categories[0]],
            fill="toself", name=player_name,
            text=player_hover + [player_hover[0]], hovertemplate="%{text}<extra></extra>",
            line=dict(color="#FF6B35", width=2.5), fillcolor="rgba(255, 107, 53, 0.25)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=league_norm + [league_norm[0]], theta=categories + [categories[0]],
            fill="toself", name=avg_label,
            text=league_hover + [league_hover[0]], hovertemplate="%{text}<extra></extra>",
            line=dict(color="#636EFA", width=1.5, dash="dash"), fillcolor="rgba(99, 110, 250, 0.10)",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 105], showticklabels=False)),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            height=480, margin=dict(l=60, r=60, t=40, b=60),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── 逐場趨勢圖 ────────────────────────────────────────────
    POS_TREND_MAP = {
        "OH": ("match_asr", "攻擊成功率 (%)", "attack_points", "attack_total"),
        "OP": ("match_asr", "攻擊成功率 (%)", "attack_points", "attack_total"),
        "MB": ("match_asr", "攻擊成功率 (%)", "attack_points", "attack_total"),
        "S":  ("match_set_pct", "舉球到位率 (%)", "set_excellent", "set_total"),
        "L":  ("match_def_pct", "綜合防守到位率 (%)", None, None),
    }
    POS_BAR_MAP = {
        "OH": ("total_points", "總得分"), "OP": ("total_points", "總得分"),
        "MB": ("block_points", "攔網得分"), "S": ("set_excellent", "舉球好球"),
        "L": ("dig_excellent", "防守好球"),
    }
    trend_info = POS_TREND_MAP.get(player_position, ("match_asr", "攻擊成功率 (%)", "attack_points", "attack_total"))
    trend_col, trend_label = trend_info[0], trend_info[1]
    bar_col, bar_label = POS_BAR_MAP.get(player_position, ("total_points", "總得分"))

    with chart_right:
        st.subheader(f"逐場趨勢：{trend_label} + {bar_label}")

        plot_df = stats_df.copy()
        if player_position == "L":
            plot_df[trend_col] = vec_pct(
                plot_df["receive_excellent"] + plot_df["dig_excellent"],
                plot_df["receive_total"] + plot_df["dig_total"],
            )
        else:
            num_col, den_col = trend_info[2], trend_info[3]
            plot_df[trend_col] = vec_pct(plot_df[num_col], plot_df[den_col])
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
            marker=dict(size=7, color="#FF6B35"), line=dict(color="#FF6B35", width=2),
            hovertemplate=f"{trend_label}: %{{y:.1f}}%<extra></extra>",
        ))
        fig_trend.update_layout(
            xaxis=dict(title="比賽", tickangle=-45),
            yaxis=dict(title=trend_label, side="left", rangemode="tozero"),
            yaxis2=dict(title=bar_label, side="right", overlaying="y", rangemode="tozero"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=480, margin=dict(l=50, r=50, t=50, b=80), bargap=0.3,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

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
            use_container_width=True, hide_index=True,
        )
