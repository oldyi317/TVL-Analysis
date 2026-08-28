"""
球隊層級特徵與真實勝負標籤（Phase 4）。
不依賴 streamlit，訓練 script 與測試皆直接 import。
"""

import pandas as pd

from src.utils.constants import LEGACY_TEAMS, MATCH_TEAM_ALIASES, OPP_SHORT_TO_TEAM


def normalize_match_team(name: str) -> str:
    short = MATCH_TEAM_ALIASES.get(name, name)
    if short in OPP_SHORT_TO_TEAM or short in LEGACY_TEAMS:
        return short
    raise ValueError(f"未知隊名：{name}（請補 constants.MATCH_TEAM_ALIASES）")


GAME_STAT_COLS = ["ASR", "GP_pct", "DIG_pct", "BLK_per_set", "ACE_pct"]

_TEAM_MATCH_SQL = """
    SELECT s.match_date,
           r.team_id,
           r.gender,
           s.opponent,
           SUM(s.attack_points)     AS atk_pts,
           SUM(s.attack_total)      AS atk_tot,
           SUM(s.block_points)      AS blk_pts,
           SUM(s.serve_points)      AS srv_pts,
           SUM(s.serve_total)       AS srv_tot,
           SUM(s.receive_excellent) AS rcv_exc,
           SUM(s.receive_total)     AS rcv_tot,
           SUM(s.dig_excellent)     AS dig_exc,
           SUM(s.dig_total)         AS dig_tot,
           SUM(s.total_points)      AS total_points,
           MAX(s.sets_played)       AS total_sets
    FROM player_match_stats s
    JOIN roster_registrations r ON s.registration_id = r.registration_id
    WHERE s.is_golden_set = 0
    GROUP BY s.match_date, r.team_id, r.gender, s.opponent
"""


def _pct(num, den):
    return (num / den * 100).where(den > 0, 0.0)


def load_team_match_stats(conn) -> pd.DataFrame:
    df = pd.read_sql_query(_TEAM_MATCH_SQL, conn)
    df["ASR"] = _pct(df["atk_pts"], df["atk_tot"])
    df["GP_pct"] = _pct(df["rcv_exc"], df["rcv_tot"])
    df["DIG_pct"] = _pct(df["dig_exc"], df["dig_tot"])
    df["ACE_pct"] = _pct(df["srv_pts"], df["srv_tot"])
    df["BLK_per_set"] = (df["blk_pts"] / df["total_sets"]).where(df["total_sets"] > 0, 0.0)
    return df
