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


def build_match_labels(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"legacy_skipped": 0, "invalid_skipped": 0}
    rows = []
    for _, m in matches.iterrows():
        if m["is_golden_set"] == 1:
            continue
        home = normalize_match_team(m["home_team"])
        away = normalize_match_team(m["away_team"])
        if home in LEGACY_TEAMS or away in LEGACY_TEAMS:
            report["legacy_skipped"] += 1
            continue
        if (pd.isna(m["home_sets_won"]) or pd.isna(m["away_sets_won"])
                or m["home_sets_won"] == m["away_sets_won"]):
            report["invalid_skipped"] += 1
            continue
        home_tid, home_g = OPP_SHORT_TO_TEAM[home]
        away_tid, away_g = OPP_SHORT_TO_TEAM[away]
        if home_g != m["gender"] or away_g != m["gender"]:
            raise ValueError(
                f"隊伍性別對不上 matches.gender：{m['home_team']} vs {m['away_team']}（{m['gender']}）")
        home_win = int(m["home_sets_won"] > m["away_sets_won"])
        rows.append((m["match_date"], home_tid, home_g, home_win))
        rows.append((m["match_date"], away_tid, away_g, 1 - home_win))
    labels = pd.DataFrame(rows, columns=["match_date", "team_id", "gender", "win"])
    dup = labels.duplicated(["match_date", "team_id", "gender"], keep=False)
    if dup.any():
        raise ValueError(f"同日同隊出現多筆標籤：\n{labels[dup].to_string(index=False)}")
    return labels, report


def attach_labels(team_match: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = team_match.merge(labels, on=["match_date", "team_id", "gender"], how="left")
    missing = merged[merged["win"].isna()]
    if not missing.empty:
        detail = missing[["match_date", "team_id", "gender", "opponent"]].to_string(index=False)
        raise ValueError(f"{len(missing)} 筆球隊單場統計找不到對應比分：\n{detail}")
    merged["win"] = merged["win"].astype(int)
    return merged


ROLLING_FEATURES = (
    [f"{c}_roll3" for c in GAME_STAT_COLS]
    + [f"{c}_roll5" for c in GAME_STAT_COLS]
    + ["win_streak"]
)

_GROUP_KEYS = ["team_id", "gender"]


def compute_win_streak(wins: pd.Series) -> list[int]:
    streaks, current = [], 0
    for w in wins:
        streaks.append(current)
        if w == 1:
            current = current + 1 if current > 0 else 1
        else:
            current = current - 1 if current < 0 else -1
    return streaks


def add_rolling_features(labeled: pd.DataFrame) -> pd.DataFrame:
    df = (labeled.sort_values(_GROUP_KEYS + ["match_date", "opponent"])
          .reset_index(drop=True))
    gkey = df[_GROUP_KEYS].apply(tuple, axis=1)
    for col in GAME_STAT_COLS:
        shifted = df.groupby(_GROUP_KEYS)[col].shift(1)
        df[f"{col}_roll3"] = shifted.groupby(gkey).transform(
            lambda x: x.rolling(3, min_periods=1).mean())
        df[f"{col}_roll5"] = shifted.groupby(gkey).transform(
            lambda x: x.rolling(5, min_periods=1).mean())
    df["win_streak"] = df.groupby(_GROUP_KEYS)["win"].transform(compute_win_streak)
    return (df.dropna(subset=[f"{GAME_STAT_COLS[0]}_roll3"])
            .reset_index(drop=True))


def build_training_frame(conn) -> tuple[pd.DataFrame, dict]:
    team_match = load_team_match_stats(conn)
    matches = pd.read_sql_query("SELECT * FROM matches", conn)
    labels, report = build_match_labels(matches)
    labeled = attach_labels(team_match, labels)
    report["team_match_rows"] = len(labeled)
    frame = add_rolling_features(labeled)
    report["training_rows"] = len(frame)
    return frame, report
