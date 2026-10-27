"""
球隊層級特徵與真實勝負標籤（Phase 4）。
訓練 script 與測試皆直接 import。
"""

import pandas as pd

from src.utils.constants import LEGACY_TEAM_IDS
from src.utils.logger import get_logger
from src.utils.match_status import has_result

logger = get_logger(__name__)


GAME_STAT_COLS = ["ASR", "GP_pct", "DIG_pct", "BLK_per_set", "ACE_pct"]

# total_sets：個人上場局數優先，為 NULL（官網 box score 來源）時退回該場賽事的實際局數。
_TEAM_MATCH_SQL = """
    WITH team_match AS (
        SELECT s.match_date,
               r.team_id,
               r.gender,
               s.opponent,
               a.team_id                AS opp_team_id,
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
               MAX(s.sets_played)       AS played_sets
        FROM player_match_stats s
        JOIN roster_registrations r ON s.registration_id = r.registration_id
        LEFT JOIN team_aliases a
               ON a.alias = TRIM(s.opponent) AND a.gender = r.gender AND a.source <> 'ext_stats'
        WHERE s.is_golden_set = 0
        GROUP BY s.match_date, r.team_id, r.gender, s.opponent, a.team_id
    )
    SELECT t.match_date, t.team_id, t.gender, t.opponent,
           t.atk_pts, t.atk_tot, t.blk_pts, t.srv_pts, t.srv_tot,
           t.rcv_exc, t.rcv_tot, t.dig_exc, t.dig_tot, t.total_points,
           COALESCE(t.played_sets, m.home_sets_won + m.away_sets_won) AS total_sets
    FROM team_match t
    LEFT JOIN matches m
           ON m.match_date = t.match_date
          AND m.gender = t.gender
          AND m.is_golden_set = 0
          AND ((m.home_team_id = t.team_id AND m.away_team_id = t.opp_team_id)
               OR (m.home_team_id = t.opp_team_id AND m.away_team_id = t.team_id))
"""


def _pct(num, den):
    return (num / den * 100).where(den > 0, 0.0)


_TEAM_MATCH_KEYS = ["match_date", "team_id", "gender", "opponent"]


def load_team_match_stats(conn) -> pd.DataFrame:
    df = pd.read_sql_query(_TEAM_MATCH_SQL, conn)
    dup = df.duplicated(_TEAM_MATCH_KEYS, keep=False)
    if dup.any():
        raise ValueError(f"同日同隊對同一對手有多場賽事，無法決定局數：\n{df.loc[dup, _TEAM_MATCH_KEYS].to_string(index=False)}")
    no_sets = df[df["total_sets"].isna()]
    if not no_sets.empty:
        logger.warning("%d 筆球隊單場統計無上場局數且對不回賽事，每局攔網以 0 計：\n%s",
                       len(no_sets), no_sets[_TEAM_MATCH_KEYS].to_string(index=False))
    df["ASR"] = _pct(df["atk_pts"], df["atk_tot"])
    df["GP_pct"] = _pct(df["rcv_exc"], df["rcv_tot"])
    df["DIG_pct"] = _pct(df["dig_exc"], df["dig_tot"])
    df["ACE_pct"] = _pct(df["srv_pts"], df["srv_tot"])
    df["BLK_per_set"] = (df["blk_pts"] / df["total_sets"]).where(df["total_sets"] > 0, 0.0)
    return df


def build_match_labels(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"legacy_skipped": 0, "invalid_skipped": 0, "unresolved_skipped": 0}
    rows = []
    for _, m in matches.iterrows():
        if m["is_golden_set"] == 1:
            continue
        if (not has_result(m["home_sets_won"], m["away_sets_won"])
                or m["home_sets_won"] == m["away_sets_won"]):
            report["invalid_skipped"] += 1
            continue
        if pd.isna(m["home_team_id"]) or pd.isna(m["away_team_id"]):
            report["unresolved_skipped"] += 1
            continue
        home_tid, away_tid = int(m["home_team_id"]), int(m["away_team_id"])
        if home_tid in LEGACY_TEAM_IDS or away_tid in LEGACY_TEAM_IDS:
            report["legacy_skipped"] += 1
            continue
        home_win = int(m["home_sets_won"] > m["away_sets_won"])
        rows.append((m["match_date"], home_tid, m["gender"], home_win))
        rows.append((m["match_date"], away_tid, m["gender"], 1 - home_win))
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


def load_labeled_history(conn) -> tuple[pd.DataFrame, dict]:
    """所有有技術統計且已有勝負的球隊單場列（訓練與賽前勝率共用的歷史）。"""
    team_match = load_team_match_stats(conn)
    matches = pd.read_sql_query("SELECT * FROM matches", conn)
    labels, report = build_match_labels(matches)
    labeled = attach_labels(team_match, labels)
    report["team_match_rows"] = len(labeled)
    return labeled, report


def latest_rolling_features(labeled: pd.DataFrame, team_id: int, gender: str,
                            cutoff_date: str) -> pd.Series | None:
    """
    給定球隊與截止日期，回傳該隊在 cutoff_date 當天賽前的滾動特徵向量；
    沒有任何 cutoff_date 之前的歷史則回 None。
    做法是在歷史後面接一列虛擬賽事再走 add_rolling_features，與訓練路徑同一套計算。
    """
    hist = labeled[(labeled["team_id"] == team_id) & (labeled["gender"] == gender)
                   & (labeled["match_date"] < cutoff_date)]
    if hist.empty:
        return None
    sentinel = pd.DataFrame([{"team_id": team_id, "gender": gender,
                              "match_date": cutoff_date, "opponent": None}])
    frame = add_rolling_features(pd.concat([hist, sentinel], ignore_index=True))
    return frame.iloc[-1][ROLLING_FEATURES]


def build_training_frame(conn) -> tuple[pd.DataFrame, dict]:
    labeled, report = load_labeled_history(conn)
    frame = add_rolling_features(labeled)
    report["training_rows"] = len(frame)
    return frame, report
