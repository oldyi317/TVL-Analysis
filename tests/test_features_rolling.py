import pandas as pd
import pytest

from src.models.features import (
    GAME_STAT_COLS, ROLLING_FEATURES, add_rolling_features, compute_win_streak,
)


def _labeled_frame():
    rows = []
    for i, (asr, win) in enumerate([(40.0, 1), (50.0, 1), (60.0, 0), (30.0, 1)]):
        row = {"match_date": f"2026-01-{10 + i:02d}", "team_id": 1, "gender": "M",
               "opponent": "獅子王", "win": win,
               "ASR": asr, "GP_pct": 50.0, "DIG_pct": 30.0,
               "BLK_per_set": 1.5, "ACE_pct": 5.0}
        rows.append(row)
    return pd.DataFrame(rows)


def test_rolling_features_order_and_count():
    assert len(ROLLING_FEATURES) == 11
    assert ROLLING_FEATURES[:5] == [f"{c}_roll3" for c in GAME_STAT_COLS]
    assert ROLLING_FEATURES[5:10] == [f"{c}_roll5" for c in GAME_STAT_COLS]
    assert ROLLING_FEATURES[10] == "win_streak"


def test_first_match_dropped_and_shift_excludes_current():
    df = add_rolling_features(_labeled_frame())
    assert len(df) == 3  # 首場無歷史被丟
    # 第二場的 roll3 只含第一場的 ASR=40，不含當場 50
    assert df.iloc[0]["ASR_roll3"] == pytest.approx(40.0)
    # 第四場 roll3 = mean(40, 50, 60)
    assert df.iloc[2]["ASR_roll3"] == pytest.approx(50.0)


def test_win_streak_is_pregame_state():
    wins = pd.Series([1, 1, 0, 0, 1])
    assert compute_win_streak(wins) == [0, 1, 2, -1, -2]


def test_win_streak_from_real_labels():
    df = add_rolling_features(_labeled_frame())
    # 賽前狀態：第二場前連勝1、第三場前連勝2、第四場前連敗1
    assert df["win_streak"].tolist() == [1, 2, -1]
