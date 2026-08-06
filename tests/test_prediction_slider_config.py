"""
迴歸測試：修正前的 bug 讀取 artifact["feature_names"]（不存在的鍵），導致 n_features
恆為 0、恆定判定為 V1。確保現在正確讀取 artifact["feature_cols"]。
"""

from src.app.tabs.prediction import V1_SLIDER_CFG, V2_SLIDER_CFG, _select_slider_config


def test_select_slider_config_reads_feature_cols_key():
    artifact = {"feature_cols": ["ASR", "GP_pct", "DIG_pct", "BLK_per_set", "ACE_pct"]}
    cfg, label, n = _select_slider_config(artifact)
    assert n == 5
    assert cfg == V1_SLIDER_CFG
    assert "V1" in label


def test_select_slider_config_detects_v2_11_features():
    artifact = {"feature_cols": [f"f{i}" for i in range(11)]}
    cfg, label, n = _select_slider_config(artifact)
    assert n == 11
    assert cfg == V2_SLIDER_CFG
    assert "V2" in label


def test_select_slider_config_ignores_wrong_key_name():
    # 即使 artifact 內誤留有 "feature_names" 鍵，也必須以 "feature_cols" 為準
    artifact = {
        "feature_names": [],
        "feature_cols": [f"f{i}" for i in range(11)],
    }
    cfg, label, n = _select_slider_config(artifact)
    assert n == 11
    assert cfg == V2_SLIDER_CFG


def test_select_slider_config_defaults_to_v1_when_feature_cols_missing():
    cfg, label, n = _select_slider_config({})
    assert n == 0
    assert cfg == V1_SLIDER_CFG
    assert "V1" in label


def test_real_pkl_resolves_to_v1_slider_config():
    """驗證真實模型 pkl 的 feature_cols 鍵名與數量，捕捉未來的模型漂移。"""
    import joblib
    from src.app.helpers import MODEL_PATH

    artifact = joblib.load(MODEL_PATH)
    cfg, version, n_features = _select_slider_config(artifact)
    assert n_features == 5
    assert cfg is V1_SLIDER_CFG
    assert "V1" in version


# ─────────────────────────────────────────────────────────────────────

import pandas as pd
from sqlalchemy import text

from src.etl.db_loader import insert_players, insert_teams
from src.etl.stats_crawler import upsert_stats


def _seed_league_data_for_ranges(engine, season: str) -> None:
    """插入一筆 sets_played=5（剛好達到 get_league_aggregated_stats 的 HAVING 門檻）的球員數據。"""
    df = pd.DataFrame([{
        "team_id": 1, "team_name": "屏東台電", "gender": "M",
        "jersey_number": 4, "name": "李元", "position": "OH",
        "dob": None, "height_cm": None, "weight_kg": None,
    }])
    insert_teams(engine, df)
    insert_players(engine, df, season=season)
    with engine.begin() as conn:
        pid = conn.execute(
            text("SELECT player_id FROM players WHERE name = '李元' AND season = :s"),
            {"s": season},
        ).scalar_one()
    row = dict(
        match_date="2026-01-05", opponent="雲林美津濃", sets_played=5,
        attack_total=20, attack_points=10, block_points=2,
        serve_total=10, serve_points=2, receive_total=10, receive_excellent=6,
        dig_total=10, dig_excellent=4, set_total=0, set_excellent=0,
        total_points=14, is_golden_set=0,
    )
    upsert_stats(engine, pid, [row], season)


def test_get_data_ranges_does_not_silently_swallow_season_mismatch(sqlite_engine):
    """
    迴歸測試：_get_data_ranges 若沒有同步傳入 season，get_league_aggregated_stats 會拋
    TypeError，且該 TypeError 會被 _get_data_ranges 的 bare except 吞掉、回傳 {}。
    這裡斷言回傳值「非空」，確保呼叫端簽名確實同步更新，而不是被 except 悄悄蓋過去。
    """
    from src.app.tabs.prediction import _get_data_ranges

    _seed_league_data_for_ranges(sqlite_engine, season="2025-26")
    _get_data_ranges.clear()

    ranges = _get_data_ranges("M", "2025-26")

    assert ranges != {}, "_get_data_ranges 回傳空字典——很可能是呼叫端與 season 簽名不同步，TypeError 被 bare except 吞掉"
    assert "ASR" in ranges
