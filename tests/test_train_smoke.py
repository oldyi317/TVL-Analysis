import numpy as np
import pandas as pd

from src.models.features import GAME_STAT_COLS, ROLLING_FEATURES
from src.models.train import tune_and_train

REQUIRED_KEYS = {
    "model", "model_name", "version", "feature_cols", "feature_labels",
    "label_source", "best_params", "optuna_best_f1", "cv_f1_mean",
    "training_samples", "trained_at", "xgboost_version", "game_stat_cols",
}


def _fake_frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.uniform(20, 60, n) for f in ROLLING_FEATURES})
    df["win_streak"] = rng.integers(-3, 4, n)
    df["match_date"] = pd.date_range("2026-01-01", periods=n).astype(str)
    df["team_id"] = rng.integers(1, 5, n)
    df["win"] = rng.integers(0, 2, n)
    return df


def test_tune_and_train_artifact_contract():
    artifact = tune_and_train(_fake_frame(), n_trials=2)
    assert REQUIRED_KEYS <= set(artifact)
    assert artifact["version"] == "v2"
    assert artifact["label_source"] == "matches.sets_won"
    assert artifact["feature_cols"] == ROLLING_FEATURES
    assert artifact["game_stat_cols"] == GAME_STAT_COLS
    assert artifact["training_samples"] == 40
    proba = artifact["model"].predict_proba(
        _fake_frame(seed=1)[ROLLING_FEATURES].values[:3])
    assert proba.shape == (3, 2)
