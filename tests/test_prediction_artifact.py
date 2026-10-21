from pathlib import Path

import joblib
import numpy as np

from src.models.features import ROLLING_FEATURES

PKL_PATH = Path(__file__).resolve().parents[1] / "src" / "models" / "match_predictor_v2.pkl"

REQUIRED_KEYS = {
    "model", "model_name", "version", "feature_cols", "feature_labels",
    "label_source", "best_params", "optuna_best_f1", "cv_f1_mean",
    "training_samples", "trained_at", "xgboost_version", "game_stat_cols",
}


def test_shipped_pkl_matches_train_contract():
    artifact = joblib.load(PKL_PATH)
    assert REQUIRED_KEYS <= set(artifact)
    assert artifact["version"] == "v2"
    assert artifact["feature_cols"] == ROLLING_FEATURES
    X = np.array([[40.0] * 10 + [0.0]])
    proba = artifact["model"].predict_proba(X)
    assert proba.shape == (1, 2)


def test_old_v1_pkl_removed():
    assert not (PKL_PATH.parent / "match_predictor.pkl").exists()
