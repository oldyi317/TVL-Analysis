from pathlib import Path

import joblib
import numpy as np

from src.app.tabs.prediction import KNOWN_VERSIONS, SLIDER_CFG, _artifact_error
from src.models.features import ROLLING_FEATURES

PKL_PATH = Path(__file__).resolve().parents[1] / "src" / "models" / "match_predictor_v2.pkl"

REQUIRED_KEYS = {
    "model", "model_name", "version", "feature_cols", "feature_labels",
    "label_source", "best_params", "optuna_best_f1", "cv_f1_mean",
    "training_samples", "trained_at", "xgboost_version", "game_stat_cols",
}


def test_slider_cfg_covers_rolling_features_exactly():
    assert set(SLIDER_CFG) == set(ROLLING_FEATURES)


def test_artifact_error_accepts_v2():
    artifact = {"version": "v2", "feature_cols": ROLLING_FEATURES}
    assert _artifact_error(artifact) is None


def test_artifact_error_rejects_unknown_version():
    assert _artifact_error({"version": None, "feature_cols": []}) is not None
    assert _artifact_error({"version": "v1", "feature_cols": ROLLING_FEATURES}) is not None


def test_artifact_error_rejects_feature_mismatch():
    artifact = {"version": "v2", "feature_cols": ROLLING_FEATURES[:5]}
    assert _artifact_error(artifact) is not None
    artifact = {"version": "v2", "feature_cols": ROLLING_FEATURES + ["extra"]}
    assert _artifact_error(artifact) is not None


def test_shipped_pkl_matches_app_contract():
    artifact = joblib.load(PKL_PATH)
    assert REQUIRED_KEYS <= set(artifact)
    assert _artifact_error(artifact) is None
    X = np.array([[40.0] * 10 + [0.0]])
    proba = artifact["model"].predict_proba(X)
    assert proba.shape == (1, 2)


def test_old_v1_pkl_removed():
    assert not (PKL_PATH.parent / "match_predictor.pkl").exists()
