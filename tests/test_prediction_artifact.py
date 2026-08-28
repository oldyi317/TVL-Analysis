from src.app.tabs.prediction import KNOWN_VERSIONS, SLIDER_CFG, _artifact_error
from src.models.features import ROLLING_FEATURES


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
