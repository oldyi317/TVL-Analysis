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
