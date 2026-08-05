"""
驗證套件版本升級後 src/models/match_predictor.pkl 仍可正確載入與預測。
對應 PCAI 搬遷 spec §4e：釘版本前必須實測模型可載入。
"""

from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "src" / "models" / "match_predictor.pkl"


def test_model_pkl_loads_and_predicts():
    assert MODEL_PATH.exists(), f"模型檔案不存在：{MODEL_PATH}"
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    X = np.zeros((1, len(feature_cols)))
    proba = model.predict_proba(X)

    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(), 1.0)


def test_model_shap_explainer_works():
    import shap

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    explainer = shap.TreeExplainer(model)
    X = np.zeros((1, len(feature_cols)))
    shap_values = explainer.shap_values(X)

    assert shap_values is not None
    assert np.array(shap_values).shape[-1] == len(feature_cols)
