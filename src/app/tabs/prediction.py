"""
Tab 5：賽果預測 (ML Match Prediction)
使用訓練好的模型預測勝率，並以 SHAP 解釋特徵貢獻。
"""

import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from src.app.helpers import MODEL_PATH, load_data, vec_pct


# ---------------------------------------------------------------------------
# 從資料庫取得各指標的實際範圍
# ---------------------------------------------------------------------------
@st.cache_data
def _get_data_ranges(gender_code: str) -> dict[str, tuple[float, float]]:
    """從聯盟聚合數據計算各滑桿指標的實際 (min, max)，加 10% 緩衝。"""
    try:
        from src.app.helpers import get_league_aggregated_stats
        df = get_league_aggregated_stats(gender_code)
        if df.empty:
            return {}

        def _range(col):
            lo, hi = float(df[col].min()), float(df[col].max())
            buf = (hi - lo) * 0.1 if hi > lo else 1.0
            return (round(max(0, lo - buf), 1), round(hi + buf, 1))

        ranges = {}
        if "asr" in df.columns:
            ranges["ASR"] = _range("asr")
            ranges["ASR_roll3"] = _range("asr")
            ranges["ASR_roll5"] = _range("asr")
        if "gp_pct" in df.columns:
            ranges["GP_pct"] = _range("gp_pct")
            ranges["GP_pct_roll3"] = _range("gp_pct")
            ranges["GP_pct_roll5"] = _range("gp_pct")
        if "dig_pct" in df.columns:
            ranges["DIG_pct"] = _range("dig_pct")
            ranges["DIG_pct_roll3"] = _range("dig_pct")
            ranges["DIG_pct_roll5"] = _range("dig_pct")
        if "blk_per_set" in df.columns:
            ranges["BLK_per_set"] = _range("blk_per_set")
            ranges["BLK_per_set_roll3"] = _range("blk_per_set")
            ranges["BLK_per_set_roll5"] = _range("blk_per_set")
        if "ace_pct" in df.columns:
            ranges["ACE_pct"] = _range("ace_pct")
            ranges["ACE_pct_roll3"] = _range("ace_pct")
            ranges["ACE_pct_roll5"] = _range("ace_pct")
        return ranges
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Slider 設定：V1 (5 特徵) / V2 (11 特徵)
# ---------------------------------------------------------------------------
V1_SLIDER_CFG = [
    ("ASR",         "攻擊成功率 (ASR %)",     30.0, 65.0, 42.0, 0.5),
    ("GP_pct",      "接發到位率 (GP %)",       20.0, 80.0, 50.0, 0.5),
    ("DIG_pct",     "防守起球率 (DIG %)",      10.0, 75.0, 32.0, 0.5),
    ("BLK_per_set", "局均攔網 (BLK/Set)",       0.0,  5.0,  1.8, 0.1),
    ("ACE_pct",     "發球破壞率 (ACE %)",       0.0, 18.0,  4.0, 0.5),
]

V2_SLIDER_CFG = [
    ("ASR_roll3",         "近3場 攻擊率 (%)",       25.0, 65.0, 42.0, 0.5),
    ("ASR_roll5",         "近5場 攻擊率 (%)",       25.0, 65.0, 42.0, 0.5),
    ("GP_pct_roll3",      "近3場 接發率 (%)",       15.0, 85.0, 50.0, 0.5),
    ("GP_pct_roll5",      "近5場 接發率 (%)",       15.0, 85.0, 50.0, 0.5),
    ("DIG_pct_roll3",     "近3場 防守率 (%)",        5.0, 75.0, 32.0, 0.5),
    ("DIG_pct_roll5",     "近5場 防守率 (%)",        5.0, 75.0, 32.0, 0.5),
    ("BLK_per_set_roll3", "近3場 局均攔網",          0.0,  5.0,  1.8, 0.1),
    ("BLK_per_set_roll5", "近5場 局均攔網",          0.0,  5.0,  1.8, 0.1),
    ("ACE_pct_roll3",     "近3場 發球率 (%)",        0.0, 18.0,  4.0, 0.5),
    ("ACE_pct_roll5",     "近5場 發球率 (%)",        0.0, 18.0,  4.0, 0.5),
    ("win_streak",        "連勝/連敗 (正=連勝)",    -8.0,  8.0,  0.0, 1.0),
]

FEAT_LABELS_MAP = {
    "ASR": "攻擊成功率", "GP_pct": "接發到位率",
    "DIG_pct": "防守起球率", "BLK_per_set": "局均攔網",
    "ACE_pct": "發球破壞率",
    "ASR_roll3": "近3場攻擊率", "ASR_roll5": "近5場攻擊率",
    "GP_pct_roll3": "近3場接發率", "GP_pct_roll5": "近5場接發率",
    "DIG_pct_roll3": "近3場防守率", "DIG_pct_roll5": "近5場防守率",
    "BLK_per_set_roll3": "近3場局均攔網", "BLK_per_set_roll5": "近5場局均攔網",
    "ACE_pct_roll3": "近3場發球率", "ACE_pct_roll5": "近5場發球率",
    "win_streak": "連勝/連敗",
}


# ---------------------------------------------------------------------------
# 模型載入（快取）
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_model_and_explainer():
    import joblib
    import shap

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    explainer = shap.TreeExplainer(model)
    return artifact, model, explainer


# ---------------------------------------------------------------------------
# 主渲染函數
# ---------------------------------------------------------------------------
def render(ctx, cjk_font_path=None, cjk_font_stack=None):
    """繪製 Tab 5 — 賽果預測頁面。"""

    st.header("賽果預測 (ML Match Prediction)")

    # ------ 檢查模型檔案是否存在 ------
    if not MODEL_PATH.exists():
        st.info("尚未訓練模型，請先執行模型訓練流程以產生模型檔案。")
        return

    # ------ 載入模型與 SHAP 解釋器 ------
    artifact, model, explainer = _load_model_and_explainer()
    feature_names = artifact.get("feature_names", [])
    n_features = len(feature_names)

    # 根據特徵數量決定使用哪組 Slider
    if n_features == 11:
        slider_cfg = V2_SLIDER_CFG
        version_label = "V2（滑動窗口 + 連勝）"
    else:
        slider_cfg = V1_SLIDER_CFG
        version_label = "V1（基本五指標）"

    st.caption(f"模型版本：{version_label}｜特徵數：{n_features}")

    # ------ 動態 Slider UI（從實際資料取得範圍） ------
    st.subheader("調整比賽指標")

    # 嘗試從資料庫取得各指標的實際範圍，給予 10% 緩衝
    _data_ranges = _get_data_ranges(ctx.get("gender_code", "M"))

    input_values = {}
    cols = st.columns(2)
    for idx, (key, label, min_v, max_v, default_v, step) in enumerate(slider_cfg):
        # 若有實際資料範圍，使用資料驅動的範圍
        if key in _data_ranges:
            d_min, d_max = _data_ranges[key]
            min_v = min(min_v, d_min)
            max_v = max(max_v, d_max)
        # 對齊 step 並確保 default 在範圍內
        import math
        min_v = round(math.floor(min_v / step) * step, 4)
        max_v = round(math.ceil(max_v / step) * step, 4)
        default_v = round(round(default_v / step) * step, 4)
        default_v = float(max(min_v, min(default_v, max_v)))
        min_v, max_v, step = float(min_v), float(max_v), float(step)
        with cols[idx % 2]:
            input_values[key] = st.slider(
                label, min_value=min_v, max_value=max_v,
                value=default_v, step=step, key=f"pred_{key}",
            )

    # ------ 組裝特徵向量並預測 ------
    X = np.array([[input_values[k] for k, *_ in slider_cfg]])
    prob = model.predict_proba(X)[0]  # [P(lose), P(win)]
    win_prob = prob[1] * 100

    # ------ 顯示預測結果 ------
    st.divider()
    st.subheader("預測結果")

    if win_prob >= 60:
        color, icon, verdict = "green", "▲", "勝面較大"
    elif win_prob >= 40:
        color, icon, verdict = "orange", "●", "勝負五五開"
    else:
        color, icon, verdict = "red", "▼", "勝面較小"

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.metric("預測勝率", f"{win_prob:.1f}%")
    with col_right:
        st.markdown(
            f"<h3 style='color:{color};'>{icon} {verdict}</h3>",
            unsafe_allow_html=True,
        )

    # ------ SHAP 瀑布圖 ------
    st.divider()
    if st.button("顯示 SHAP 特徵貢獻圖", key="btn_shap"):
        import shap

        shap_values = explainer.shap_values(X)

        # 若為二分類，取 class 1 的 SHAP 值
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        # 建立 SHAP Explanation 物件（附帶中文特徵名稱）
        display_names = [
            FEAT_LABELS_MAP.get(k, k) for k, *_ in slider_cfg
        ]
        explanation = shap.Explanation(
            values=sv,
            base_values=(
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            ),
            data=X[0],
            feature_names=display_names,
        )

        # 繪製瀑布圖
        fig_shap, ax_shap = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(explanation, show=False)
        fig_shap = plt.gcf()

        # 套用 CJK 字型至所有文字物件
        _cjk_fp = (
            FontProperties(fname=cjk_font_path)
            if cjk_font_path
            else FontProperties(family=cjk_font_stack or [])
        )
        for text_obj in fig_shap.findobj(matplotlib.text.Text):
            text_obj.set_fontproperties(_cjk_fp)

        st.pyplot(fig_shap)
        plt.close(fig_shap)
