"""
賽果預測模型訓練（v2）：TimeSeriesSplit + Optuna + XGBoost。
用法：python -m src.models.train [--trials 50] [--output src/models/match_predictor_v2.pkl]
"""

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

from src.models.features import GAME_STAT_COLS, ROLLING_FEATURES, build_training_frame
from src.utils.db_config import get_connection

SEED = 42
N_SPLITS = 5

ROLLING_LABELS = {
    "ASR_roll3": "近3場 攻擊率", "ASR_roll5": "近5場 攻擊率",
    "GP_pct_roll3": "近3場 接發率", "GP_pct_roll5": "近5場 接發率",
    "DIG_pct_roll3": "近3場 防守率", "DIG_pct_roll5": "近5場 防守率",
    "BLK_per_set_roll3": "近3場 局均攔網", "BLK_per_set_roll5": "近5場 局均攔網",
    "ACE_pct_roll3": "近3場 發球率", "ACE_pct_roll5": "近5場 發球率",
    "win_streak": "連勝/連敗",
}


def tune_and_train(frame, n_trials, seed=SEED) -> dict:
    df_ts = frame.sort_values(["match_date", "team_id"]).reset_index(drop=True)
    X_all = df_ts[ROLLING_FEATURES].values
    y_all = df_ts["win"].values
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": seed,
            "eval_metric": "logloss",
        }
        model = xgb.XGBClassifier(**params)
        scores = []
        for train_idx, test_idx in tscv.split(X_all):
            model.fit(X_all[train_idx], y_all[train_idx])
            scores.append(f1_score(y_all[test_idx], model.predict(X_all[test_idx]),
                                   zero_division=0))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best_params = {**study.best_params, "random_state": seed, "eval_metric": "logloss"}
    cv_scores = []
    model = xgb.XGBClassifier(**best_params)
    for train_idx, test_idx in tscv.split(X_all):
        model.fit(X_all[train_idx], y_all[train_idx])
        cv_scores.append(f1_score(y_all[test_idx], model.predict(X_all[test_idx]),
                                  zero_division=0))
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_all, y_all)

    return {
        "model": final_model,
        "model_name": "XGBoost (Optuna-tuned, v2)",
        "version": "v2",
        "feature_cols": ROLLING_FEATURES,
        "feature_labels": [ROLLING_LABELS[c] for c in ROLLING_FEATURES],
        "label_source": "matches.sets_won",
        "best_params": best_params,
        "optuna_best_f1": float(study.best_value),
        "cv_f1_mean": float(np.mean(cv_scores)),
        "training_samples": len(df_ts),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "xgboost_version": xgb.__version__,
        "game_stat_cols": GAME_STAT_COLS,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="重訓賽果預測模型（v2）")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--output", type=Path,
                        default=Path("src/models/match_predictor_v2.pkl"))
    args = parser.parse_args(argv)

    conn = get_connection()
    try:
        frame, report = build_training_frame(conn)
    finally:
        conn.close()
    print(f"資料報告：{report}")

    artifact = tune_and_train(frame, n_trials=args.trials)
    joblib.dump(artifact, args.output)
    print(f"樣本數：{artifact['training_samples']}｜"
          f"Optuna 最佳 F1：{artifact['optuna_best_f1']:.3f}｜"
          f"最佳參數重驗 cv F1：{artifact['cv_f1_mean']:.3f}")
    print(f"已輸出：{args.output}")


if __name__ == "__main__":
    main()
