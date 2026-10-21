"""
賽前勝率計算（ADR 0003）：對所有未開打賽事以 match_predictor_v2.pkl 算主隊勝率，upsert 進 match_predictions。
用法：python -m src.models.predict [--model src/models/match_predictor_v2.pkl]

模型是單隊視角（給一隊的近況特徵 → 該隊勝率），主隊勝率取兩隊各自勝率的正規化：p_home / (p_home + p_away)，
兩隊相加恰為 1。未開打 = 各局比分皆 NULL；已開打賽事不在查詢範圍內，既有預測自然凍結。
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from src.models.features import latest_rolling_features, load_labeled_history
from src.utils.db_config import PROJECT_ROOT, get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = PROJECT_ROOT / "src" / "models" / "match_predictor_v2.pkl"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"

_SET_COLS = [f"{side}_set{i}" for side in ("home", "away") for i in range(1, 6)]
_UNPLAYED_SQL = f"""
    SELECT match_id, match_date, gender, home_team, away_team, home_team_id, away_team_id
    FROM matches
    WHERE is_golden_set = 0 AND {" AND ".join(f"{c} IS NULL" for c in _SET_COLS)}
    ORDER BY match_date, match_id
"""
_UPSERT_SQL = """
    INSERT INTO match_predictions (match_id, model_version, home_win_prob, computed_at)
    VALUES (?, ?, ?, ?)
    ON CONFLICT (match_id, model_version) DO UPDATE
    SET home_win_prob = excluded.home_win_prob, computed_at = excluded.computed_at
    WHERE home_win_prob IS NOT excluded.home_win_prob
"""


def _team_win_prob(model, vec, feature_cols) -> float:
    X = np.array([[float(vec[c]) for c in feature_cols]])
    return float(model.predict_proba(X)[0][1])


def normalize_pair(p_home: float, p_away: float) -> float:
    total = p_home + p_away
    return 0.5 if total == 0 else p_home / total


def run_predictions(conn: sqlite3.Connection, artifact: dict) -> dict:
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    labeled, _ = load_labeled_history(conn)
    model, version, feature_cols = artifact["model"], artifact["version"], artifact["feature_cols"]
    computed_at = datetime.now().isoformat(timespec="seconds")
    report = {"unplayed": 0, "written": 0, "skipped_no_history": 0}

    for match_id, date, gender, home, away, home_id, away_id in conn.execute(_UNPLAYED_SQL).fetchall():
        report["unplayed"] += 1
        home_vec = latest_rolling_features(labeled, home_id, gender, date) if home_id else None
        away_vec = latest_rolling_features(labeled, away_id, gender, date) if away_id else None
        if home_vec is None or away_vec is None:
            report["skipped_no_history"] += 1
            logger.info("略過 %s %s vs %s：至少一隊無歷史技術統計", date, home, away)
            continue
        prob = normalize_pair(_team_win_prob(model, home_vec, feature_cols),
                              _team_win_prob(model, away_vec, feature_cols))
        conn.execute(_UPSERT_SQL, (match_id, version, prob, computed_at))
        report["written"] += 1
    conn.commit()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="計算未開打賽事的主隊賽前勝率")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args(argv)

    artifact = joblib.load(args.model)
    conn = get_connection()
    try:
        report = run_predictions(conn, artifact)
    finally:
        conn.close()
    logger.info("賽前勝率（%s）：未開打 %d 場，寫入 %d，無歷史略過 %d",
                artifact["version"], report["unplayed"], report["written"], report["skipped_no_history"])


if __name__ == "__main__":
    main()
