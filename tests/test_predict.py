import numpy as np
import pandas as pd
import pytest

from src.models.features import (
    ROLLING_FEATURES, add_rolling_features, build_match_labels, latest_rolling_features,
    load_labeled_history,
)
from src.models.predict import run_predictions
from src.utils.constants import OPP_SHORT_TO_TEAM
from tests.conftest import connect_seeded

MATCH_COLS = (
    "game_id, gender, cup_id, match_date, is_golden_set, home_team, away_team, "
    "home_team_id, away_team_id, home_sets_won, away_sets_won, "
    "home_set1, home_set2, home_set3, home_set4, away_set1, away_set2, away_set3, away_set4"
)


def _match(conn, game_id, home, away, sets, *, date, gender="M", cup=22):
    def tid(name):
        return OPP_SHORT_TO_TEAM[name][0] if name in OPP_SHORT_TO_TEAM else None
    hs = [s[0] for s in sets] + [None] * (4 - len(sets))
    aw = [s[1] for s in sets] + [None] * (4 - len(sets))
    hsw = sum(h > a for h, a in sets) if sets else None
    asw = sum(a > h for h, a in sets) if sets else None
    cur = conn.execute(
        f"INSERT INTO matches ({MATCH_COLS}) VALUES ({', '.join('?' * 19)})",
        (game_id, gender, cup, date, 0, home, away, tid(home), tid(away), hsw, asw, *hs, *aw))
    return cur.lastrowid


def _stat(conn, team_id, gender, date, opponent, atk_pts, atk_tot=100):
    pid = conn.execute("INSERT INTO players (name, gender) VALUES (?, ?)",
                       (f"p{team_id}{date}", gender)).lastrowid
    rid = conn.execute(
        """INSERT INTO roster_registrations (player_id, team_id, gender, cup_id, week_label, source)
           VALUES (?, ?, ?, 21, ?, 'match_page')""", (pid, team_id, gender, f"w{date}")).lastrowid
    conn.execute(
        """INSERT INTO player_match_stats
           (registration_id, match_date, opponent, sets_played, attack_total, attack_points,
            block_points, serve_total, serve_points, receive_total, receive_excellent,
            dig_total, dig_excellent, set_total, set_excellent, total_points, is_golden_set)
           VALUES (?, ?, ?, 3, ?, ?, 2, 10, 1, 10, 5, 10, 3, 10, 5, ?, 0)""",
        (rid, date, opponent, atk_tot, atk_pts, atk_pts + 3))


def _seed_history(conn):
    """屏東台電(1,M) 與 雲林美津濃(2,M) 互打兩場，各有兩筆技術統計。"""
    _match(conn, 1, "屏東台電", "雲林美津濃", [(25, 20)] * 3 + [(20, 25)], date="2026-01-10")
    _match(conn, 2, "雲林美津濃", "屏東台電", [(25, 20)] * 3, date="2026-01-17")
    _stat(conn, 1, "M", "2026-01-10", "雲林美津濃", atk_pts=50)
    _stat(conn, 2, "M", "2026-01-10", "屏東台電", atk_pts=30)
    _stat(conn, 1, "M", "2026-01-17", "雲林美津濃", atk_pts=60)
    _stat(conn, 2, "M", "2026-01-17", "屏東台電", atk_pts=40)


class _StubModel:
    """P(win) = ASR_roll3 / 100，讓測試可手算。"""

    def predict_proba(self, X):
        p = np.asarray(X)[:, 0] / 100
        return np.column_stack([1 - p, p])


ARTIFACT = {"model": _StubModel(), "version": "test", "feature_cols": ROLLING_FEATURES}


@pytest.fixture
def db(tmp_db_path):
    conn = connect_seeded(tmp_db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    _seed_history(conn)
    yield conn
    conn.close()


def _rows(conn):
    return conn.execute(
        "SELECT match_id, model_version, home_win_prob, computed_at FROM match_predictions ORDER BY match_id"
    ).fetchall()


def test_latest_features_match_training_path(db):
    labeled, _ = load_labeled_history(db)
    training = add_rolling_features(labeled)
    row = training[(training["team_id"] == 1) & (training["match_date"] == "2026-01-17")].iloc[0]
    vec = latest_rolling_features(labeled, 1, "M", "2026-01-17")
    pd.testing.assert_series_equal(vec, row[ROLLING_FEATURES], check_names=False)
    assert vec["ASR_roll3"] == pytest.approx(50.0)
    assert vec["win_streak"] == 1


def test_latest_features_use_all_history_before_cutoff(db):
    labeled, _ = load_labeled_history(db)
    vec = latest_rolling_features(labeled, 1, "M", "2026-02-01")
    assert vec["ASR_roll3"] == pytest.approx(55.0)
    assert vec["win_streak"] == -1
    assert latest_rolling_features(labeled, 5, "M", "2026-02-01") is None


def test_unplayed_written_and_played_frozen(db):
    played = _match(db, 3, "屏東台電", "雲林美津濃", [(25, 20)] * 3, date="2026-01-24")
    db.execute("INSERT INTO match_predictions VALUES (?, 'test', 0.42, '2026-01-23T00:00:00')", (played,))
    up1 = _match(db, 4, "屏東台電", "雲林美津濃", [], date="2026-02-01")
    up2 = _match(db, 5, "雲林美津濃", "屏東台電", [], date="2026-02-08")
    db.commit()

    report = run_predictions(db, ARTIFACT)

    assert report["written"] == 2
    rows = _rows(db)
    assert [r[0] for r in rows] == [played, up1, up2]
    assert rows[0][2:] == (0.42, "2026-01-23T00:00:00")
    # 主隊 屏東台電 roll3=55 → 0.55；客隊 雲林美津濃 roll3=35 → 0.35；正規化後主隊 0.55/0.90
    assert rows[1][2] == pytest.approx(0.55 / 0.90)
    assert rows[2][2] == pytest.approx(0.35 / 0.90)
    assert all(r[1] == "test" for r in rows)


def test_match_with_only_some_sets_filled_is_frozen(db):
    """比賽進行中（只填到第一局、sets_won 仍 NULL）也算已開打：不重算、不覆寫。"""
    partial = _match(db, 3, "屏東台電", "雲林美津濃", [], date="2026-01-24")
    db.execute("UPDATE matches SET home_set1 = 25, away_set1 = 20 WHERE match_id = ?", (partial,))
    db.execute("INSERT INTO match_predictions VALUES (?, 'test', 0.42, '2026-01-23T00:00:00')", (partial,))
    db.commit()

    report = run_predictions(db, ARTIFACT)

    assert report["unplayed"] == 0
    assert _rows(db) == [(partial, "test", 0.42, "2026-01-23T00:00:00")]


def test_team_without_history_is_skipped(db):
    _match(db, 6, "屏東台電", "桃園臺產", [], date="2026-02-01")
    _match(db, 7, "臺北國北獅", "獅子王", [], date="2026-02-01")
    _match(db, 8, "三福產業", "屏東台電", [], date="2026-02-01")
    db.commit()

    report = run_predictions(db, ARTIFACT)

    assert report["written"] == 0
    assert report["skipped_no_history"] == 3
    assert _rows(db) == []


def test_rerun_upserts_without_duplicates(db):
    up = _match(db, 4, "屏東台電", "雲林美津濃", [], date="2026-02-01")
    db.commit()
    run_predictions(db, ARTIFACT)
    first = _rows(db)
    run_predictions(db, ARTIFACT)
    assert _rows(db) == first
    db.execute("UPDATE match_predictions SET home_win_prob = 0.1 WHERE match_id = ?", (up,))
    run_predictions(db, ARTIFACT)
    assert _rows(db)[0][2] == pytest.approx(0.55 / 0.90)


def test_labels_skip_unplayed_unknown_team_before_normalizing():
    matches = pd.DataFrame([{
        "match_date": "2026-10-24", "gender": "M", "is_golden_set": 0,
        "home_team": "三福產業", "away_team": "臺北鳳凰",
        "home_sets_won": None, "away_sets_won": None,
    }])
    labels, report = build_match_labels(matches)
    assert labels.empty
    assert report["invalid_skipped"] == 1
