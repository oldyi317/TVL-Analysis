import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.models.features import (
    GAME_STAT_COLS,
    attach_labels,
    build_match_labels,
    load_labeled_history,
    load_team_match_stats,
)
from src.utils.db_config import DB_PATH
from tests.conftest import connect_seeded

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


def _make_db(tmp_db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def _seed_team(conn, team_id, name, gender):
    conn.execute("INSERT INTO teams (team_id, team_name, gender) VALUES (?, ?, ?)", (team_id, name, gender))


def _seed_stat(conn, team_id, gender, match_date, opponent, *,
               atk_pts=10, atk_tot=20, total_points=25, golden=0, sets_played=3, blk_pts=2):
    cur = conn.execute(
        "INSERT INTO players (name, gender) VALUES (?, ?)", (f"p{team_id}{match_date}{golden}", gender))
    pid = cur.lastrowid
    cur = conn.execute(
        """INSERT INTO roster_registrations
           (player_id, team_id, gender, cup_id, week_label, source)
           VALUES (?, ?, ?, 21, ?, 'match_page')""",
        (pid, team_id, gender, f"w{match_date}"))
    rid = cur.lastrowid
    conn.execute(
        """INSERT INTO player_match_stats
           (registration_id, match_date, opponent, sets_played,
            attack_total, attack_points, block_points, serve_total, serve_points,
            receive_total, receive_excellent, dig_total, dig_excellent,
            set_total, set_excellent, total_points, is_golden_set)
           VALUES (?, ?, ?, ?, ?, ?, ?, 10, 1, 10, 5, 10, 3, 10, 5, ?, ?)""",
        (rid, match_date, opponent, sets_played, atk_tot, atk_pts, blk_pts, total_points, golden))
    conn.commit()


def _match_row(home_id, away_id, gender, hsw, asw, *, date="2026-01-10",
               h_total=75, a_total=70, golden=0):
    return {
        "match_date": date, "gender": gender, "is_golden_set": golden,
        "home_team_id": home_id, "away_team_id": away_id,
        "home_total": h_total, "away_total": a_total,
        "home_sets_won": hsw, "away_sets_won": asw,
    }


def test_team_match_aggregation_excludes_golden_set(tmp_db_path):
    conn = _make_db(tmp_db_path)
    _seed_team(conn, 1, "屏東台電", "M")
    _seed_stat(conn, 1, "M", "2026-01-10", "獅子王")
    _seed_stat(conn, 1, "M", "2026-01-10", "獅子王", golden=1)
    df = load_team_match_stats(conn)
    conn.close()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["team_id"] == 1 and row["gender"] == "M"
    assert row["ASR"] == pytest.approx(50.0)  # 10/20*100
    for col in GAME_STAT_COLS:
        assert col in df.columns


def test_label_uses_sets_won_not_total_points():
    # 總得分較高但局數落敗（proxy 會標錯的 13 場型態）
    matches = pd.DataFrame([_match_row(
        1, 7, "M", 2, 3, h_total=110, a_total=105)])
    labels, _ = build_match_labels(matches)
    home = labels[labels["team_id"] == 1].iloc[0]
    away = labels[labels["team_id"] == 7].iloc[0]
    assert home["win"] == 0 and away["win"] == 1


def test_labels_skip_golden_set_and_legacy():
    matches = pd.DataFrame([
        _match_row(1, 7, "M", 3, 1),
        _match_row(1, 7, "M", 1, 0, golden=1, date="2026-01-11"),
        _match_row(9, 10, "M", 3, 2, date="2024-12-01"),
    ])
    labels, report = build_match_labels(matches)
    assert len(labels) == 2  # 只有第一場的主客兩筆
    assert report["legacy_skipped"] == 1


def test_labels_skip_null_team_id_as_unresolved():
    matches = pd.DataFrame([_match_row(None, 7, "M", 3, 0)])
    labels, report = build_match_labels(matches)
    assert labels.empty
    assert report["unresolved_skipped"] == 1
    assert report["invalid_skipped"] == 0


def test_attach_labels_fails_loud_on_missing_score(tmp_db_path):
    conn = _make_db(tmp_db_path)
    _seed_team(conn, 1, "屏東台電", "M")
    _seed_stat(conn, 1, "M", "2026-01-10", "獅子王")
    team_match = load_team_match_stats(conn)
    conn.close()
    empty_labels = pd.DataFrame(columns=["match_date", "team_id", "gender", "win"])
    with pytest.raises(ValueError, match="找不到對應比分"):
        attach_labels(team_match, empty_labels)


def _seed_match(conn, game_id, home_id, away_id, gender, hsw, asw, *, date,
                home="屏東台電", away="雲林美津濃"):
    conn.execute(
        """INSERT INTO matches (game_id, gender, cup_id, match_date, is_golden_set,
                                home_team, away_team, home_team_id, away_team_id,
                                home_sets_won, away_sets_won)
           VALUES (?, ?, 22, ?, 0, ?, ?, ?, ?, ?, ?)""",
        (game_id, gender, date, home, away, home_id, away_id, hsw, asw))


def test_labeled_history_drops_legacy_team_rows_like_labels_do(tmp_db_path):
    # #81：舊隊賽事的標籤刻意跳過（#48），統計列也要同步排除，不能炸在 attach_labels
    conn = connect_seeded(tmp_db_path)
    _seed_match(conn, 1, 10, 1, "M", 0, 3, date="2024-10-19", home="連莊", away="屏東台電男排")
    _seed_match(conn, 2, 1, 2, "M", 3, 1, date="2024-10-20")
    _seed_stat(conn, 10, "M", "2024-10-19", "屏東台電")
    _seed_stat(conn, 1, "M", "2024-10-19", "連莊")
    _seed_stat(conn, 1, "M", "2024-10-20", "雲林美津濃")
    _seed_stat(conn, 2, "M", "2024-10-20", "屏東台電")
    labeled, report = load_labeled_history(conn)
    conn.close()
    assert sorted(labeled["match_date"].unique()) == ["2024-10-20"]
    assert len(labeled) == 2
    assert report["legacy_skipped"] == 1
    assert report["legacy_stat_rows_skipped"] == 2


def test_total_sets_falls_back_to_match_sets_when_sets_played_null(tmp_db_path):
    conn = connect_seeded(tmp_db_path)
    _seed_match(conn, 1, 1, 2, "M", 3, 1, date="2026-10-25")
    _seed_match(conn, 2, 2, 1, "M", 3, 2, date="2026-11-01", home="雲林美津濃", away="屏東台電")
    _seed_stat(conn, 1, "M", "2026-10-25", "雲林美津濃", sets_played=None, blk_pts=4)
    _seed_stat(conn, 2, "M", "2026-10-25", "屏東台電", sets_played=None, blk_pts=8)
    _seed_stat(conn, 1, "M", "2026-11-01", "雲林美津濃", sets_played=3, blk_pts=6)
    df = load_team_match_stats(conn).set_index(["match_date", "team_id"])
    conn.close()
    assert df.loc[("2026-10-25", 1), "total_sets"] == 4
    assert df.loc[("2026-10-25", 1), "BLK_per_set"] == pytest.approx(1.0)
    assert df.loc[("2026-10-25", 2), "BLK_per_set"] == pytest.approx(2.0)
    # 個人局數已知時維持原值，不被比賽局數（5）取代
    assert df.loc[("2026-11-01", 1), "total_sets"] == 3
    assert df.loc[("2026-11-01", 1), "BLK_per_set"] == pytest.approx(2.0)


def test_total_sets_null_with_warning_when_match_unresolvable(tmp_db_path, caplog):
    conn = connect_seeded(tmp_db_path)
    _seed_stat(conn, 1, "M", "2026-10-25", "雲林美津濃", sets_played=None, blk_pts=4)
    with caplog.at_level("WARNING"):
        df = load_team_match_stats(conn)
    conn.close()
    assert pd.isna(df.iloc[0]["total_sets"])
    assert df.iloc[0]["BLK_per_set"] == 0.0
    assert "對不回賽事" in caplog.text and "2026-10-25" in caplog.text


def test_duplicate_match_same_day_fails_loud(tmp_db_path):
    conn = connect_seeded(tmp_db_path)
    _seed_match(conn, 1, 1, 2, "M", 3, 1, date="2026-10-25")
    _seed_match(conn, 2, 1, 2, "M", 3, 0, date="2026-10-25")
    _seed_stat(conn, 1, "M", "2026-10-25", "雲林美津濃", sets_played=None)
    with pytest.raises(ValueError, match="多場賽事"):
        load_team_match_stats(conn)
    conn.close()


@pytest.mark.skipif(not DB_PATH.exists(), reason="真實 DB 不存在")
def test_real_db_historical_total_sets_unchanged():
    conn = sqlite3.connect(DB_PATH)
    df = load_team_match_stats(conn)
    legacy = pd.read_sql_query(
        """SELECT s.match_date, r.team_id, r.gender, s.opponent, r.cup_id,
                  MAX(s.sets_played) AS legacy_sets
           FROM player_match_stats s
           JOIN roster_registrations r ON s.registration_id = r.registration_id
           WHERE s.is_golden_set = 0 AND r.cup_id IN (20, 21)
           GROUP BY s.match_date, r.team_id, r.gender, s.opponent""", conn)
    conn.close()
    assert not legacy.empty
    merged = legacy.merge(df, on=["match_date", "team_id", "gender", "opponent"], how="left")
    diff = merged[merged["legacy_sets"] != merged["total_sets"]]
    assert diff.empty, f"cup 20/21 total_sets 與改前不同：\n{diff.to_string(index=False)}"
