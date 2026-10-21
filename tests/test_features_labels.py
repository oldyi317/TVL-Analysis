import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.models.features import (
    GAME_STAT_COLS,
    attach_labels,
    build_match_labels,
    load_team_match_stats,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


def _make_db(tmp_db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def _seed_team(conn, team_id, name, gender):
    conn.execute("INSERT INTO teams (team_id, team_name, gender) VALUES (?, ?, ?)", (team_id, name, gender))


def _seed_stat(conn, team_id, gender, match_date, opponent, *,
               atk_pts=10, atk_tot=20, total_points=25, golden=0):
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
           VALUES (?, ?, ?, 3, ?, ?, 2, 10, 1, 10, 5, 10, 3, 10, 5, ?, ?)""",
        (rid, match_date, opponent, atk_tot, atk_pts, total_points, golden))
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
