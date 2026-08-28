import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.models.features import GAME_STAT_COLS, load_team_match_stats, normalize_match_team
from src.utils.constants import LEGACY_TEAMS, OPP_SHORT_TO_TEAM

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


def _make_db(tmp_db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def _seed_team(conn, team_id, name, gender):
    conn.execute("INSERT INTO teams VALUES (?, ?, ?)", (team_id, name, gender))


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

# matches 表實際出現過的 14 種隊名（2026-08-27 實查）
MATCH_TEAM_NAMES = [
    "屏東台電", "屏東台電男排", "彰化三大有線", "新北中纖", "桃園臺產",
    "獅子王", "義力營造", "臺北Conti", "臺北國北獅", "臺北鯨華",
    "連莊", "雲林美津濃", "高雄台電", "高雄台電女排",
]


def test_all_match_team_names_normalize():
    for name in MATCH_TEAM_NAMES:
        short = normalize_match_team(name)
        assert short in OPP_SHORT_TO_TEAM or short in LEGACY_TEAMS


def test_suffix_variants_map_to_short():
    assert normalize_match_team("屏東台電男排") == "屏東台電"
    assert normalize_match_team("高雄台電女排") == "高雄台電"


def test_unknown_team_raises():
    with pytest.raises(ValueError, match="未知隊名"):
        normalize_match_team("不存在的隊")


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
