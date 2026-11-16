import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from src.export.site_data import export_site_data
from src.utils.teams import apply_team_seed

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SQL = ROOT / "sql" / "schema.sql"
SCHEMA_DIR = ROOT / "web" / "schema"


def _validate(payload: dict, schema_name: str) -> None:
    schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(payload)


@pytest.fixture
def seeded_db(tmp_db_path):
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
    apply_team_seed(conn)
    conn.executemany(
        """
        INSERT INTO matches (game_id, gender, cup_id, match_date, home_team, away_team,
                             home_team_id, away_team_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (1, "M", 20, "2024-11-23", "彰化三大有線", "屏東台電", 8, 1),
            (2, "M", 21, "2025-11-01", "屏東台電", "雲林美津濃", 1, 2),
            (3, "F", 21, "2025-11-01", "高雄台電", "新北中纖", 4, 5),
            (4, "M", 21, "2025-11-08", "臺北國北獅", "桃園臺灣產險", 4, 5),
            (5, "M", 22, "2026-11-07", "臺北鳳凰", "龍邁建設", 4, 5),
            (6, "M", 20, "2024-11-30", "桃園臺灣產險", "屏東台電", 5, 1),
        ],
    )
    conn.commit()
    conn.close()
    return tmp_db_path


@pytest.fixture
def exported(seeded_db, tmp_path):
    out_dir = tmp_path / "out"
    export_site_data(seeded_db, out_dir)
    return out_dir


def test_seasons_json_lists_all_seasons_and_marks_latest_as_current(exported):
    payload = json.loads((exported / "seasons.json").read_text(encoding="utf-8"))
    _validate(payload, "seasons.schema.json")
    assert payload["current_cup_id"] == 22
    assert [s["name"] for s in payload["seasons"]] == ["2024-25", "2025-26", "2026-27"]
    assert payload["seasons"][1] == {
        "cup_id": 21,
        "name": "2025-26",
        "start_date": "2025-07-01",
        "end_date": "2026-06-30",
    }


def _team(payload: dict, group: str, team_id: int) -> dict:
    return next(t for t in payload["teams"] if t["group"] == group and t["team_id"] == team_id)


def test_teams_json_carries_identity_and_season_participation(exported):
    payload = json.loads((exported / "teams.json").read_text(encoding="utf-8"))
    _validate(payload, "teams.schema.json")
    assert len(payload["teams"]) == 16
    assert _team(payload, "men", 1) == {
        "team_id": 1,
        "group": "men",
        "name": "屏東台電",
        "short_name": "屏東台電",
        "color": "#0B304B",
        "is_active": True,
        "cup_ids": [20, 21],
        "names_by_cup": {
            "20": {"name": "屏東台電", "short_name": "屏東台電"},
            "21": {"name": "屏東台電", "short_name": "屏東台電"},
        },
    }
    legacy = _team(payload, "men", 8)
    assert legacy["is_active"] is False
    assert legacy["cup_ids"] == [20]
    assert _team(payload, "women", 4)["name"] == "高雄台電"
    assert _team(payload, "women", 4)["cup_ids"] == [21]
    assert _team(payload, "men", 4)["cup_ids"] == [21, 22]
    assert _team(payload, "men", 5)["cup_ids"] == [20, 21, 22]
    assert _team(payload, "men", 7)["cup_ids"] == []


def test_teams_json_names_by_cup_uses_season_names_and_falls_back_to_current(exported):
    payload = json.loads((exported / "teams.json").read_text(encoding="utf-8"))
    _validate(payload, "teams.schema.json")
    lions = _team(payload, "men", 4)["names_by_cup"]
    assert lions["21"] == {"name": "臺北國北獅", "short_name": "臺北國北獅"}
    assert lions["22"] == {"name": "臺北鳳凰", "short_name": "臺北鳳凰"}
    old_name = {"name": "桃園臺灣產險", "short_name": "桃園臺產"}
    hawks = _team(payload, "men", 5)["names_by_cup"]
    assert hawks["20"] == old_name
    assert hawks["21"] == old_name
    assert hawks["22"]["name"] == _team(payload, "men", 5)["name"]
    assert hawks["22"]["short_name"] == _team(payload, "men", 5)["short_name"]
    assert _team(payload, "women", 4)["names_by_cup"] == {
        "21": {"name": "高雄台電", "short_name": _team(payload, "women", 4)["short_name"]}
    }
    assert _team(payload, "men", 7)["names_by_cup"] == {}
    for team in payload["teams"]:
        assert set(team["names_by_cup"]) == {str(c) for c in team["cup_ids"]}


def test_teams_json_sorted_men_first_then_team_id(exported):
    payload = json.loads((exported / "teams.json").read_text(encoding="utf-8"))
    keys = [(t["group"], t["team_id"]) for t in payload["teams"]]
    assert keys[:5] == [("men", 1), ("men", 2), ("men", 4), ("men", 5), ("men", 7)]
    assert keys[-5:] == [("women", 3), ("women", 4), ("women", 5), ("women", 7), ("women", 13)]


def test_site_json_records_generation_time_in_taipei(seeded_db, tmp_path):
    taipei = timezone(timedelta(hours=8))
    before = datetime.now(taipei).replace(microsecond=0)
    out_dir = tmp_path / "out"
    export_site_data(seeded_db, out_dir)
    after = datetime.now(taipei)

    payload = json.loads((out_dir / "site.json").read_text(encoding="utf-8"))
    _validate(payload, "site.schema.json")
    generated_at = payload["generated_at"]
    assert generated_at.endswith("+08:00")
    parsed = datetime.fromisoformat(generated_at)
    assert parsed.utcoffset() == timedelta(hours=8)
    assert before <= parsed <= after


MATCH_COLUMNS = (
    "game_id, gender, cup_id, match_date, home_team, away_team, home_team_id, away_team_id, "
    "round_name, is_golden_set, start_time, home_sets_won, away_sets_won, home_set1, away_set1"
)


def _insert_matches(db_path: Path, rows: list[tuple]) -> None:
    conn = sqlite3.connect(db_path)
    conn.executemany(f"INSERT INTO matches ({MATCH_COLUMNS}) VALUES ({', '.join('?' * 15)})", rows)
    conn.commit()
    conn.close()


def _export_push(db_path: Path, tmp_path: Path) -> dict:
    push_dir = tmp_path / "push"
    export_site_data(db_path, tmp_path / "out", push_dir)
    return json.loads((push_dir / "matches.json").read_text(encoding="utf-8"))


@pytest.fixture
def push_matches(seeded_db, tmp_path):
    return _export_push(seeded_db, tmp_path)


def test_push_matches_covers_only_current_season(push_matches):
    _validate(push_matches, "push_matches.schema.json")
    assert push_matches["version"] == 1
    assert push_matches["cup_id"] == 22
    assert [(m["group"], m["game_id"], m["date"]) for m in push_matches["matches"]] == [
        ("men", 5, "2026-11-07"),
    ]


def test_push_matches_skips_placeholder_games_and_golden_sets_as_separate_entries(seeded_db, tmp_path):
    _insert_matches(seeded_db, [
        (8, "M", 22, "2026-11-14", "臺北鳳凰", "龍邁建設", 4, 5, "例行賽 Week 2", 0, None, None, None, None, None),
        (9, "M", 22, "2026-11-14", "臺北鳳凰", "龍邁建設", 4, 5, "例行賽 Week 2", 1, None, None, None, 15, 13),
        (7, "M", 22, "2026-11-21", "待定", "待定", None, None, "季後賽", 0, None, None, None, None, None),
    ])
    payload = _export_push(seeded_db, tmp_path)
    _validate(payload, "push_matches.schema.json")
    assert [m["game_id"] for m in payload["matches"]] == [5, 8]


def test_push_matches_names_teams_by_current_season_short_name(seeded_db, tmp_path):
    conn = sqlite3.connect(seeded_db)
    conn.executemany(
        "INSERT INTO team_season_names (team_id, gender, cup_id, team_name, short_name) VALUES (?, 'M', 22, ?, ?)",
        [(4, "臺北鳳凰排球隊", "北鳳"), (5, "龍邁建設隊", "龍邁")],
    )
    conn.commit()
    conn.close()
    payload = _export_push(seeded_db, tmp_path)
    assert [(m["game_id"], m["home"], m["away"]) for m in payload["matches"]] == [(5, "北鳳", "龍邁")]


def test_push_matches_allows_null_time_and_prob_and_carries_result_and_prob_when_known(seeded_db, tmp_path):
    _insert_matches(seeded_db, [
        (8, "F", 22, "2026-11-14", "高雄台電", "新北中纖", 4, 5, "例行賽 Week 2", 0, "18:30", 3, 2, None, None),
    ])
    conn = sqlite3.connect(seeded_db)
    conn.execute(
        "INSERT INTO match_predictions (match_id, model_version, home_win_prob, computed_at) "
        "SELECT match_id, 'v2', 0.625, '2026-11-13T09:00:00' FROM matches WHERE game_id = 8 AND gender = 'F'"
    )
    conn.commit()
    conn.close()
    payload = _export_push(seeded_db, tmp_path)
    _validate(payload, "push_matches.schema.json")
    unscored, scored = payload["matches"]
    assert unscored == {
        "group": "men", "game_id": 5, "date": "2026-11-07", "start_time": None,
        "home": "臺北鳳凰", "away": "龍邁建設",
        "home_sets_won": None, "away_sets_won": None, "home_win_prob": None,
    }
    assert scored == {
        "group": "women", "game_id": 8, "date": "2026-11-14", "start_time": "18:30",
        "home": "高雄台電", "away": "新北中纖",
        "home_sets_won": 3, "away_sets_won": 2, "home_win_prob": 0.625,
    }


def test_push_matches_links_point_to_current_season_match_pages(push_matches):
    assert push_matches["links"] == {
        "men": "/2026-27/men/matches/",
        "women": "/2026-27/women/matches/",
        "both": "/matches/",
    }


def test_export_without_push_dir_writes_no_push_matches_file(seeded_db, tmp_path):
    export_site_data(seeded_db, tmp_path / "out")
    assert (tmp_path / "out" / "matches.json").exists()
    assert not list(tmp_path.rglob("push"))
