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
    assert len(payload["teams"]) == 14
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
    assert keys[-4:] == [("women", 3), ("women", 4), ("women", 5), ("women", 7)]


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
