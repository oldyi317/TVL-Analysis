import json
import sqlite3
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from src.export.site_data import export_site_data
from src.utils.teams import apply_team_seed

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SQL = ROOT / "sql" / "schema.sql"
SCHEMA_PATH = ROOT / "web" / "schema" / "matches.schema.json"
COLUMNS = (
    "game_id, gender, cup_id, match_date, start_time, venue, round_name, game_label, is_golden_set, "
    "home_team, away_team, home_team_id, away_team_id, home_sets_won, away_sets_won, "
    "home_set1, home_set2, home_set3, home_set4, home_set5, "
    "away_set1, away_set2, away_set3, away_set4, away_set5"
)


def match(game_id, home, away, sets, *, date="2025-11-01", cup=21, gender="M",
          round_name="例行賽 Week 1", venue="臺北體育館", start_time=None, golden=0, label=None):
    home_sets = [s[0] for s in sets] + [None] * (5 - len(sets))
    away_sets = [s[1] for s in sets] + [None] * (5 - len(sets))
    home_won = sum(h > a for h, a in sets) if sets else None
    away_won = sum(a > h for h, a in sets) if sets else None
    return {
        "game_id": game_id, "gender": gender, "cup_id": cup, "match_date": date,
        "start_time": start_time, "venue": venue, "round_name": round_name,
        "game_label": label or f"Game {game_id}", "is_golden_set": golden,
        "home_team": str(home), "away_team": str(away), "home_team_id": home, "away_team_id": away,
        "home_sets_won": home_won, "away_sets_won": away_won,
        **{f"home_set{i}": v for i, v in enumerate(home_sets, 1)},
        **{f"away_set{i}": v for i, v in enumerate(away_sets, 1)},
    }


@pytest.fixture
def export_matches(tmp_db_path, tmp_path):
    def run(rows):
        conn = sqlite3.connect(tmp_db_path)
        conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
        apply_team_seed(conn)
        placeholders = ", ".join(f":{c.strip()}" for c in COLUMNS.split(","))
        conn.executemany(f"INSERT INTO matches ({COLUMNS}) VALUES ({placeholders})", rows)
        conn.commit()
        conn.close()
        out_dir = tmp_path / "out"
        export_site_data(tmp_db_path, out_dir)
        return json.loads((out_dir / "matches.json").read_text(encoding="utf-8"))
    return run


def group(payload, cup_id=21, name="men"):
    return next(g for g in payload["matches"] if g["cup_id"] == cup_id and g["group"] == name)


def test_played_match_json_structure(export_matches):
    payload = export_matches([
        match(10, 1, 2, [(25, 20), (23, 25), (25, 18), (25, 22)], date="2025-11-02",
              round_name="例行賽 Week 3", venue="高雄市鳳山體育館"),
    ])
    assert group(payload)["matches"] == [{
        "game_id": 10,
        "date": "2025-11-02",
        "weekday": "日",
        "start_time": None,
        "venue": "高雄市鳳山體育館",
        "stage": "例行賽",
        "week": 3,
        "home_team_id": 1,
        "away_team_id": 2,
        "sets": [[25, 20], [23, 25], [25, 18], [25, 22]],
        "home_sets_won": 3,
        "away_sets_won": 1,
        "golden_set": None,
    }]
    assert {(g["cup_id"], g["group"]) for g in payload["matches"]} == {
        (20, "men"), (20, "women"), (21, "men"), (21, "women"),
    }
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(payload)


def test_round_name_splits_into_stage_and_week(export_matches):
    payload = export_matches([
        match(1, 1, 2, [(25, 20)] * 3, date="2026-01-10", round_name="例行賽 Week 12"),
        match(2, 1, 2, [(25, 20)] * 3, date="2026-03-01", round_name="挑戰賽 Week 18"),
        match(3, 1, 2, [(25, 20)] * 3, date="2026-03-20", round_name="總決賽 Week 19"),
        match(4, 1, 2, [(25, 20)] * 3, date="2026-03-24", round_name=None),
    ])
    stage_week = [(m["stage"], m["week"]) for m in group(payload)["matches"]]
    assert stage_week == [("例行賽", 12), ("挑戰賽", 18), ("總決賽", 19), (None, None)]


def test_golden_set_is_attached_to_parent_match(export_matches):
    payload = export_matches([
        match(327, 1, 2, [(25, 20)] * 3, date="2026-03-22", round_name="總決賽 Week 19", label="Game 116"),
        match(328, 1, 2, [(13, 15)], date="2026-03-22", round_name="總決賽 Week 19",
              golden=1, label="Game 116-1 (黃金決勝局)"),
        match(300, 3, 4, [(25, 20)] * 3, date="2026-03-22", round_name="總決賽 Week 19"),
    ])
    entries = group(payload)["matches"]
    assert [m["game_id"] for m in entries] == [300, 327]
    by_id = {m["game_id"]: m for m in entries}
    assert by_id[327]["golden_set"] == [13, 15]
    assert by_id[327]["sets"] == [[25, 20]] * 3
    assert by_id[300]["golden_set"] is None


def test_golden_set_without_parent_fails_loudly(export_matches):
    with pytest.raises(ValueError, match="328"):
        export_matches([
            match(328, 1, 2, [(13, 15)], date="2026-03-22", golden=1, label="Game 116-1 (黃金決勝局)"),
        ])


def test_unplayed_match_has_empty_sets_and_keeps_schedule(export_matches):
    payload = export_matches([
        match(50, 5, 6, [], date="2026-04-04", start_time="14:00", venue="臺北體育館"),
        match(49, 1, 2, [(25, 20)] * 3, date="2026-03-28"),
    ])
    entries = group(payload)["matches"]
    assert [m["game_id"] for m in entries] == [49, 50]
    assert entries[1] == {
        "game_id": 50, "date": "2026-04-04", "weekday": "六", "start_time": "14:00",
        "venue": "臺北體育館", "stage": "例行賽", "week": 1,
        "home_team_id": 5, "away_team_id": 6,
        "sets": [], "home_sets_won": None, "away_sets_won": None, "golden_set": None,
    }
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(payload)


def test_golden_set_with_swapped_sides_is_flipped_to_parent_order(export_matches):
    payload = export_matches([
        match(1, 1, 2, [(25, 20)] * 3, date="2026-03-22"),
        match(2, 2, 1, [(15, 9)], date="2026-03-22", golden=1),
    ])
    (entry,) = group(payload)["matches"]
    assert (entry["home_team_id"], entry["golden_set"]) == (1, [9, 15])


def test_golden_set_with_two_candidate_parents_fails_loudly(export_matches):
    with pytest.raises(ValueError, match="2 場"):
        export_matches([
            match(1, 1, 2, [(25, 20)] * 3, date="2026-03-22"),
            match(2, 2, 1, [(25, 20)] * 3, date="2026-03-22"),
            match(3, 1, 2, [(15, 9)], date="2026-03-22", golden=1),
        ])
