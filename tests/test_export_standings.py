import json
import sqlite3
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from src.export.site_data import export_site_data
from src.utils.db_config import DB_PATH
from src.utils.teams import apply_team_seed

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SQL = ROOT / "sql" / "schema.sql"
SCHEMA_PATH = ROOT / "web" / "schema" / "standings.schema.json"
REGULAR = "例行賽 Week 1"


def _set_scores(sets_won, sets_lost, win_pts, lose_pts):
    winner = [win_pts] * sets_won + [lose_pts] * sets_lost
    loser = [lose_pts] * sets_won + [win_pts] * sets_lost
    pad = [None] * (5 - len(winner))
    return winner + pad, loser + pad


def match(game_id, home, away, home_sets_won, away_sets_won, *, date="2025-11-01", cup=21,
          gender="M", round_name=REGULAR, golden=0, pts=(25, 20)):
    if home_sets_won > away_sets_won:
        home_sets, away_sets = _set_scores(home_sets_won, away_sets_won, *pts)
    else:
        away_sets, home_sets = _set_scores(away_sets_won, home_sets_won, *pts)
    return (game_id, gender, cup, date, round_name, golden, home, away, home, away,
            home_sets_won, away_sets_won, *home_sets, *away_sets)


def unplayed(game_id, home, away, **kw):
    row = list(match(game_id, home, away, 3, 0, **kw))
    row[10:] = [None] * (len(row) - 10)
    return tuple(row)


@pytest.fixture
def export_matches(tmp_db_path, tmp_path):
    def run(rows):
        conn = sqlite3.connect(tmp_db_path)
        conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
        apply_team_seed(conn)
        conn.executemany(
            """
            INSERT INTO matches (game_id, gender, cup_id, match_date, round_name, is_golden_set,
                                 home_team, away_team, home_team_id, away_team_id,
                                 home_sets_won, away_sets_won,
                                 home_set1, home_set2, home_set3, home_set4, home_set5,
                                 away_set1, away_set2, away_set3, away_set4, away_set5)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [(r[0], r[1], r[2], r[3], r[4], r[5], str(r[6]), str(r[7]), *r[8:]) for r in rows],
        )
        conn.commit()
        conn.close()
        out_dir = tmp_path / "out"
        export_site_data(tmp_db_path, out_dir)
        return json.loads((out_dir / "standings.json").read_text(encoding="utf-8"))

    return run


def group(payload, cup_id=21, name="men"):
    return next(g for g in payload["standings"] if g["cup_id"] == cup_id and g["group"] == name)


def row(payload, team_id, cup_id=21, name="men"):
    return next(r for r in group(payload, cup_id, name)["rows"] if r["team_id"] == team_id)


def order(payload, cup_id=21, name="men"):
    return [r["team_id"] for r in group(payload, cup_id, name)["rows"]]


def test_points_for_each_set_score(export_matches):
    p = export_matches([
        match(1, 1, 2, 3, 0),
        match(2, 1, 4, 3, 1),
        match(3, 1, 5, 3, 2),
        match(4, 7, 1, 3, 2),
        match(5, 7, 2, 3, 1),
        match(6, 7, 4, 0, 3),
    ])
    assert row(p, 1)["points"] == 3 + 3 + 2 + 1
    assert row(p, 1)["set_record"] == {"3-0": 1, "3-1": 1, "3-2": 1, "2-3": 1, "1-3": 0, "0-3": 0}
    assert row(p, 2)["points"] == 0
    assert row(p, 2)["set_record"]["0-3"] == 1 and row(p, 2)["set_record"]["1-3"] == 1
    assert row(p, 5)["points"] == 1
    assert row(p, 7)["points"] == 2 + 3
    assert row(p, 4)["points"] == 3


def test_totals_and_ratios(export_matches):
    p = export_matches([match(1, 1, 2, 3, 1), match(2, 2, 1, 3, 0, date="2025-11-02")])
    r1 = row(p, 1)
    assert (r1["wins"], r1["losses"], r1["win_pct"]) == (1, 1, 0.5)
    assert (r1["sets_won"], r1["sets_lost"], r1["set_ratio"]) == (3, 4, 0.75)
    assert r1["points_for"] == 3 * 25 + 20 + 3 * 20
    assert r1["points_against"] == 3 * 20 + 25 + 3 * 25
    assert r1["point_ratio"] == round(r1["points_for"] / r1["points_against"], 3)


def test_golden_set_rows_are_ignored(export_matches):
    p = export_matches([match(1, 1, 2, 3, 0), match(2, 2, 1, 1, 0, golden=1)])
    assert (row(p, 2)["wins"], row(p, 2)["losses"], row(p, 2)["points"]) == (0, 1, 0)
    assert (row(p, 2)["sets_won"], row(p, 2)["sets_lost"]) == (0, 3)
    assert row(p, 1)["last5"] == ["W"]


def test_non_regular_rounds_and_unplayed_are_not_counted(export_matches):
    p = export_matches([
        match(1, 1, 2, 3, 0),
        match(2, 1, 7, 3, 0, round_name="挑戰賽 Week 18"),
        match(3, 1, 7, 3, 0, round_name=None),
        unplayed(4, 1, 5),
    ])
    assert row(p, 1)["wins"] == 1
    assert row(p, 1)["last5"] == ["W"]
    for team_id in (5, 7):
        r = row(p, team_id)
        assert (r["wins"], r["losses"], r["points"], r["sets_won"], r["points_for"]) == (0, 0, 0, 0, 0)
        assert (r["win_pct"], r["set_ratio"], r["point_ratio"], r["last5"]) == (None, None, None, [])
    assert sorted(order(p)) == [1, 2, 5, 7]
    assert order(p)[-2:] == [5, 7]


def test_rank_by_wins_before_points(export_matches):
    p = export_matches([
        match(1, 1, 4, 3, 2),
        match(2, 1, 5, 3, 2),
        match(3, 2, 7, 3, 0),
        match(4, 4, 2, 3, 2),
        match(5, 5, 2, 3, 2),
    ])
    assert row(p, 1)["points"] == 4 and row(p, 2)["points"] == 5
    assert order(p).index(1) < order(p).index(2)


def test_rank_by_points_when_wins_tie(export_matches):
    p = export_matches([match(1, 1, 5, 3, 2), match(2, 2, 4, 3, 0)])
    assert order(p)[:2] == [2, 1]


def test_rank_by_set_ratio_with_null_as_infinity(export_matches):
    p = export_matches([match(1, 1, 4, 3, 1), match(2, 2, 5, 3, 0)])
    assert row(p, 2)["set_ratio"] is None
    assert order(p)[:2] == [2, 1]


def test_rank_by_point_ratio(export_matches):
    p = export_matches([
        match(1, 1, 4, 3, 1, pts=(25, 23)),
        match(2, 2, 5, 3, 1, pts=(25, 15)),
    ])
    assert row(p, 1)["set_ratio"] == row(p, 2)["set_ratio"]
    assert order(p)[:2] == [2, 1]


def test_rank_falls_back_to_team_id(export_matches):
    p = export_matches([match(1, 2, 5, 3, 0), match(2, 1, 4, 3, 0)])
    assert order(p)[:2] == [1, 2]
    assert [r["rank"] for r in group(p)["rows"]] == [1, 2, 3, 4]


def test_point_ratio_is_null_when_no_points_conceded(export_matches):
    p = export_matches([match(1, 1, 2, 3, 0, pts=(25, 0))])
    assert row(p, 1)["point_ratio"] is None
    assert row(p, 1)["set_ratio"] is None
    assert row(p, 2)["point_ratio"] == 0.0


def test_last5_is_oldest_to_newest_and_capped(export_matches):
    results = [(3, 0), (3, 0), (0, 3), (3, 0), (0, 3), (0, 3), (3, 0)]
    dates = ["2025-11-01", "2025-11-02", "2025-11-03", "2025-11-04", "2025-11-05",
             "2025-11-08", "2025-11-08"]
    rows = [match(10 + i, 1, 2, hs, as_, date=d)
            for i, ((hs, as_), d) in reversed(list(enumerate(zip(results, dates))))]
    p = export_matches(rows)
    assert row(p, 1)["last5"] == ["L", "W", "L", "L", "W"]
    assert row(p, 2)["last5"] == ["W", "L", "W", "W", "L"]


def test_games_without_team_id_are_skipped(export_matches):
    broken = list(match(2, 1, 2, 3, 0, date="2025-11-02"))
    broken[9] = None
    payload = export_matches([match(1, 1, 2, 3, 1), tuple(broken)])
    assert row(payload, 1)["wins"] == 1
    assert row(payload, 2)["losses"] == 1


def test_aggregates_by_team_id_not_name(export_matches):
    rows = [match(1, 1, 2, 3, 0), match(2, 1, 2, 3, 0, date="2025-11-02")]
    rows[1] = (*rows[1][:6], "屏東台電男排", "雲林美津濃", *rows[1][8:])
    p = export_matches(rows)
    assert row(p, 1)["wins"] == 2


def test_groups_cover_every_season_and_group_in_order(export_matches):
    p = export_matches([match(1, 1, 2, 3, 0)])
    assert [(g["cup_id"], g["group"]) for g in p["standings"]] == [
        (20, "men"), (20, "women"), (21, "men"), (21, "women"),
    ]
    assert group(p, 21, "women")["rows"] == []


def test_men_and_women_are_separate(export_matches):
    p = export_matches([
        match(1, 4, 5, 3, 0, gender="F"),
        match(2, 4, 5, 0, 3, gender="M"),
    ])
    assert row(p, 4, name="women")["wins"] == 1
    assert row(p, 4, name="men")["wins"] == 0


def test_note_only_for_2024_25_women(export_matches):
    p = export_matches([match(1, 4, 5, 3, 0, gender="F", cup=20)])
    notes = {(g["cup_id"], g["group"]): g["note"] for g in p["standings"]}
    assert notes[(20, "women")] == "2024-25 女子組官網只收錄 8 場賽事，戰績不完整"
    assert [k for k, v in notes.items() if v is not None] == [(20, "women")]


def test_standings_json_matches_schema(export_matches):
    p = export_matches([match(1, 1, 2, 3, 0), unplayed(2, 1, 5)])
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(p)


@pytest.fixture(scope="module")
def real_standings(tmp_path_factory):
    if not DB_PATH.exists():
        pytest.skip("真實 DB 不存在")
    out_dir = tmp_path_factory.mktemp("real_out")
    export_site_data(DB_PATH, out_dir)
    standings = json.loads((out_dir / "standings.json").read_text(encoding="utf-8"))
    teams = json.loads((out_dir / "teams.json").read_text(encoding="utf-8"))["teams"]
    ids = {(t["group"], t["name"]): t["team_id"] for t in teams}
    return standings, ids


def test_real_db_matches_official_points(real_standings):
    p, ids = real_standings
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(p)
    expected = [
        ("men", "屏東台電", 64), ("men", "雲林美津濃", 52),
        ("women", "臺北鯨華", 55), ("women", "義力營造", 19),
    ]
    for name, team, points in expected:
        assert row(p, ids[(name, team)], 21, name)["points"] == points, team


def test_real_db_2024_25_keeps_legacy_teams_and_flags_women(real_standings):
    p, _ = real_standings
    assert {8, 9, 10} <= set(order(p, 20, "men"))
    assert group(p, 20, "women")["note"] is not None
