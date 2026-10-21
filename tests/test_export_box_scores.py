import json
import logging

import pytest

from tests.conftest import STAT_COLUMNS, seed_player, seed_registration, seed_stat


def _match(conn, game_id, home, away, date, gender="M", cup=21, golden=0):
    conn.execute(
        """
        INSERT INTO matches (game_id, gender, cup_id, match_date, is_golden_set, home_team, away_team,
                             home_team_id, away_team_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (game_id, gender, cup, date, golden, str(home), str(away), home, away),
    )


def _points(n):
    return (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n)


def _row(player_id, team_id, jersey, position, total_points):
    return {
        "player_id": player_id, "team_id": team_id, "jersey_number": jersey, "position": position,
        **dict.fromkeys(STAT_COLUMNS, 0), "sets_played": 1, "total_points": total_points,
    }


@pytest.fixture
def export(export_json):
    def run(build):
        payload, out_dir = export_json(build, "box_scores")
        return payload["box_scores"], out_dir
    return run


def _season_with_one_game(conn):
    for pid, name, gender in ((1, "甲", "M"), (2, "乙", "M"), (3, "丙", "M")):
        seed_player(conn, pid, name, gender)
    seed_registration(conn, 1, 1, 1, jersey=7, position="OH", week="W1", week_start="2025-11-01")
    seed_registration(conn, 2, 1, 1, jersey=17, position="OP", week="W2", week_start="2025-11-08")
    seed_registration(conn, 3, 2, 1, jersey=None, position=None, source="backfill")
    seed_registration(conn, 4, 3, 2, jersey=3, position="MB")
    _match(conn, 101, 1, 2, "2025-11-01")
    _match(conn, 901, 1, 2, "2025-11-01", golden=1)


def test_stat_rows_attach_to_home_and_away_by_registered_team_with_row_own_registration(export):
    def build(conn):
        _season_with_one_game(conn)
        seed_stat(conn, 1, _points(5), match_date="2025-11-01", opponent="雲林美津濃")
        seed_stat(conn, 3, _points(2), match_date="2025-11-01", opponent="雲林美津濃")
        seed_stat(conn, 4, _points(9), match_date="2025-11-01", opponent="屏東台電")

    box_scores, _ = export(build)

    groups = {(b["cup_id"], b["group"]): b["games"] for b in box_scores}
    [game] = groups[(21, "men")]
    assert game["game_id"] == 101
    assert game["home"] == [_row(1, 1, 7, "OH", 5), _row(2, 1, None, None, 2)]
    assert game["away"] == [_row(3, 2, 3, "MB", 9)]
    assert game["golden_set"] is None


def test_registered_team_that_is_away_side_goes_to_away_and_reverse_fixture_matches(export):
    def build(conn):
        _season_with_one_game(conn)
        _match(conn, 102, 2, 1, "2025-11-08")
        seed_stat(conn, 2, _points(4), match_date="2025-11-08", opponent="雲林美津濃")
        seed_stat(conn, 4, _points(6), match_date="2025-11-08", opponent="屏東台電")

    box_scores, _ = export(build)

    [game] = next(b["games"] for b in box_scores if (b["cup_id"], b["group"]) == (21, "men"))
    assert game["game_id"] == 102
    assert game["home"] == [_row(3, 2, 3, "MB", 6)]
    assert game["away"] == [_row(1, 1, 17, "OP", 4)]


def test_golden_set_stat_rows_attach_to_parent_match_golden_set(export):
    def build(conn):
        _season_with_one_game(conn)
        seed_stat(conn, 1, _points(5), match_date="2025-11-01", opponent="雲林美津濃")
        seed_stat(conn, 1, _points(1), match_date="2025-11-01", opponent="雲林美津濃", golden=1)
        seed_stat(conn, 4, _points(2), match_date="2025-11-01", opponent="屏東台電", golden=1)

    box_scores, _ = export(build)

    [game] = next(b["games"] for b in box_scores if (b["cup_id"], b["group"]) == (21, "men"))
    assert game["home"] == [_row(1, 1, 7, "OH", 5)]
    assert game["away"] == []
    assert game["golden_set"] == {
        "home": [_row(1, 1, 7, "OH", 1)],
        "away": [_row(3, 2, 3, "MB", 2)],
    }


def test_golden_set_stat_rows_attach_even_when_matches_has_no_golden_row(export):
    def build(conn):
        _season_with_one_game(conn)
        conn.execute("DELETE FROM matches WHERE game_id = 901")
        seed_stat(conn, 4, _points(2), match_date="2025-11-01", opponent="屏東台電", golden=1)

    box_scores, _ = export(build)

    [game] = next(b["games"] for b in box_scores if (b["cup_id"], b["group"]) == (21, "men"))
    assert game["home"] == [] and game["away"] == []
    assert game["golden_set"] == {"home": [], "away": [_row(3, 2, 3, "MB", 2)]}


def test_unmatched_rows_are_excluded_with_warning_and_export_continues(export, caplog):
    def build(conn):
        _season_with_one_game(conn)
        _match(conn, 103, 1, 2, "2025-11-15")
        _match(conn, 104, 2, 1, "2025-11-15")
        seed_stat(conn, 1, _points(5), match_date="2025-11-01", opponent="雲林美津濃")
        seed_stat(conn, 1, _points(7), match_date="2025-12-01", opponent="雲林美津濃")
        seed_stat(conn, 1, _points(8), match_date="2025-11-01", opponent="不存在的隊")
        seed_stat(conn, 1, _points(3), match_date="2025-11-15", opponent="雲林美津濃")
        seed_stat(conn, 1, _points(4), match_date="2025-11-01", opponent="臺北鯨華")

    with caplog.at_level(logging.WARNING, logger="src.export.site_data"):
        box_scores, out_dir = export(build)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 4
    assert any("2025-12-01" in w and "雲林美津濃" in w for w in warnings)
    assert any("不存在的隊" in w for w in warnings)
    assert any("2025-11-15" in w for w in warnings)
    assert any("臺北鯨華" in w for w in warnings)
    assert all("stat_id=" in w and "team_id=1" in w for w in warnings)
    assert sum("對手隊名無法解析" in w for w in warnings) == 1
    assert sum("對手組別與登錄組別不符" in w for w in warnings) == 1
    assert sum("符合 0 場" in w for w in warnings) == 1
    assert sum("符合 2 場" in w for w in warnings) == 1
    [game] = next(b["games"] for b in box_scores if (b["cup_id"], b["group"]) == (21, "men"))
    assert game["home"] == [_row(1, 1, 7, "OH", 5)]
    players = json.loads((out_dir / "players.json").read_text(encoding="utf-8"))["players"]
    assert players[0]["seasons"][0]["games"] == 5


def test_empty_data_still_lists_every_cup_and_group_with_no_games(export):
    box_scores, _ = export(lambda conn: None)

    assert box_scores == [
        {"cup_id": 20, "group": "men", "games": []},
        {"cup_id": 20, "group": "women", "games": []},
        {"cup_id": 21, "group": "men", "games": []},
        {"cup_id": 21, "group": "women", "games": []},
        {"cup_id": 22, "group": "men", "games": []},
        {"cup_id": 22, "group": "women", "games": []},
    ]


def test_games_sorted_by_date_then_game_id_and_only_those_with_matched_rows(export):
    def build(conn):
        _season_with_one_game(conn)
        _match(conn, 50, 1, 2, "2025-11-08")
        _match(conn, 70, 1, 2, "2025-11-20")
        seed_stat(conn, 2, _points(1), match_date="2025-11-08", opponent="雲林美津濃")
        seed_stat(conn, 1, _points(1), match_date="2025-11-01", opponent="雲林美津濃")

    box_scores, _ = export(build)

    games = next(b["games"] for b in box_scores if (b["cup_id"], b["group"]) == (21, "men"))
    assert [g["game_id"] for g in games] == [101, 50]


def test_cup_20_stats_attach_to_cup_20_men_group(export):
    def build(conn):
        seed_player(conn, 1, "甲", "M")
        seed_registration(conn, 1, 1, 1, cup_id=20, jersey=7, position="OH", week="W1", week_start="2024-11-02",
                          source="backfill")
        _match(conn, 201, 1, 2, "2024-11-02", cup=20)
        seed_stat(conn, 1, _points(4), match_date="2024-11-02", opponent="雲林美津濃")

    box_scores, _ = export(build)

    groups = {(b["cup_id"], b["group"]): b["games"] for b in box_scores}
    [game] = groups[(20, "men")]
    assert game["game_id"] == 201
    assert game["home"] == [_row(1, 1, 7, "OH", 4)]
