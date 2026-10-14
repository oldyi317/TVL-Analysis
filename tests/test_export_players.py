import pytest

from tests.conftest import STAT_COLUMNS, seed_player, seed_registration, seed_stat


@pytest.fixture
def export(export_json):
    return lambda build: export_json(build, "players")[0]["players"]


def test_players_listed_by_group_then_id_with_identity_only(export):
    def build(conn):
        seed_player(conn, 30, "女甲", "F", dob="2000-05-06", height=170.0, weight=60.0)
        seed_player(conn, 20, "男乙", "M")
        seed_player(conn, 10, "男甲", "M", dob="1998-01-01", height=185.0, weight=80.0)

    players = export(build)

    assert [p["player_id"] for p in players] == [10, 20, 30]
    assert players[0] == {
        "player_id": 10, "group": "men", "name": "男甲",
        "dob": "1998-01-01", "height_cm": 185.0, "weight_kg": 80.0, "seasons": [],
    }
    assert players[1]["dob"] is None and players[1]["height_cm"] is None and players[1]["weight_kg"] is None
    assert players[2]["group"] == "women"
    assert all("jersey_number" not in p and "position" not in p for p in players)


def test_season_totals_count_golden_set_rows_as_games_and_sum_all_columns(export):
    """賽季合計沿用舊站 COUNT(*) / SUM 口徑：黃金決勝局列也算一場、各欄照加；NULL 當 0。"""
    def build(conn):
        seed_player(conn, 1, "男甲")
        seed_registration(conn, 1, 1, team_id=1, jersey=7, position="OH")
        seed_stat(conn, 1, (3, 10, 5, 1, 4, 2, 6, 3, 8, 4, 0, 0, 8), match_date="2025-11-01")
        seed_stat(conn, 1, (2, 20, 10, 2, 5, 1, 4, 2, 6, 3, 10, 5, 13), match_date="2025-11-08")
        seed_stat(conn, 1, (1, 3, 2, 0, 1, 1, 0, 0, 2, 1, 0, 0, 3), match_date="2025-11-08", golden=1)
        seed_stat(conn, 1, (None,) * 13, match_date="2025-11-15")

    [player] = export(build)

    assert player["seasons"] == [{
        "cup_id": 21, "team_id": 1, "jersey_number": 7, "position": "OH", "games": 4,
        "sets_played": 6, "attack_total": 33, "attack_points": 17, "block_points": 3,
        "serve_total": 10, "serve_points": 4, "receive_total": 10, "receive_excellent": 5,
        "dig_total": 16, "dig_excellent": 8, "set_total": 10, "set_excellent": 5, "total_points": 24,
    }]


def test_one_season_entry_per_registered_cup_ascending_and_zero_when_no_stats(export):
    def build(conn):
        seed_player(conn, 1, "男甲")
        seed_player(conn, 2, "無登錄")
        seed_registration(conn, 1, 1, team_id=1, cup_id=21, jersey=7)
        seed_registration(conn, 2, 1, team_id=2, cup_id=20, jersey=3, position="MB")

    players = export(build)

    assert [s["cup_id"] for s in players[0]["seasons"]] == [20, 21]
    assert players[0]["seasons"][0] == {
        "cup_id": 20, "team_id": 2, "jersey_number": 3, "position": "MB", "games": 0,
        **dict.fromkeys(STAT_COLUMNS, 0),
    }
    assert players[1]["seasons"] == []


def test_season_uses_latest_registration_by_week_start_date_then_registration_id(export):
    def build(conn):
        for pid in (1, 2, 3):
            seed_player(conn, pid, f"球員{pid}")
        seed_registration(conn, 1, 1, team_id=1, week="W1", week_start="2025-11-01", jersey=7, position="OH")
        seed_registration(conn, 2, 1, team_id=2, week="W2", week_start="2025-11-08", jersey=9, position="MB")
        seed_registration(conn, 3, 2, team_id=1, week="W1", week_start="2025-11-01", jersey=5, position="OH")
        seed_registration(conn, 4, 2, team_id=1, week="W2", week_start=None, jersey=6, position="OP")
        seed_registration(conn, 5, 3, team_id=1, week="W1", week_start=None, jersey=1)
        seed_registration(conn, 6, 3, team_id=2, week="W2", week_start=None, jersey=2)

    players = export(build)

    latest = {p["player_id"]: p["seasons"][0] for p in players}
    assert (latest[1]["team_id"], latest[1]["jersey_number"], latest[1]["position"]) == (2, 9, "MB")
    assert (latest[2]["team_id"], latest[2]["jersey_number"], latest[2]["position"]) == (1, 5, "OH")
    assert (latest[3]["team_id"], latest[3]["jersey_number"], latest[3]["position"]) == (2, 2, None)


def test_player_who_changes_team_mid_season_sums_both_teams_under_latest_team(export):
    def build(conn):
        seed_player(conn, 1, "男甲")
        seed_registration(conn, 1, 1, team_id=1, week="W1", week_start="2025-11-01", jersey=7, position="OH")
        seed_registration(conn, 2, 1, team_id=2, week="W2", week_start="2025-11-08", jersey=9, position="OP")
        seed_stat(conn, 1, (3, 10, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4), match_date="2025-11-01")
        seed_stat(conn, 2, (2, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1), match_date="2025-11-08")

    [player] = export(build)

    [season] = player["seasons"]
    assert (season["team_id"], season["jersey_number"], season["position"]) == (2, 9, "OP")
    assert (season["games"], season["sets_played"], season["attack_total"], season["total_points"]) == (2, 5, 15, 5)


def test_empty_database_exports_empty_players(export):
    assert export(lambda conn: None) == []
