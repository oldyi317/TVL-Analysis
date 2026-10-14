import pytest

from tests.conftest import seed_player, seed_registration


@pytest.fixture
def export(export_json):
    return lambda build: export_json(build, "rosters")[0]["rosters"]


def test_rosters_grouped_by_cup_group_team_with_sorted_weeks_and_players(export):
    def build(conn):
        for pid, gender in ((1, "M"), (2, "M"), (3, "M"), (4, "F"), (5, "M")):
            seed_player(conn, pid, f"球員{pid}", gender)
        seed_registration(conn, 1, 1, 1, week="例行賽 Week 2", week_start="2025-11-08", jersey=9, position="OH")
        seed_registration(conn, 2, 2, 1, week="例行賽 Week 1", week_start="2025-11-01", jersey=None, source="backfill")
        seed_registration(conn, 3, 3, 1, week="例行賽 Week 1", week_start="2025-11-01", jersey=4, position="MB")
        seed_registration(conn, 4, 5, 1, week="例行賽 Week 1", week_start="2025-11-01", jersey=4, position="OP")
        seed_registration(conn, 5, 1, 1, week="加賽", week_start=None, jersey=9)
        seed_registration(conn, 6, 1, 1, week="例行賽 Week 0", week_start=None, jersey=9)
        seed_registration(conn, 7, 4, 3, gender="F", week="例行賽 Week 1", week_start="2025-11-01", jersey=1, position="S")
        seed_registration(conn, 8, 1, 2, cup_id=20, week="例行賽 Week 1", week_start="2024-11-23", jersey=8)

    rosters = export(build)

    assert [(r["cup_id"], r["group"], r["team_id"]) for r in rosters] == [
        (20, "men", 2), (21, "men", 1), (21, "women", 3),
    ]
    weeks = rosters[1]["weeks"]
    assert [(w["week_label"], w["week_start_date"]) for w in weeks] == [
        ("例行賽 Week 1", "2025-11-01"), ("例行賽 Week 2", "2025-11-08"),
        ("例行賽 Week 0", None), ("加賽", None),
    ]
    assert weeks[0]["players"] == [
        {"player_id": 3, "jersey_number": 4, "position": "MB", "source": "match_page"},
        {"player_id": 5, "jersey_number": 4, "position": "OP", "source": "match_page"},
        {"player_id": 2, "jersey_number": None, "position": None, "source": "backfill"},
    ]
    assert rosters[2]["weeks"][0]["players"] == [
        {"player_id": 4, "jersey_number": 1, "position": "S", "source": "match_page"},
    ]


def test_empty_database_exports_empty_rosters(export):
    assert export(lambda conn: None) == []
