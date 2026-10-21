import pandas as pd
import pytest

from src.etl.db_loader import sync_season_rosters
from tests.conftest import seed_player

TODAY = "2026-10-19"


@pytest.fixture
def export_rosters(export_json):
    return lambda build: export_json(build, "rosters")[0]["rosters"]


def seed_match(conn, cup_id, game_id=1, gender="M", match_date="2026-10-24"):
    conn.execute(
        "INSERT INTO matches (game_id, gender, cup_id, match_date, home_team, away_team) VALUES (?, ?, ?, ?, ?, ?)",
        (game_id, gender, cup_id, match_date, "屏東台電", "雲林美津濃"),
    )


def roster_df(*rows):
    return pd.DataFrame(
        [{"gender": g, "team_id": site_id, "team_name": team, "jersey_number": jersey, "name": name, "position": pos}
         for g, site_id, team, jersey, name, pos in rows]
    )


def season_roster(rosters, cup_id, team_id, group="men"):
    [entry] = [r for r in rosters if (r["cup_id"], r["group"], r["team_id"]) == (cup_id, group, team_id)]
    return entry["season_roster"]


def test_team_resolved_by_name_not_site_number_and_written_to_latest_cup(export_rosters):
    def build(conn):
        seed_match(conn, cup_id=21, game_id=1, match_date="2025-11-01")
        seed_match(conn, cup_id=22, game_id=2)
        seed_player(conn, 1, "可利甲")
        seed_player(conn, 2, "可利乙")
        sync_season_rosters(conn, TODAY, roster_df(
            ("M", 9, "新竹可利工程師", 12, "可利乙", "S"),
            ("M", 9, "新竹可利工程師", 3, "可利甲", "OH"),
        ))

    rosters = export_rosters(build)

    assert [(r["cup_id"], r["team_id"]) for r in rosters] == [(22, 12)]
    assert rosters[0]["weeks"] == []
    assert season_roster(rosters, 22, 12) == [
        {"player_id": 1, "jersey_number": 3, "position": "OH"},
        {"player_id": 2, "jersey_number": 12, "position": "S"},
    ]


def test_resync_replaces_whole_team_but_skips_teams_with_no_players_or_unknown_name(export_rosters):
    def build(conn):
        seed_match(conn, cup_id=22)
        for pid, name in [(1, "台電甲"), (2, "台電乙"), (3, "台電丙"), (4, "美津濃甲")]:
            seed_player(conn, pid, name)
        sync_season_rosters(conn, TODAY, roster_df(
            ("M", 1, "屏東台電", 1, "台電甲", "OH"),
            ("M", 1, "屏東台電", 2, "台電乙", "MB"),
            ("M", 2, "雲林美津濃", 5, "美津濃甲", "L"),
        ))
        sync_season_rosters(conn, TODAY, roster_df(
            ("M", 1, "屏東台電", 9, "台電甲", "OP"),
            ("M", 1, "屏東台電", 3, "台電丙", "S"),
            ("M", 2, "雲林美津濃", 5, "官網未建檔", "L"),
            ("M", 99, "不存在隊", 1, "台電乙", "OH"),
        ))

    rosters = export_rosters(build)

    assert season_roster(rosters, 22, 1) == [
        {"player_id": 3, "jersey_number": 3, "position": "S"},
        {"player_id": 1, "jersey_number": 9, "position": "OP"},
    ]
    assert season_roster(rosters, 22, 2) == [{"player_id": 4, "jersey_number": 5, "position": "L"}]
    assert {r["team_id"] for r in rosters} == {1, 2}


def test_skips_sync_when_latest_cup_has_already_ended(export_rosters):
    """休賽期 matches 最新仍是舊季，官網若先換新季名單，不能蓋掉舊季的賽季報名名單。"""
    def build(conn):
        seed_match(conn, cup_id=21, match_date="2026-03-21")
        seed_player(conn, 1, "台電甲")
        sync_season_rosters(conn, "2026-08-01", roster_df(("M", 1, "屏東台電", 1, "台電甲", "OH")))

    assert export_rosters(build) == []


def test_explicit_cup_id_writes_that_season_even_after_it_ended(export_rosters):
    """歷史賽季名單（如企排 20 的圖檔轉錄）指定 cup_id 直接寫入該季，不受「最新且未結束」規則限制。"""
    def build(conn):
        seed_match(conn, cup_id=21, match_date="2026-03-21")
        seed_player(conn, 1, "連莊甲")
        seed_player(conn, 2, "連莊乙")
        sync_season_rosters(conn, "2026-10-20", roster_df(
            ("M", 10, "連莊", 16, "連莊甲", None),
            ("M", 10, "連莊", 3, "連莊乙", None),
            ("M", 10, "連莊", 7, "連莊乙", None),
        ).drop(columns=["position"]), cup_id=20)

    rosters = export_rosters(build)

    assert [(r["cup_id"], r["team_id"]) for r in rosters] == [(20, 10)]
    assert season_roster(rosters, 20, 10) == [
        {"player_id": 2, "jersey_number": 7, "position": None},
        {"player_id": 1, "jersey_number": 16, "position": None},
    ]


def test_team_name_is_resolved_within_its_own_gender(export_rosters):
    def build(conn):
        seed_match(conn, cup_id=22)
        seed_player(conn, 1, "同名", "F")
        seed_player(conn, 2, "同名", "M")
        sync_season_rosters(conn, TODAY, roster_df(("F", 1, "屏東台電", 1, "同名", "OH")))

    assert export_rosters(build) == []
