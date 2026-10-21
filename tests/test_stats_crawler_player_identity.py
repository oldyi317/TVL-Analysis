import logging
import sqlite3

import pytest
import requests

from src.etl import stats_crawler
from src.utils.constants import EXT_TEAM_MAPS
from tests.conftest import connect_seeded, seed_player

TEAM_F_ZHONGXIAN = 5
EXT_TEAM_F_ZHONGXIAN = {
    cup: next(ext for ext, v in teams.items() if v == (TEAM_F_ZHONGXIAN, "F"))
    for cup, teams in EXT_TEAM_MAPS.items() if cup in (20, 21)
}


def _record(date, **over):
    base = dict.fromkeys(
        ["attack_total", "attack_points", "block_points", "serve_total", "serve_points", "receive_total",
         "receive_excellent", "dig_total", "dig_excellent", "set_total", "set_excellent", "total_points"], 1)
    base.update(match_date=date, opponent="義力營造", sets_played=3, is_golden_set=0)
    base.update(over)
    return base


def _seed(db_path, cup_id, roster):
    conn = connect_seeded(db_path)
    date = "2024-11-02" if cup_id == 20 else "2025-11-01"
    conn.execute(
        "INSERT INTO matches (game_id, gender, cup_id, match_date, round_name, home_team, away_team) "
        "VALUES (1, 'F', ?, ?, '例行賽 Week 1', '新北中纖', '義力營造')",
        (cup_id, date),
    )
    for pid, name, jersey, position in roster:
        seed_player(conn, pid, name, "F")
        conn.execute(
            "INSERT INTO season_rosters (player_id, team_id, gender, cup_id, jersey_number, position) "
            "VALUES (?, ?, 'F', ?, ?, ?)",
            (pid, TEAM_F_ZHONGXIAN, cup_id, jersey, position),
        )
    conn.commit()
    return conn


@pytest.fixture
def run_main(tmp_db_path, monkeypatch):
    def run(cup_id, external):
        monkeypatch.setattr(stats_crawler, "get_connection", lambda: sqlite3.connect(tmp_db_path))
        ext_team = EXT_TEAM_F_ZHONGXIAN[cup_id]
        monkeypatch.setattr(
            stats_crawler, "fetch_player_list",
            lambda team_id, cup=None: (
                [{"ext_player_id": i, "name": n, "jersey_number": j} for i, (n, j) in enumerate(external, 100)]
                if team_id == ext_team and cup == cup_id else []
            ),
        )
        date = "2024-11-02" if cup_id == 20 else "2025-11-01"

        def fetch_stats(season, team_id, ext_pid):
            assert season.cup_id == cup_id
            return [_record(date)]

        monkeypatch.setattr(stats_crawler, "fetch_player_stats", fetch_stats)
        monkeypatch.setattr(stats_crawler.time, "sleep", lambda s: None)
        stats_crawler.main(cup_id=cup_id)
    return run


def _query(db_path, sql):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def test_name_match_uses_season_roster_and_records_stats(tmp_db_path, run_main, caplog):
    _seed(tmp_db_path, 20, [(1, "張瓈文", 2, "OH")]).close()

    with caplog.at_level(logging.INFO):
        run_main(20, [("張瓈文", 2)])

    assert _query(tmp_db_path, "SELECT COUNT(*) FROM player_match_stats") == [(1,)]
    assert _query(tmp_db_path, "SELECT player_id, cup_id FROM roster_registrations") == [(1, 20)]
    assert "以背號對上：無" in caplog.text
    assert "略過：無" in caplog.text


def test_jersey_match_when_names_differ_and_listed_in_report(tmp_db_path, run_main, caplog):
    _seed(tmp_db_path, 20, [(1, "Darin Pinsuwan", 16, "OH")]).close()

    with caplog.at_level(logging.INFO):
        run_main(20, [("Darin", 16)])

    assert _query(tmp_db_path, "SELECT player_id FROM roster_registrations") == [(1,)]
    assert _query(tmp_db_path, "SELECT COUNT(*) FROM player_match_stats") == [(1,)]
    assert _query(tmp_db_path, "SELECT COUNT(*) FROM players") == [(1,)]
    line = next(r.getMessage() for r in caplog.records if "以背號對上" in r.getMessage() and "Darin Pinsuwan" in r.getMessage())
    assert "Darin" in line and "新北中纖" in line and "16" in line


def test_unknown_player_is_skipped_and_reported(tmp_db_path, run_main, caplog):
    _seed(tmp_db_path, 20, [(1, "張瓈文", 2, "OH")]).close()

    with caplog.at_level(logging.INFO):
        run_main(20, [("陌生人", 77)])

    assert _query(tmp_db_path, "SELECT COUNT(*) FROM players") == [(1,)]
    assert _query(tmp_db_path, "SELECT COUNT(*) FROM roster_registrations") == [(0,)]
    assert _query(tmp_db_path, "SELECT COUNT(*) FROM player_match_stats") == [(0,)]
    line = next(r.getMessage() for r in caplog.records if "略過（" in r.getMessage())
    assert "新北中纖" in line and "77" in line and "陌生人" in line


def test_name_without_jersey_and_no_name_match_is_skipped(tmp_db_path, run_main):
    _seed(tmp_db_path, 20, [(1, "張瓈文", 2, "OH")]).close()

    run_main(20, [("陌生人", None)])

    assert _query(tmp_db_path, "SELECT COUNT(*) FROM player_match_stats") == [(0,)]


def test_backfill_registration_takes_jersey_and_position_from_season_roster(tmp_db_path, run_main):
    _seed(tmp_db_path, 20, [(1, "張瓈文", 2, "OH")]).close()

    run_main(20, [("張瓈文", 2)])

    assert _query(tmp_db_path, "SELECT source, jersey_number, position, cup_id FROM roster_registrations") == [
        ("backfill", 2, "OH", 20)
    ]


def test_cup_without_season_roster_keeps_legacy_insert_and_rerun_is_stable(tmp_db_path, run_main, caplog):
    _seed(tmp_db_path, 21, []).close()

    with caplog.at_level(logging.INFO):
        run_main(21, [("新球員", 9)])
        run_main(21, [("新球員", 9)])

    assert _query(tmp_db_path, "SELECT name FROM players") == [("新球員",)]
    assert _query(tmp_db_path, "SELECT COUNT(*) FROM player_match_stats") == [(1,)]
    assert _query(tmp_db_path, "SELECT source, jersey_number FROM roster_registrations") == [("backfill", None)]
    assert "cup_id=21 無賽季報名名單，沿用現行姓名比對與動態新增" in caplog.text


def test_crawl_all_rosters_skips_unknown_player_without_inserting(tmp_db_path, monkeypatch):
    conn = _seed(tmp_db_path, 20, [(1, "Darin Pinsuwan", 16, "OH")])
    monkeypatch.setattr(stats_crawler, "fetch_match_list", lambda cup_id: [{"match_id": 1, "label": ""}])
    monkeypatch.setattr(stats_crawler.time, "sleep", lambda s: None)
    rows = [
        {"match_date": "2024-11-02", "title_text": "t", "team_id": 5, "team_gender": "F",
         "jersey_number": j, "name": n, "position": "OH"}
        for n, j in (("Darin", 16), ("陌生人", 77))
    ]
    monkeypatch.setattr(stats_crawler, "fetch_match_roster", lambda conn, season, match_id: rows)

    stats = stats_crawler.crawl_all_rosters(conn, cup_id=20)

    assert stats["skipped_players"] == 1
    assert stats["registrations_upserted"] == 1
    assert stats["new_players"] == 0
    assert conn.execute("SELECT COUNT(*) FROM players").fetchone() == (1,)
    assert conn.execute("SELECT player_id FROM roster_registrations").fetchall() == [(1,)]
    conn.close()


def test_crawl_all_rosters_legacy_cup_inserts_new_player(tmp_db_path, monkeypatch):
    conn = _seed(tmp_db_path, 21, [])
    monkeypatch.setattr(stats_crawler, "fetch_match_list", lambda cup_id: [{"match_id": 1, "label": ""}])
    monkeypatch.setattr(stats_crawler.time, "sleep", lambda s: None)
    rows = [{"match_date": "2025-11-01", "title_text": "t", "team_id": 5, "team_gender": "F",
             "jersey_number": 9, "name": "新球員", "position": "OH"}]
    monkeypatch.setattr(stats_crawler, "fetch_match_roster", lambda conn, season, match_id: rows)

    stats = stats_crawler.crawl_all_rosters(conn, cup_id=21)

    assert stats["new_players"] == 1
    assert stats["skipped_players"] == 0
    conn.close()


class _Resp:
    def __init__(self, text):
        self.text = text


@pytest.mark.parametrize("label, name, jersey", [
    ("No.16-Darin", "Darin", 16),
    ("No.5-Jean-Paul", "Jean-Paul", 5),
    ("無前綴", "無前綴", None),
])
def test_fetch_player_list_parses_jersey_and_name(monkeypatch, label, name, jersey):
    seen = {}

    def fake_get(url, params, headers, timeout):
        seen.update(params)
        return _Resp(f"<select id='playerSelect'><option value='23'>{label}</option></select>")

    monkeypatch.setattr(requests, "get", fake_get)

    assert stats_crawler.fetch_player_list(9, cup_id=20) == [
        {"ext_player_id": 23, "name": name, "jersey_number": jersey}
    ]
    assert seen == {"CupID": 20, "TeamID": 9}
