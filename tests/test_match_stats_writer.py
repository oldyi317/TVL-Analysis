import sqlite3
from pathlib import Path

import pytest

from src.etl import match_stats_writer
from src.etl.match_stats_writer import candidate_matches, write_match_stats
from src.etl.official_box_score import parse_box_score
from src.utils.db_config import get_connection
from tests.conftest import connect_seeded

FIXTURES = Path(__file__).parent / "fixtures"
GAMES = {
    ("F", 281): ("2026-10-24", "臺北鯨華", "高雄台電", 3, 4, 3, 1, 92, 89),
    ("M", 331): ("2026-10-25", "龍邁建設", "雲林美津濃", 5, 2, 0, 3, 56, 75),
}


def fixture(gender, game_id):
    prefix = "wgame" if gender == "F" else "game"
    return (FIXTURES / f"boxscore_{prefix}_{game_id}.html").read_text(encoding="utf-8")


def seed_match(conn, gender, game_id, cup_id=22, played=True, golden=0, date=None, as_game_id=None):
    match_date, home, away, home_id, away_id, hs, aws, ht, at = GAMES[(gender, game_id)]
    conn.execute(
        """INSERT INTO matches (game_id, gender, cup_id, match_date, round_name, is_golden_set,
                                home_team, away_team, home_team_id, away_team_id,
                                home_sets_won, away_sets_won, home_total, away_total)
           VALUES (?, ?, ?, ?, '例行賽 Week 1', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (as_game_id or game_id, gender, cup_id, date or match_date, golden, home, away, home_id, away_id,
         hs if played else None, aws if played else None, ht if played else None, at if played else None),
    )


def seed_roster_from_fixture(conn, gender, game_id, cup_id=22, skip=()):
    """依 fixture 的 box score 建 players 與 season_rosters（skip 內的姓名不建，模擬對不上名單）。"""
    _, _, _, home_id, away_id, *_ = GAMES[(gender, game_id)]
    teams = parse_box_score(fixture(gender, game_id))["teams"]
    for team, team_id in zip(teams, (home_id, away_id)):
        for p in team["players"]:
            if p["name"] in skip:
                continue
            pid = conn.execute("INSERT INTO players (name, gender) VALUES (?, ?)", (p["name"], gender)).lastrowid
            conn.execute(
                "INSERT INTO season_rosters (player_id, team_id, gender, cup_id, jersey_number, position) VALUES (?, ?, ?, ?, ?, ?)",
                (pid, team_id, gender, cup_id, p["jersey_number"], p["position"]),
            )


@pytest.fixture
def conn(tmp_db_path):
    c = connect_seeded(tmp_db_path)
    c.execute("PRAGMA foreign_keys = ON")
    yield c
    c.close()


def test_candidates_are_current_season_matches_with_result_and_no_stats(conn):
    seed_match(conn, "F", 281)
    seed_match(conn, "M", 331, played=False)
    seed_match(conn, "F", 281, cup_id=21, date="2025-10-24", as_game_id=201)
    conn.commit()
    found = [(m["gender"], m["game_id"], m["cup_id"]) for m in candidate_matches(conn)]
    assert found == [("F", 281, 22)]


def fetch_fixture(gender, game_id):
    return fixture(gender, game_id)


def stat_rows(conn, extra="1=1"):
    return conn.execute(
        f"""SELECT p.name, r.team_id, r.source, r.week_label, r.jersey_number, r.position,
                   s.match_date, s.opponent, s.sets_played, s.is_golden_set,
                   s.attack_points, s.attack_total, s.block_points, s.set_excellent, s.set_total, s.total_points
            FROM player_match_stats s
            JOIN roster_registrations r ON r.registration_id = s.registration_id
            JOIN players p ON p.player_id = r.player_id WHERE {extra}"""
    ).fetchall()


def test_writes_stats_and_match_page_registrations_for_a_candidate(conn):
    seed_match(conn, "F", 281)
    seed_roster_from_fixture(conn, "F", 281)
    conn.commit()

    summary = write_match_stats(conn, fetch_fixture)

    assert summary["written"] == [("F", 281)]
    rows = stat_rows(conn)
    assert len(rows) == 15 + 12, "全 0 列（鯨華 1 人、台電 4 人）不產生統計列"
    assert {r[8] for r in rows} == {None}, "官網無上場局數，sets_played 一律 NULL"
    assert {r[2] for r in rows} == {"match_page"}
    assert {(r[3], r[6], r[9]) for r in rows} == {("例行賽 Week 1", "2026-10-24", 0)}
    lin = next(r for r in rows if r[0] == "林良黛")
    assert (lin[1], lin[7], lin[10], lin[11], lin[15]) == (3, "高雄台電", 3, 14, 3)
    kim = next(r for r in rows if r[0] == "金彩娜")
    assert (kim[13], kim[14]) == (37, 111)
    registrations = conn.execute(
        "SELECT p.name, r.team_id, r.jersey_number, r.position, r.source, r.week_start_date FROM roster_registrations r "
        "JOIN players p ON p.player_id = r.player_id WHERE r.cup_id = 22"
    ).fetchall()
    assert len(registrations) == 32, "box score 上所有有名字的球員（含 0-0 者）都有週登錄"
    wang = next(r for r in registrations if r[0] == "王妤婷")
    assert (wang[1], wang[4], wang[5]) == (3, "match_page", "2026-10-24")
    assert all(r[2] is not None and r[3] is not None for r in registrations)
    assert candidate_matches(conn) == []


def test_second_run_is_a_no_op(conn):
    seed_match(conn, "F", 281)
    seed_roster_from_fixture(conn, "F", 281)
    conn.commit()
    write_match_stats(conn, fetch_fixture)
    before = conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0]

    def fail(gender, game_id):
        pytest.fail("已有統計的場次不得再抓頁")

    summary = write_match_stats(conn, fail)
    assert summary["written"] == []
    assert conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0] == before


def test_golden_set_page_is_written_with_flag_and_exported_under_parent_match(export_json):
    def build(c):
        seed_match(c, "M", 331)
        seed_match(c, "M", 331, golden=1, as_game_id=3310)
        seed_roster_from_fixture(c, "M", 331)
        c.commit()
        same_page = lambda gender, game_id: fixture("M", 331)  # 黃金局頁結構與正規頁相同
        assert write_match_stats(c, same_page)["written"] == [("M", 331), ("M", 3310)]
        assert c.execute("SELECT COUNT(*) FROM player_match_stats WHERE is_golden_set = 1").fetchone()[0] == 11 + 14
        assert c.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()[0] == 28, "同週同隊登錄只一筆"

    payload, _ = export_json(build, "box_scores")
    groups = {(b["cup_id"], b["group"]): b["games"] for b in payload["box_scores"]}
    [game] = groups[(22, "men")]
    assert game["game_id"] == 331
    assert [len(game["home"]), len(game["away"])] == [11, 14]
    assert [len(game["golden_set"]["home"]), len(game["golden_set"]["away"])] == [11, 14]


def test_unresolved_player_rolls_back_whole_match_until_force_partial(conn, caplog):
    seed_match(conn, "F", 281)
    seed_roster_from_fixture(conn, "F", 281, skip=("林良黛",))
    conn.commit()

    with caplog.at_level("ERROR"):
        summary = write_match_stats(conn, fetch_fixture)

    assert summary["written"] == [] and [k for k, _ in summary["skipped"]] == [("F", 281)]
    assert conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()[0] == 0, "整場 rollback，不留半寫登錄"
    assert any("[F/281]" in m and "林良黛" in m and "--force-partial" in m for m in caplog.messages)
    assert candidate_matches(conn), "下次每日更新重試"

    summary = write_match_stats(conn, fetch_fixture, force_partial=("F", 281))
    assert summary["written"] == [("F", 281)]
    assert len(stat_rows(conn)) == 14 + 12
    assert conn.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()[0] == 31


def test_gate_failure_skips_match_but_score_mismatch_only_warns(conn, caplog):
    seed_match(conn, "F", 281)
    seed_roster_from_fixture(conn, "F", 281)
    conn.execute("UPDATE matches SET home_total = 90")
    conn.commit()
    broken = fixture("F", 281).replace('title="win_points">3</td>', 'title="win_points">4</td>', 1)

    with caplog.at_level("WARNING"):
        summary = write_match_stats(conn, lambda g, i: broken)
    assert summary["written"] == []
    assert conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0] == 0
    assert any("[F/281]" in m and "閘門不通過" in m for m in caplog.messages)

    caplog.clear()
    with caplog.at_level("WARNING"):
        summary = write_match_stats(conn, fetch_fixture)
    assert summary["written"] == [("F", 281)]
    assert any("[F/281]" in m and "頁面總計 92" in m and "90" in m for m in caplog.messages)


def test_unexpected_exception_skips_only_that_match_and_warns_unresolved_teams(conn, caplog):
    seed_match(conn, "F", 281)
    seed_match(conn, "M", 331)
    seed_roster_from_fixture(conn, "F", 281)
    seed_roster_from_fixture(conn, "M", 331)
    conn.execute("UPDATE matches SET home_team_id = NULL WHERE game_id = 331")
    conn.execute(
        "INSERT INTO matches (game_id, gender, cup_id, match_date, home_team, away_team, home_sets_won, away_sets_won) "
        "VALUES (399, 'M', 22, '2026-10-26', '新隊', '三福產業', 3, 0)"
    )
    conn.commit()

    def fetch(gender, game_id):
        if gender == "M":
            raise KeyError("boom")
        return fixture(gender, game_id)

    with caplog.at_level("WARNING"):
        summary = write_match_stats(conn, fetch)
    assert summary["written"] == [("F", 281)]
    assert any("[M/399]" in m and "隊伍未解析" in m for m in caplog.messages)
    assert any("[M/331]" in m and "隊伍未解析" in m for m in caplog.messages)
    assert conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0] == 27

    conn.execute("UPDATE matches SET home_team_id = 5 WHERE game_id = 331")
    conn.commit()
    with caplog.at_level("ERROR"):
        summary = write_match_stats(conn, fetch)
    assert [k for k, _ in summary["skipped"]] == [("M", 331)]
    assert any("[M/331]" in m and "KeyError" in m for m in caplog.messages)


def test_cli_force_partial_resolves_candidate_and_writes(conn, tmp_db_path, monkeypatch):
    seed_match(conn, "F", 281)
    seed_roster_from_fixture(conn, "F", 281, skip=("林良黛",))
    conn.commit()
    monkeypatch.setattr(match_stats_writer, "get_connection", lambda: get_connection(db_path=tmp_db_path))
    monkeypatch.setattr(match_stats_writer, "fetch_page", fetch_fixture)

    assert match_stats_writer.main([]) == 1
    assert conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0] == 0
    assert match_stats_writer.main(["--force-partial", "281"]) == 0
    assert conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0] == 26
    with pytest.raises(SystemExit):
        match_stats_writer.main(["--force-partial", "999"])
