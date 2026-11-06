import json
from pathlib import Path

import pytest

from src.etl import season_backfill
from src.etl.season_backfill import (
    BackfillBlocked, build_review_list, ensure_backfill_season, fetch_match_page, main, parse_match_page,
    run_backfill,
)
from src.utils.constants import BACKFILL_SEASONS, EXT_BASE, EXT_TEAM_MAPS, EXT_TEAM_SEASON_NAMES
from tests.conftest import connect_seeded, seed_player, seed_registration

FIXTURES = Path(__file__).parent / "fixtures" / "cup19"
MATCH_IDS = [1, 2, 3, 101]
STAT_FIELDS = [
    "attack_points", "attack_total", "block_points", "serve_points", "serve_total", "receive_excellent",
    "receive_total", "dig_excellent", "dig_total", "set_excellent", "set_total", "total_points",
]


def _html(match_id):
    return (FIXTURES / f"match_{match_id}_response.html").read_text(encoding="utf-8")


@pytest.fixture
def season(tmp_db_path):
    conn = connect_seeded(tmp_db_path)
    s = ensure_backfill_season(conn, 19)
    conn.commit()
    conn.close()
    return s


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(season_backfill.time, "sleep", lambda s: None)


@pytest.fixture
def fake_fetch(monkeypatch, no_sleep):
    monkeypatch.setattr(season_backfill, "fetch_match_page", lambda cup_id, mid: _html(mid) if mid in MATCH_IDS else None)


# 1
def test_parse_match_1(season):
    page = parse_match_page(_html(1), 19, season)
    assert (page.gender, page.match_date, page.start_time, page.venue) == ("M", "2023-10-14", "13:00", "成功大學體育館")
    assert page.official_no == 1 and page.round_name == "例行賽 Week 1" and page.game_label == "Game 1"
    assert page.is_golden_set == 0
    home, away = page.home, page.away
    assert (home.ext_team_id, home.team_id, home.name) == (4, 9, "臺北Conti")
    assert (home.sets, home.total, home.sets_won) == ([29, 25, 25, None, None], 79, 3)
    assert (away.ext_team_id, away.team_id, away.name) == (5, 5, "桃園臺灣產險")
    assert (away.sets, away.total, away.sets_won) == ([27, 21, 16, None, None], 64, 0)
    assert len(home.players) == 14 and len(away.players) == 14
    p = next(x for x in home.players if x["name"] == "陳朝璽")
    assert (p["jersey_number"], p["position"]) == (1, "S")
    assert [p[f] for f in STAT_FIELDS] == [4, 4, 1, 1, 10, 0, 0, 0, 1, 1, 37, 6]


def test_official_no_with_gender_prefix(season):
    """cup 18 的外部標題寫「編號：男1」「編號：女12」，組別字要略過，game_label 仍為官網的「Game N」。"""
    page = parse_match_page(_html(1).replace("編號：1 ", "編號：男1 "), 19, season)
    assert page.official_no == 1 and page.game_label == "Game 1"


def test_playoff_title_keeps_venue(season):
    """季後賽標題「季後賽(場館)」沒有週次，場館仍要取到。"""
    page = parse_match_page(_html(1).replace("第1週(成功大學體育館)", "季後賽(台北市立大學體育館)"), 19, season)
    assert page.week is None and page.round_name is None and page.stage == "季後賽"
    assert page.venue == "台北市立大學體育館"


# 2
def test_parse_match_101(season):
    page = parse_match_page(_html(101), 19, season)
    assert (page.gender, page.match_date, page.round_name, page.game_label) == ("F", "2024-02-18", "例行賽 Week 15", "Game 89")
    assert (page.away.ext_team_id, page.away.team_id) == (11, 7)
    assert len(page.home.players) + len(page.away.players) == 29


# 3
def test_unknown_ext_team_id_raises(season):
    html = _html(1).replace("TeamID=4'", "TeamID=99'")
    with pytest.raises(RuntimeError, match="99"):
        parse_match_page(html, 19, season)


def test_all_star_returns_none(season):
    assert parse_match_page(_html(1).replace("男子組 第1週", "男子組 明星賽 第1週"), 19, season) is None


def _all_rows(season):
    pages = [parse_match_page(_html(m), 19, season) for m in MATCH_IDS]
    return pages, [(pg, side, pl) for pg in pages for side in (pg.home, pg.away) for pl in side.players]


def _counts(conn):
    return tuple(
        conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("matches", "roster_registrations", "player_match_stats", "players")
    )


# 4
def test_end_to_end_and_idempotent(tmp_db_path, season, fake_fetch):
    conn = connect_seeded(tmp_db_path)
    result = run_backfill(conn, 19, match_ids=MATCH_IDS)
    assert conn.execute("SELECT name FROM seasons WHERE cup_id = 19").fetchone() == ("2023-24",)
    names = conn.execute("SELECT team_id, gender, team_name FROM team_season_names WHERE cup_id = 19 ORDER BY team_id").fetchall()
    assert names == [(5, "M", "桃園臺灣產險"), (7, "F", "凱撒飯店連鎖")]
    assert [r[0] for r in conn.execute("SELECT game_id FROM matches ORDER BY game_id")] == [19001, 19002, 19003, 19101]
    row = conn.execute("SELECT home_set1, home_total, home_sets_won, round_name, home_team, home_team_id FROM matches WHERE game_id = 19001").fetchone()
    assert row == (29, 79, 3, "例行賽 Week 1", "臺北Conti", 9)

    _, rows = _all_rows(season)
    expected_regs = {(pl["name"], side.team_id, pg.gender, pg.round_name) for pg, side, pl in rows}
    counts = _counts(conn)
    assert counts[0] == 4
    assert counts[1] == len(expected_regs)
    assert counts[2] == len(rows)
    assert conn.execute("SELECT COUNT(*) FROM roster_registrations WHERE cup_id != 19").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM player_match_stats s JOIN roster_registrations r USING (registration_id) WHERE r.cup_id = 19"
    ).fetchone()[0] == counts[2]
    assert (result.new_matches, result.stats_inserted) == (4, len(rows))

    again = run_backfill(conn, 19, match_ids=MATCH_IDS)
    assert _counts(conn) == counts
    assert (again.new_matches, again.stats_inserted) == (0, 0)
    conn.close()


def test_week_unresolved_only_writes_match(tmp_db_path, season, monkeypatch, no_sleep):
    html = _html(1).replace("第1週(成功大學體育館) ", "(成功大學體育館) ")
    monkeypatch.setattr(season_backfill, "fetch_match_page", lambda c, m: html if m == 1 else None)
    conn = connect_seeded(tmp_db_path)
    result = run_backfill(conn, 19, match_ids=[1])
    assert _counts(conn) == (1, 0, 0, 0)
    assert conn.execute("SELECT round_name FROM matches").fetchone() == (None,)
    assert [(u["kind"], u["match_id"], u["game_id"]) for u in result.unresolved] == [("week_unresolved", 1, 19001)]
    conn.close()


# 5
def test_review_list(tmp_db_path, season, fake_fetch):
    conn = connect_seeded(tmp_db_path)
    seed_player(conn, 9001, "郭柏毅", "M")
    seed_registration(conn, 9001, 9001, 1, cup_id=21)
    seed_player(conn, 9002, "陳朝璽", "F")
    seed_registration(conn, 9002, 9002, 3, cup_id=21, gender="F")
    conn.commit()
    run_backfill(conn, 19, match_ids=MATCH_IDS)

    guo = conn.execute("SELECT player_id FROM players WHERE name = '郭柏毅'").fetchall()
    assert guo == [(9001,)]
    assert conn.execute("SELECT COUNT(*) FROM roster_registrations WHERE player_id = 9001 AND cup_id = 19").fetchone()[0] >= 1

    items = build_review_list(conn, 19)
    by_kind = {}
    for it in items:
        assert {"kind", "cup_id", "player_id", "name", "gender", "team_ids", "evidence"} <= set(it)
        by_kind.setdefault(it["kind"], set()).add(it["name"])
    assert "郭柏毅" in by_kind["same_name_other_team_in_history"]
    assert "陳朝璽" in by_kind["same_name_other_group"]
    other = next(i for i in items if i["kind"] == "same_name_other_group")
    assert {e["player_id"] for e in other["evidence"]} == {9002}
    assert "周冠宇" in by_kind["new_player"]
    assert "郭柏毅" not in by_kind["new_player"]

    _, rows = _all_rows(season)
    reg_players = {r[0] for r in conn.execute("SELECT DISTINCT p.name FROM roster_registrations r JOIN players p USING (player_id) WHERE cup_id = 19")}
    assert {pl["name"] for _, _, pl in rows} == reg_players
    new_names = by_kind["new_player"]
    assert new_names == reg_players - {"郭柏毅"}

    assert build_review_list(conn, 19) == items
    run_backfill(conn, 19, match_ids=MATCH_IDS)
    assert build_review_list(conn, 19) == items
    conn.close()


def test_review_same_name_multiple_teams(tmp_db_path, season):
    conn = connect_seeded(tmp_db_path)
    seed_player(conn, 1, "王小明", "M")
    seed_registration(conn, 1, 1, 1, cup_id=19, week="例行賽 Week 1", week_start="2023-10-14")
    seed_registration(conn, 2, 1, 2, cup_id=19, week="例行賽 Week 2", week_start="2023-10-21")
    items = build_review_list(conn, 19)
    hit = [i for i in items if i["kind"] == "same_name_multiple_teams_in_season"]
    assert [(i["name"], sorted(i["team_ids"])) for i in hit] == [("王小明", [1, 2])]
    conn.close()


# 6
class FakeResponse:
    def __init__(self, status=200, text="", headers=None):
        self.status_code, self.text, self.headers, self.encoding = status, text, headers or {}, None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _fake_get(monkeypatch, responses):
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append((url, params))
        return responses[len(calls) - 1]

    monkeypatch.setattr(season_backfill.requests, "get", get)
    return calls


def test_403_blocks_and_keeps_processed(tmp_db_path, season, monkeypatch, no_sleep):
    calls = _fake_get(monkeypatch, [FakeResponse(200, _html(1)), FakeResponse(403, "forbidden"), FakeResponse(200, _html(2))])
    conn = connect_seeded(tmp_db_path)
    result = run_backfill(conn, 19, match_ids=[1, 2, 3])
    assert (result.last_match_id, result.blocked_at) == (1, 2) and "403" in result.blocked_reason
    assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1
    assert len(calls) == 2
    conn.close()


def test_blocked_review_keeps_week_unresolved(tmp_db_path, tmp_path, monkeypatch, no_sleep):
    connect_seeded(tmp_db_path).close()
    html = _html(1).replace("第1週(成功大學體育館) ", "(成功大學體育館) ")
    _fake_get(monkeypatch, [FakeResponse(200, html), FakeResponse(403)])
    out = tmp_path / "r.json"
    rc = main(["--cup-id", "19", "--db", str(tmp_db_path), "--review-out", str(out), "--match-ids", "1,2"])
    assert rc == 2
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [(d["kind"], d["match_id"]) for d in data] == [("week_unresolved", 1)]


@pytest.mark.parametrize("resp", [
    FakeResponse(200, "<h3>x</h3>", {"cf-mitigated": "challenge"}),
    FakeResponse(503, "busy"),
    FakeResponse(200, "<script src='/cdn-cgi/challenge-platform/x'></script>"),
])
def test_challenge_blocks(monkeypatch, no_sleep, resp):
    _fake_get(monkeypatch, [resp])
    with pytest.raises(BackfillBlocked):
        fetch_match_page(19, 1)


@pytest.mark.parametrize("status", [404, 500])
def test_missing_match_returns_none(monkeypatch, no_sleep, status):
    _fake_get(monkeypatch, [FakeResponse(status, "")])
    assert fetch_match_page(19, 1) is None


def test_non_match_page_returns_none(monkeypatch, no_sleep):
    _fake_get(monkeypatch, [FakeResponse(200, "<h3>錯誤</h3>")])
    assert fetch_match_page(19, 1) is None


def test_fetch_uses_statistics_domain(monkeypatch, no_sleep):
    calls = _fake_get(monkeypatch, [FakeResponse(200, _html(1))])
    assert fetch_match_page(19, 1) == _html(1)
    url, params = calls[0]
    assert url == f"{EXT_BASE}/_handler/Match.ashx" and params == {"CupID": 19, "MatchID": 1, "SetNum": 0}


# 7
def test_sleep_interval_per_request(monkeypatch):
    slept = []
    monkeypatch.setattr(season_backfill.time, "sleep", slept.append)
    _fake_get(monkeypatch, [FakeResponse(404), FakeResponse(404), FakeResponse(200, _html(1))])
    for mid in (1, 2, 3):
        fetch_match_page(19, mid)
    assert len(slept) == 3 and all(1 <= s <= 2 for s in slept)


def test_gap_limit_stops_scan(tmp_db_path, season, monkeypatch, no_sleep):
    seen = []
    monkeypatch.setattr(season_backfill, "fetch_match_page", lambda c, m: seen.append(m) or (_html(1) if m == 1 else None))
    conn = connect_seeded(tmp_db_path)
    result = run_backfill(conn, 19, gap_limit=3)
    assert seen == [1, 2, 3, 4] and result.pages == 1
    conn.close()


def test_match_ids_spec():
    assert season_backfill.parse_match_ids("1-3,101") == [1, 2, 3, 101]


# 8
@pytest.mark.parametrize("cup_id", [22, 17])
def test_cli_guard(tmp_path, cup_id):
    with pytest.raises(SystemExit):
        main(["--cup-id", str(cup_id), "--db", str(tmp_path / "x.db"), "--review-out", str(tmp_path / "r.json")])


def test_cli_end_to_end(tmp_db_path, tmp_path, fake_fetch):
    connect_seeded(tmp_db_path).close()
    out = tmp_path / "review.json"
    assert main(["--cup-id", "19", "--db", str(tmp_db_path), "--review-out", str(out), "--match-ids", "1,2,3,101"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data and {d["kind"] for d in data} >= {"new_player"}


def test_cli_blocked_exit_code(tmp_db_path, tmp_path, monkeypatch, no_sleep):
    connect_seeded(tmp_db_path).close()
    _fake_get(monkeypatch, [FakeResponse(403)])
    rc = main(["--cup-id", "19", "--db", str(tmp_db_path), "--review-out", str(tmp_path / "r.json"), "--match-ids", "1"])
    assert rc != 0


def test_ext_team_map_19_targets_exist(tmp_db_path):
    conn = connect_seeded(tmp_db_path)
    teams = set(conn.execute("SELECT team_id, gender FROM teams").fetchall())
    assert len(EXT_TEAM_MAPS[19]) == 11
    assert set(EXT_TEAM_MAPS[19].values()) <= teams
    conn.close()


def test_ext_team_map_18_targets_exist(tmp_db_path):
    """cup 18（2022-23）：男 6 隊（臺中太陽神為新非現役隊）、女 5 隊（愛山林為義力營造曾用名，沿用 team_id）。"""
    conn = connect_seeded(tmp_db_path)
    teams = {(t, g): active for t, g, active in conn.execute("SELECT team_id, gender, is_active FROM teams")}
    assert len(EXT_TEAM_MAPS[18]) == 11
    assert set(EXT_TEAM_MAPS[18].values()) <= set(teams)
    assert EXT_TEAM_MAPS[18][3] == (14, "M") and teams[(14, "M")] == 0
    assert EXT_TEAM_MAPS[18][11] == (7, "F")
    assert BACKFILL_SEASONS[18] == ("2022-23", "2022-07-01", "2023-06-30")
    assert EXT_TEAM_SEASON_NAMES[18][(5, "M")] == ("桃園臺灣產險", "桃園臺產")
    assert EXT_TEAM_SEASON_NAMES[18][(7, "F")] == ("愛山林", "愛山林")
    aliases = {a: (t, g) for a, t, g in conn.execute("SELECT alias, team_id, gender FROM team_aliases")}
    assert aliases["臺中太陽神"] == (14, "M")
    assert aliases["愛山林"] == (7, "F")
    assert aliases["Conti"] == (9, "M") and aliases["conti"] == (9, "M")
    assert aliases["雲林Mizuno"] == (2, "M")
    conn.close()


def test_ext_team_map_17_targets_exist(tmp_db_path):
    """cup 17（2021-22）：男女各 5 隊；外部 MatchID 男 1–50、女 51–100；官網隊名「桃園臺產隼鷹」要能解析、「中國人纖」為新北中纖曾用名。"""
    conn = connect_seeded(tmp_db_path)
    teams = {(t, g): active for t, g, active in conn.execute("SELECT team_id, gender, is_active FROM teams")}
    assert len(EXT_TEAM_MAPS[17]) == 10
    assert set(EXT_TEAM_MAPS[17].values()) <= set(teams)
    assert EXT_TEAM_MAPS[17][3] == (14, "M") and teams[(14, "M")] == 0
    assert EXT_TEAM_MAPS[17][4] == (9, "M")
    assert EXT_TEAM_MAPS[17][9] == (13, "F") and EXT_TEAM_MAPS[17][10] == (7, "F")
    assert BACKFILL_SEASONS[17] == ("2021-22", "2021-07-01", "2022-06-30")
    assert EXT_TEAM_SEASON_NAMES[17][(5, "M")] == ("桃園臺灣產險", "桃園臺產")
    assert EXT_TEAM_SEASON_NAMES[17][(5, "F")] == ("中國人纖", "中國人纖")
    assert EXT_TEAM_SEASON_NAMES[17][(7, "F")] == ("愛山林", "愛山林")
    aliases = {a: (t, g, src) for a, t, g, src in conn.execute("SELECT alias, team_id, gender, source FROM team_aliases")}
    assert aliases["桃園臺產隼鷹"] == (5, "M", "match")
    assert aliases["中國人纖"] == (5, "F", "former")
    assert aliases["愛山林建設"] == (7, "F", "former")  # 單場數據的對手欄寫外部隊名，匯出器要能解析
    conn.close()


def test_ext_team_map_16_targets_exist(tmp_db_path):
    """cup 16（2020-21）：男女各 5 隊；外部 MatchID 男 1–60、女 61–120、標題「編號：1」無男女字；臺中長力為臺中太陽神曾用名（名單 11/13 重疊），沿用 (14, M)。"""
    conn = connect_seeded(tmp_db_path)
    teams = {(t, g): active for t, g, active in conn.execute("SELECT team_id, gender, is_active FROM teams")}
    assert len(EXT_TEAM_MAPS[16]) == 10
    assert set(EXT_TEAM_MAPS[16].values()) <= set(teams)
    assert EXT_TEAM_MAPS[16][3] == (14, "M") and teams[(14, "M")] == 0
    assert EXT_TEAM_MAPS[16][4] == (9, "M")
    assert EXT_TEAM_MAPS[16][8] == (5, "F") and EXT_TEAM_MAPS[16][10] == (7, "F")
    assert BACKFILL_SEASONS[16] == ("2020-21", "2020-07-01", "2021-06-30")
    assert EXT_TEAM_SEASON_NAMES[16][(5, "M")] == ("桃園臺灣產險", "桃園臺產")
    assert EXT_TEAM_SEASON_NAMES[16][(14, "M")] == ("臺中長力", "臺中長力")
    assert EXT_TEAM_SEASON_NAMES[16][(5, "F")] == ("中國人纖", "中國人纖")
    assert EXT_TEAM_SEASON_NAMES[16][(7, "F")] == ("愛山林", "愛山林")
    aliases = {a: (t, g, src) for a, t, g, src in conn.execute("SELECT alias, team_id, gender, source FROM team_aliases")}
    assert aliases["臺中長力"] == (14, "M", "former")
    assert aliases["台電男排"] == (1, "M", "match") and aliases["台電女排"] == (4, "F", "match")  # 單場數據對手欄寫外部隊名
    conn.close()


def test_ensure_backfill_season_applies_team_seed(tmp_db_path):
    conn = connect_seeded(tmp_db_path)
    conn.execute("DELETE FROM team_aliases WHERE team_id = 13 AND gender = 'F'")
    conn.execute("DELETE FROM teams WHERE team_id = 13 AND gender = 'F'")
    ensure_backfill_season(conn, 19)
    assert conn.execute("SELECT team_name FROM teams WHERE team_id = 13 AND gender = 'F'").fetchone() == ("極速超跑",)
    conn.close()


def test_app_error_page_with_injected_script_is_not_block(monkeypatch, no_sleep):
    body = "<title>並未將物件參考設定為物件的執行個體。</title><script src='/cdn-cgi/challenge-platform/x'></script>"
    _fake_get(monkeypatch, [FakeResponse(500, body)])
    assert fetch_match_page(19, 1) is None


def test_week_from_db_round_name_when_title_has_no_week(tmp_db_path, season, monkeypatch, no_sleep):
    html = _html(1).replace("第1週(成功大學體育館) ", "(成功大學體育館) ")
    monkeypatch.setattr(season_backfill, "fetch_match_page", lambda c, m: html if m == 1 else None)
    conn = connect_seeded(tmp_db_path)
    run_backfill(conn, 19, match_ids=[1])
    conn.execute("UPDATE matches SET round_name = '季後賽 Week 19' WHERE game_id = 19001")
    conn.commit()
    result = run_backfill(conn, 19, match_ids=[1])
    assert result.unresolved == []
    assert _counts(conn)[1] > 0
    assert conn.execute("SELECT DISTINCT week_label FROM roster_registrations").fetchall() == [("季後賽 Week 19",)]
    conn.close()


def test_unplayed_match_is_skipped(tmp_db_path, season, monkeypatch, no_sleep):
    page = parse_match_page(_html(1), 19, season)
    for side in (page.home, page.away):
        side.sets, side.total, side.sets_won = [None] * 5, 0, 0
    monkeypatch.setattr(season_backfill, "fetch_match_page", lambda c, m: "x" if m == 1 else None)
    monkeypatch.setattr(season_backfill, "parse_match_page", lambda html, cup_id, s: page)
    conn = connect_seeded(tmp_db_path)
    result = run_backfill(conn, 19, match_ids=[1])
    assert _counts(conn)[0] == 0
    assert (result.pages, result.skipped, result.new_matches) == (1, 1, 0)
    conn.close()
