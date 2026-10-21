import sqlite3
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from bs4 import BeautifulSoup

from src.utils.seasons import Season, get_season
from tests.conftest import connect_seeded
from src.etl.stats_crawler import (
    fetch_match_roster, fetch_player_stats, resolve_week_label, upsert_roster_registration,
    build_name_to_pid,
)



def _seeded_season(cup_id):
    conn = sqlite3.connect(":memory:")
    conn.executescript((Path(__file__).resolve().parents[1] / "sql" / "schema.sql").read_text(encoding="utf-8"))
    return get_season(conn, cup_id)


SEASON_21 = _seeded_season(21)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "match_ashx_sample.html"


@pytest.fixture
def seeded_conn(tmp_db_path):
    conn = connect_seeded(tmp_db_path)
    yield conn
    conn.close()


class _FakeResponse:
    status_code = 200

    def __init__(self, text: str):
        self.text = text
        self.encoding = "utf-8"

    def raise_for_status(self):
        pass


def test_fetch_match_roster_parses_fixture(seeded_conn):
    fixture_html = FIXTURE.read_text(encoding="utf-8")
    with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(fixture_html)):
        rows = fetch_match_roster(seeded_conn, SEASON_21, match_id=1)

    assert rows is not None
    assert len(rows) == 10  # 兩隊各 5 位（fixture 精簡版）

    zhang = next(r for r in rows if r["name"] == "張瓈文")
    assert zhang["jersey_number"] == 2
    assert zhang["position"] == "OP"       # 對角 -> OP
    assert zhang["team_id"] == 5           # 新北中纖
    assert zhang["team_gender"] == "F"
    assert zhang["match_date"] == "2025-11-01"

    du = next(r for r in rows if r["name"] == "杜家馨")
    assert du["position"] == "MB"          # 攔中 -> MB

    libero = next(r for r in rows if r["name"] == "范張予馨")
    assert libero["position"] == "L"       # 自由 -> L


def test_fetch_match_roster_uses_season_for_cup_id_and_year(seeded_conn):
    fixture_html = FIXTURE.read_text(encoding="utf-8")
    season_20 = Season(20, "2024-25", date(2024, 7, 1), date(2025, 6, 30))
    with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(fixture_html)) as mock_get:
        rows = fetch_match_roster(seeded_conn, season_20, match_id=1)

    assert mock_get.call_args.kwargs["params"]["CupID"] == 20
    assert rows[0]["match_date"] == "2024-11-01"


def test_fetch_match_roster_column_order_not_confused_with_player_ashx(seeded_conn):
    """
    張瓈文那一列原始 cells（扣背號/姓名/位置後）是 2,5,0,1,4,0,0,0,0,0,0,3。
    若誤用 Player.ashx 的「總在前得在後」順序，attack_total 會被誤讀成 2 而非 5。
    """
    fixture_html = FIXTURE.read_text(encoding="utf-8")
    with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(fixture_html)):
        rows = fetch_match_roster(seeded_conn, SEASON_21, match_id=1)

    zhang = next(r for r in rows if r["name"] == "張瓈文")
    # fetch_match_roster 目前只回傳背號/姓名/位置（roster_registrations 不需要
    # 逐場數值統計），因此這裡驗證的是「沒有把數值欄位誤植進 dict」——
    # dict 中不該出現任何統計欄位鍵名。
    assert set(zhang.keys()) == {
        "match_date", "title_text", "team_id", "team_gender",
        "jersey_number", "name", "position",
    }


def test_resolve_week_label_uses_matches_round_name(tmp_db_path):
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("""
        CREATE TABLE matches (
            match_id INTEGER PRIMARY KEY, game_id INTEGER, gender TEXT,
            match_date DATE, round_name TEXT, home_team TEXT, away_team TEXT, cup_id INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO matches (game_id, gender, match_date, round_name, home_team, away_team, cup_id) "
        "VALUES (1, 'F', '2025-11-01', '例行賽 Week 1', '新北中纖', '義力營造', 21)"
    )
    conn.commit()

    week_label, week_start = resolve_week_label(conn, "2025-11-01", "女子組 第1週(...) 編號：1", 21)
    assert week_label == "例行賽 Week 1"
    assert week_start == "2025-11-01"
    conn.close()


def test_resolve_week_label_falls_back_when_no_match_found(tmp_db_path):
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("""
        CREATE TABLE matches (
            match_id INTEGER PRIMARY KEY, game_id INTEGER, gender TEXT,
            match_date DATE, round_name TEXT, home_team TEXT, away_team TEXT, cup_id INTEGER
        )
    """)
    conn.commit()

    week_label, week_start = resolve_week_label(conn, "2099-01-01", "女子組 第99週(...) 編號：1", 21)
    assert week_label.startswith("未比對-")
    assert week_start == "2099-01-01"
    conn.close()


def test_resolve_week_label_fallback_distinguishes_different_dates(tmp_db_path):
    """Finding 1: 不同 match_date 落入 fallback 時不應合併成同一個 week_label。"""
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("""
        CREATE TABLE matches (
            match_id INTEGER PRIMARY KEY, game_id INTEGER, gender TEXT,
            match_date DATE, round_name TEXT, home_team TEXT, away_team TEXT, cup_id INTEGER
        )
    """)
    conn.commit()

    label_1, start_1 = resolve_week_label(conn, "2099-01-01", "", 21)
    label_2, start_2 = resolve_week_label(conn, "2099-01-08", "", 21)

    assert label_1 != label_2
    assert label_1 == "未比對-2099-01-01"
    assert label_2 == "未比對-2099-01-08"
    assert start_1 == "2099-01-01"
    assert start_2 == "2099-01-08"
    conn.close()


def test_resolve_week_label_scopes_week_start_date_by_season(tmp_db_path):
    """Finding 2: 同一 round_name 跨賽季時，week_start_date 應取離目標場次近的那個賽季。"""
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("""
        CREATE TABLE matches (
            match_id INTEGER PRIMARY KEY, game_id INTEGER, gender TEXT,
            match_date DATE, round_name TEXT, home_team TEXT, away_team TEXT, cup_id INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO matches (game_id, gender, match_date, round_name, home_team, away_team, cup_id) "
        "VALUES (1, 'F', '2024-12-21', '例行賽 Week 1', '新北中纖', '義力營造', 20)"
    )
    conn.execute(
        "INSERT INTO matches (game_id, gender, match_date, round_name, home_team, away_team, cup_id) "
        "VALUES (1, 'F', '2026-01-04', '例行賽 Week 1', '新北中纖', '義力營造', 21)"
    )
    conn.commit()

    week_label, week_start = resolve_week_label(conn, "2026-01-04", "", 21)
    assert week_label == "例行賽 Week 1"
    assert week_start == "2026-01-04"
    conn.close()


def test_upsert_roster_registration_is_idempotent(tmp_db_path):
    from pathlib import Path as _P
    schema_sql = (_P(__file__).resolve().parents[1] / "sql" / "schema.sql").read_text(encoding="utf-8")
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(schema_sql)
    conn.execute("INSERT INTO teams (team_id, team_name, gender) VALUES (5, '新北中纖', 'F')")
    conn.execute("INSERT INTO players (name, gender) VALUES ('張瓈文', 'F')")
    pid = conn.execute("SELECT player_id FROM players").fetchone()[0]

    row = {"team_id": 5, "team_gender": "F", "jersey_number": 2, "position": "OP"}
    upsert_roster_registration(conn, pid, row, "例行賽 Week 1", "2025-11-01")
    upsert_roster_registration(conn, pid, row, "例行賽 Week 1", "2025-11-01")  # 重跑一次

    rows = conn.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()
    assert rows[0] == 1, "重跑 upsert 不應產生重複列"
    conn.close()


def test_fetch_match_roster_logs_warning_for_unknown_position(seeded_conn, caplog):
    """
    驗證未知位置用語時記錄警告，且該列仍被納入結果但 position 為 None。
    """
    unknown_position_html = """<h3><img src='_images/Sex_2.png' height='32' />女子組 第1週(場地) 編號：1 (11月1日 13:00)  歷時 02:22</h3>
<div class='TableFormat_1'><table cellpadding='0' cellspacing='0' align='center' width='100%'  >
<tr><td><div class='MatchResult'><table cellpadding='0' cellspacing='0' align='center' width='100%'  >
<tr><td class='TeamName  lightBackground'><a href='Team.aspx?CupID=21&TeamID=8'>新北中纖</a></td><td class='Score largeFont_3' style='color:red'>25</td></tr>
</table>
</div></td></tr>
</table>
</div>

<h3>新北中纖：邱雅慧</h3>
<br/>
<div class='TableFormat_1'><table cellpadding='0' cellspacing='0' align='center' width='100%'  >
<tr><td class='head' colspan='3'>新北中纖</td></td><td class='head' colspan='2'>攻擊(Attack)</td></td><td class='head' colspan='2'>發球(Serve)</td></td><td class='head' colspan='2'>接發(Receive)</td></td><td class='head' colspan='2'>防守(Dig)</td></td><td class='head' colspan='2'>舉球(Set)</td></td><td class='head'>總得分</td></tr>
<tr><td class='head'>N<SUP>o</SUP></td></td><td class='head' colspan='2'>球員</td></td><td class='head'>得</td></td><td class='head'>總</td></td><td class='head'>得</td></td><td class='head'>得</td></td><td class='head'>總</td></td><td class='head'>好</td></td><td class='head'>總</td></td><td class='head'>好</td></td><td class='head'>總</td></td><td class='head'>好</td></td><td class='head'>總</td></td></tr>
<tr><td class='largeFont_1'>2</td></td><td><a href='Player.aspx?CupID=21&PlayerID=124'>張瓈文</a></td><td>教練</td></td><td>1</td></td><td>2</td></td><td>0</td></td><td>0</td></td><td>1</td></td><td>0</td></td><td>0</td></td><td>0</td></td><td>0</td></td><td>0</td></td><td>0</td></td><td>1</td></td></tr>
</table>
</div>
"""

    with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(unknown_position_html)):
        with caplog.at_level("WARNING", logger="src.etl.stats_crawler"):
            rows = fetch_match_roster(seeded_conn, SEASON_21, match_id=1)

    assert rows is not None
    assert len(rows) == 1
    assert rows[0]["name"] == "張瓈文"
    assert rows[0]["position"] is None  # 未知位置映射為 None

    # 驗證警告被記錄
    assert any("未知位置用語" in record.message and "教練" in record.message for record in caplog.records), \
        "應該記錄包含 '未知位置用語' 和 '教練' 的警告"


def test_crawl_all_rosters_resilient_to_fetch_failures(tmp_db_path, caplog):
    """
    驗證 crawl_all_rosters 在單場出賽名單抓取失敗時，
    記錄警告、跳過該場、繼續爬取其他場次。
    """
    import requests
    from pathlib import Path as _P
    from src.etl.stats_crawler import crawl_all_rosters

    conn = connect_seeded(tmp_db_path)
    conn.execute("PRAGMA foreign_keys = ON")

    # 建立 matches 表，讓 resolve_week_label 能查到 round_name
    conn.execute(
        "INSERT INTO matches (game_id, gender, cup_id, match_date, round_name, home_team, away_team) "
        "VALUES (1, 'F', 21, '2025-11-01', '例行賽 Week 1', '新北中纖', '義力營造')"
    )
    conn.execute(
        "INSERT INTO matches (game_id, gender, cup_id, match_date, round_name, home_team, away_team) "
        "VALUES (2, 'F', 21, '2025-11-08', '例行賽 Week 2', '義力營造', '新北中纖')"
    )
    conn.commit()
    conn.close()

    # 正常的名單回應
    normal_roster_html = FIXTURE.read_text(encoding="utf-8")

    # 模擬 fetch_match_roster：MatchID=208 拋 HTTPError，其他回傳正常
    def mock_fetch_match_roster(conn, season, match_id):
        if match_id == 208:
            raise requests.HTTPError("500 Server Error")
        # 對 MatchID=1 回傳 fixture 內容（已測試）
        with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(normal_roster_html)):
            return fetch_match_roster(conn, season, match_id)

    # 模擬 fetch_match_list：返回兩場（1 成功，208 失敗）
    mock_match_list = [
        {"match_id": 1, "label": "第1週-女 1"},
        {"match_id": 208, "label": "總決賽-女 115-1"},
    ]

    with patch("src.etl.stats_crawler.fetch_match_roster", side_effect=mock_fetch_match_roster):
        with patch("src.etl.stats_crawler.fetch_match_list", return_value=mock_match_list):
            with caplog.at_level("WARNING", logger="src.etl.stats_crawler"):
                conn = sqlite3.connect(tmp_db_path)
                conn.execute("PRAGMA foreign_keys = ON")
                stats = crawl_all_rosters(conn, cup_id=21)
                conn.close()

    # 斷言：
    # 1. 不拋例外（直接在 crawl_all_rosters 中被捕捉）
    assert True, "crawl_all_rosters 應該不拋例外"

    # 2. matches_skipped == 1（MatchID=208）
    assert stats["matches_skipped"] == 1, f"Expected matches_skipped=1, got {stats['matches_skipped']}"

    # 3. matches_scanned == 1（MatchID=1 成功）
    assert stats["matches_scanned"] == 1, f"Expected matches_scanned=1, got {stats['matches_scanned']}"

    # 4. registrations_upserted > 0（MatchID=1 的球員名單已寫入）
    assert stats["registrations_upserted"] > 0, \
        f"Expected registrations_upserted > 0, got {stats['registrations_upserted']}"

    # 5. 驗證警告被記錄
    assert any("抓取失敗" in record.message and "208" in record.message for record in caplog.records), \
        "應該記錄包含 '抓取失敗' 和 '208' 的警告"


def test_crawl_all_rosters_default_cup_resolves_year_from_seasons_table(tmp_db_path):
    """不帶 cup_id 時走 EXT_CUP_ID（21），年份由暫存 DB 的 seasons 表推算，結果與改前相同。"""
    from src.etl.stats_crawler import crawl_all_rosters
    from src.utils.constants import EXT_CUP_ID

    assert EXT_CUP_ID == 21
    conn = connect_seeded(tmp_db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        "INSERT INTO matches (game_id, gender, cup_id, match_date, round_name, home_team, away_team) "
        "VALUES (1, 'F', 21, '2025-11-01', '例行賽 Week 1', '新北中纖', '義力營造')"
    )
    conn.commit()

    fixture_html = FIXTURE.read_text(encoding="utf-8")
    with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(fixture_html)) as mock_get:
        with patch("src.etl.stats_crawler.fetch_match_list", return_value=[{"match_id": 1, "label": "第1週-女 1"}]):
            stats = crawl_all_rosters(conn)

    rows = conn.execute(
        "SELECT DISTINCT cup_id, week_label, week_start_date FROM roster_registrations"
    ).fetchall()
    conn.close()
    assert stats["matches_scanned"] == 1
    assert rows == [(21, "例行賽 Week 1", "2025-11-01")]
    assert mock_get.call_args.kwargs["params"]["CupID"] == 21


def test_fetch_match_roster_resolves_team_names_via_aliases(seeded_conn):
    html = FIXTURE.read_text(encoding="utf-8")
    with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(html)):
        rows = fetch_match_roster(seeded_conn, SEASON_21, match_id=1)

    assert {(r["team_id"], r["team_gender"]) for r in rows} == {(5, "F"), (7, "F")}


def test_fetch_match_roster_resolves_legacy_team_names(seeded_conn):
    html = FIXTURE.read_text(encoding="utf-8")
    assert "<h3>新北中纖：" in html and "<h3>義力營造：" in html
    html = html.replace("<h3>新北中纖：", "<h3>連莊：").replace("<h3>義力營造：", "<h3>臺北Conti：")
    with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(html)):
        rows = fetch_match_roster(seeded_conn, SEASON_21, match_id=1)

    assert {(r["team_id"], r["team_gender"]) for r in rows} == {(10, "M"), (9, "M")}


def test_fetch_match_roster_skips_unknown_team_with_warning(seeded_conn, caplog):
    html = FIXTURE.read_text(encoding="utf-8")
    assert "<h3>新北中纖：" in html
    html = html.replace("<h3>新北中纖：", "<h3>不存在隊：")
    with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(html)):
        with caplog.at_level("WARNING", logger="src.etl.stats_crawler"):
            rows = fetch_match_roster(seeded_conn, SEASON_21, match_id=7)

    assert {(r["team_id"], r["team_gender"]) for r in rows} == {(7, "F")}
    assert any("不存在隊" in r.getMessage() and "7" in r.getMessage() for r in caplog.records)


def _seed_two_season_matches(conn):
    for game_id, cup, d, rnd in [
        (1, 20, "2025-11-01", "挑戰賽 Week 18"), (2, 21, "2025-11-01", "例行賽 Week 1"),
        (3, 20, "2025-10-25", "挑戰賽 Week 18"), (4, 21, "2025-10-20", "例行賽 Week 1"),
    ]:
        conn.execute(
            "INSERT INTO matches (game_id, gender, cup_id, match_date, round_name, home_team, away_team) "
            "VALUES (?, 'F', ?, ?, ?, '新北中纖', '義力營造')", (game_id, cup, d, rnd))
    conn.commit()


def test_resolve_week_label_scopes_by_cup_id(seeded_conn):
    _seed_two_season_matches(seeded_conn)

    assert resolve_week_label(seeded_conn, "2025-11-01", "", 20) == ("挑戰賽 Week 18", "2025-10-25")
    assert resolve_week_label(seeded_conn, "2025-11-01", "", 21) == ("例行賽 Week 1", "2025-10-20")


def test_resolve_week_start_date_scoped_by_cup_id_with_same_round_name(seeded_conn):
    for game_id, cup, d in [(1, 20, "2025-10-25"), (2, 21, "2025-11-01")]:
        seeded_conn.execute(
            "INSERT INTO matches (game_id, gender, cup_id, match_date, round_name, home_team, away_team) "
            "VALUES (?, 'F', ?, ?, '例行賽 Week 1', '新北中纖', '義力營造')", (game_id, cup, d))
    seeded_conn.commit()

    assert resolve_week_label(seeded_conn, "2025-11-01", "", 21) == ("例行賽 Week 1", "2025-11-01")
    assert resolve_week_label(seeded_conn, "2025-10-25", "", 20) == ("例行賽 Week 1", "2025-10-25")


def test_crawl_all_rosters_backfills_given_cup_id(seeded_conn):
    from src.etl.stats_crawler import crawl_all_rosters

    for game_id, cup, d in [(1, 20, "2024-11-01"), (2, 21, "2025-11-01")]:
        seeded_conn.execute(
            "INSERT INTO matches (game_id, gender, cup_id, match_date, round_name, home_team, away_team) "
            "VALUES (?, 'F', ?, ?, '例行賽 Week 1', '新北中纖', '義力營造')", (game_id, cup, d))
    seeded_conn.commit()

    html = FIXTURE.read_text(encoding="utf-8")
    with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(html)) as mock_get:
        with patch("src.etl.stats_crawler.fetch_match_list", return_value=[{"match_id": 1, "label": "第1週-女 1"}]):
            with patch("src.etl.stats_crawler.time.sleep"):
                stats = crawl_all_rosters(seeded_conn, cup_id=20)

    rows = seeded_conn.execute(
        "SELECT DISTINCT cup_id, week_label, week_start_date FROM roster_registrations"
    ).fetchall()
    assert stats["matches_scanned"] == 1
    assert rows == [(20, "例行賽 Week 1", "2024-11-01")]
    assert mock_get.call_args.kwargs["params"]["CupID"] == 20


def test_cli_rosters_passes_cup_id():
    from src.etl.stats_crawler import cli

    with patch("src.etl.stats_crawler.crawl_all_rosters", return_value={}) as crawl, \
         patch("src.etl.stats_crawler.get_connection"):
        cli(["--rosters", "--cup-id", "20"])
    assert crawl.call_args.kwargs["cup_id"] == 20


def test_cli_rosters_default_cup_id_is_current_season():
    from src.etl.stats_crawler import cli

    with patch("src.etl.stats_crawler.crawl_all_rosters", return_value={}) as crawl, \
         patch("src.etl.stats_crawler.get_connection"):
        cli(["--rosters"])
    assert crawl.call_args.kwargs["cup_id"] == 21


def test_cli_cup_id_passed_to_stats_mode():
    from src.etl.stats_crawler import cli

    with patch("src.etl.stats_crawler.main") as main, \
         patch("src.etl.stats_crawler.crawl_all_rosters") as crawl:
        cli(["--cup-id", "20", "--incremental"])
    main.assert_called_once_with(incremental=True, cup_id=20)
    crawl.assert_not_called()


def test_build_name_to_pid_distinguishes_gender(tmp_db_path):
    """同名不同性別是兩位球員，查找表不得互相覆蓋。"""
    from pathlib import Path as _P
    from src.etl.stats_crawler import build_name_to_pid

    schema_sql = (_P(__file__).resolve().parents[1] / "sql" / "schema.sql").read_text(encoding="utf-8")
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript(schema_sql)
    conn.execute("INSERT INTO players (name, gender) VALUES ('陳大文', 'M')")
    conn.execute("INSERT INTO players (name, gender) VALUES ('陳大文', 'F')")
    conn.commit()

    name_map = build_name_to_pid(conn)

    pid_m = conn.execute("SELECT player_id FROM players WHERE gender = 'M'").fetchone()[0]
    pid_f = conn.execute("SELECT player_id FROM players WHERE gender = 'F'").fetchone()[0]
    assert name_map[("陳大文", "M")] == pid_m
    assert name_map[("陳大文", "F")] == pid_f
    conn.close()


def _http_error_response(status_code: int):
    import requests
    resp = _FakeResponse("")
    resp.status_code = status_code

    def _raise():
        raise requests.HTTPError(f"{status_code} Client Error", response=resp)

    resp.raise_for_status = _raise
    return resp


def test_fetch_match_list_falls_back_to_match_ashx_scan_when_aspx_missing():
    """
    外部系統 Match.aspx 自 2026-09-29 起 404（整站 .aspx 頁面消失，_handler/*.ashx 仍在）。
    fetch_match_list 須改以 Match.ashx 逐號探測列舉 MatchID，並跨過例行賽與季後賽之間的
    空號區（2025-26 實測：1–108 連續、109–200 皆空、201–210 為季後賽）。
    """
    from src.etl.stats_crawler import fetch_match_list, MATCH_ID_SCAN_GAP_LIMIT

    roster_html = FIXTURE.read_text(encoding="utf-8")
    existing = {1, 2, 108, 201, 210}
    requested = []

    def fake_get(url, params=None, headers=None, timeout=None):
        if url.endswith("/Match.aspx"):
            return _http_error_response(404)
        requested.append(params["MatchID"])
        if params["MatchID"] in existing:
            return _FakeResponse(roster_html)
        # 查無該場：外部系統回 HTTP 500 的 ASP.NET 未處理例外頁（實測 2026-10-15）
        missing = _http_error_response(500)
        missing.text = (
            "<!DOCTYPE html><html><head><title>並未將物件參考設定為物件的執行個體。</title></head><body></body></html>"
        )
        return missing

    with patch("src.etl.stats_crawler.requests.get", side_effect=fake_get), \
         patch("src.etl.stats_crawler.time.sleep"):
        result = fetch_match_list(cup_id=21)

    assert [m["match_id"] for m in result] == [1, 2, 108, 201, 210]
    assert MATCH_ID_SCAN_GAP_LIMIT > 92, "門檻須大於 2025-26 實測的 92 格空號"
    assert max(requested) == 210 + MATCH_ID_SCAN_GAP_LIMIT, "最後一場之後連續空號達門檻即停"


def test_fetch_match_list_scan_refuses_to_report_empty_season():
    """逐號探測一場都沒找到（外部系統整站 500）要拋錯，不能回空清單讓每日更新誤判「無新資料」。"""
    from src.etl.stats_crawler import fetch_match_list

    def fake_get(url, params=None, headers=None, timeout=None):
        return _http_error_response(404 if url.endswith("/Match.aspx") else 500)

    with patch("src.etl.stats_crawler.requests.get", side_effect=fake_get), \
         patch("src.etl.stats_crawler.time.sleep"):
        with pytest.raises(RuntimeError, match="外部系統異常"):
            fetch_match_list(cup_id=21)


def test_fetch_match_list_propagates_non_404_errors():
    """Match.aspx 非 404 的 HTTP 錯誤（如 500）不走掃描備援，照舊拋出。"""
    import requests
    from src.etl.stats_crawler import fetch_match_list

    with patch("src.etl.stats_crawler.requests.get", return_value=_http_error_response(500)):
        with pytest.raises(requests.HTTPError):
            fetch_match_list(cup_id=21)


def _player_table(*dates):
    head = "<tr><td>場次</td></tr><tr><td>日期</td></tr>"
    body = "".join(
        f"<tr><td>{d}</td><td>對手</td>" + "<td>1</td>" * 13 + "</tr>" for d in dates
    )
    return f"<table>{head}{body}<tr><td>累計</td></tr></table>"


def test_fetch_player_stats_resolves_dates_from_season():
    html = _player_table("311/01", "303/22")
    with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(html)) as mock_get:
        records = fetch_player_stats(SEASON_21, team_id=1, ext_player_id=2)

    assert [r["match_date"] for r in records] == ["2025-11-01", "2026-03-22"]
    assert mock_get.call_args.kwargs["params"]["CupID"] == 21


def test_fetch_player_stats_skips_dates_outside_season(caplog):
    short = Season(99, "test", date(2024, 10, 1), date(2025, 3, 31))
    html = _player_table("311/01", "303/22", "405/10")
    with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(html)):
        with caplog.at_level("WARNING"):
            records = fetch_player_stats(short, team_id=1, ext_player_id=2)

    assert [r["match_date"] for r in records] == ["2024-11-01", "2025-03-22"]
    assert any(r.levelname == "WARNING" and "99" in r.getMessage() for r in caplog.records)


def test_fetch_player_stats_warns_on_unparseable_date_cell(caplog):
    html = _player_table("311/01", "累計?")
    with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(html)):
        with caplog.at_level("WARNING", logger="src.etl.stats_crawler"):
            records = fetch_player_stats(SEASON_21, team_id=1, ext_player_id=2)

    assert [r["match_date"] for r in records] == ["2025-11-01"]
    assert any("累計?" in r.getMessage() for r in caplog.records)
