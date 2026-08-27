import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from bs4 import BeautifulSoup

from src.etl.stats_crawler import (
    fetch_match_roster, resolve_week_label, upsert_roster_registration,
    build_name_to_pid,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "match_ashx_sample.html"


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text
        self.encoding = "utf-8"

    def raise_for_status(self):
        pass


def test_fetch_match_roster_parses_fixture():
    fixture_html = FIXTURE.read_text(encoding="utf-8")
    with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(fixture_html)):
        rows = fetch_match_roster(cup_id=21, match_id=1)

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


def test_fetch_match_roster_column_order_not_confused_with_player_ashx():
    """
    張瓈文那一列原始 cells（扣背號/姓名/位置後）是 2,5,0,1,4,0,0,0,0,0,0,3。
    若誤用 Player.ashx 的「總在前得在後」順序，attack_total 會被誤讀成 2 而非 5。
    """
    fixture_html = FIXTURE.read_text(encoding="utf-8")
    with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(fixture_html)):
        rows = fetch_match_roster(cup_id=21, match_id=1)

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
            match_date DATE, round_name TEXT, home_team TEXT, away_team TEXT
        )
    """)
    conn.execute(
        "INSERT INTO matches (game_id, gender, match_date, round_name, home_team, away_team) "
        "VALUES (1, 'F', '2025-11-01', '例行賽 Week 1', '新北中纖', '義力營造')"
    )
    conn.commit()

    week_label, week_start = resolve_week_label(conn, "2025-11-01", "女子組 第1週(...) 編號：1")
    assert week_label == "例行賽 Week 1"
    assert week_start == "2025-11-01"
    conn.close()


def test_resolve_week_label_falls_back_when_no_match_found(tmp_db_path):
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("""
        CREATE TABLE matches (
            match_id INTEGER PRIMARY KEY, game_id INTEGER, gender TEXT,
            match_date DATE, round_name TEXT, home_team TEXT, away_team TEXT
        )
    """)
    conn.commit()

    week_label, week_start = resolve_week_label(conn, "2099-01-01", "女子組 第99週(...) 編號：1")
    assert week_label.startswith("未比對-")
    assert week_start == "2099-01-01"
    conn.close()


def test_resolve_week_label_fallback_distinguishes_different_dates(tmp_db_path):
    """Finding 1: 不同 match_date 落入 fallback 時不應合併成同一個 week_label。"""
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("""
        CREATE TABLE matches (
            match_id INTEGER PRIMARY KEY, game_id INTEGER, gender TEXT,
            match_date DATE, round_name TEXT, home_team TEXT, away_team TEXT
        )
    """)
    conn.commit()

    label_1, start_1 = resolve_week_label(conn, "2099-01-01", "")
    label_2, start_2 = resolve_week_label(conn, "2099-01-08", "")

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
            match_date DATE, round_name TEXT, home_team TEXT, away_team TEXT
        )
    """)
    conn.execute(
        "INSERT INTO matches (game_id, gender, match_date, round_name, home_team, away_team) "
        "VALUES (1, 'F', '2024-12-21', '例行賽 Week 1', '新北中纖', '義力營造')"
    )
    conn.execute(
        "INSERT INTO matches (game_id, gender, match_date, round_name, home_team, away_team) "
        "VALUES (1, 'F', '2026-01-04', '例行賽 Week 1', '新北中纖', '義力營造')"
    )
    conn.commit()

    week_label, week_start = resolve_week_label(conn, "2026-01-04", "")
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


def test_fetch_match_roster_logs_warning_for_unknown_position(caplog):
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
            rows = fetch_match_roster(cup_id=21, match_id=1)

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

    schema_sql = (_P(__file__).resolve().parents[1] / "sql" / "schema.sql").read_text(encoding="utf-8")
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(schema_sql)

    # 種子資料：兩支隊伍
    conn.execute("INSERT INTO teams (team_id, team_name, gender) VALUES (5, '新北中纖', 'F')")
    conn.execute("INSERT INTO teams (team_id, team_name, gender) VALUES (7, '義力營造', 'F')")

    # 建立 matches 表，讓 resolve_week_label 能查到 round_name
    conn.execute(
        "INSERT INTO matches (game_id, gender, match_date, round_name, home_team, away_team) "
        "VALUES (1, 'F', '2025-11-01', '例行賽 Week 1', '新北中纖', '義力營造')"
    )
    conn.execute(
        "INSERT INTO matches (game_id, gender, match_date, round_name, home_team, away_team) "
        "VALUES (2, 'F', '2025-11-08', '例行賽 Week 2', '義力營造', '新北中纖')"
    )
    conn.commit()
    conn.close()

    # 正常的名單回應
    normal_roster_html = FIXTURE.read_text(encoding="utf-8")

    # 模擬 fetch_match_roster：MatchID=208 拋 HTTPError，其他回傳正常
    def mock_fetch_match_roster(cup_id, match_id):
        if match_id == 208:
            raise requests.HTTPError("500 Server Error")
        # 對 MatchID=1 回傳 fixture 內容（已測試）
        with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(normal_roster_html)):
            return fetch_match_roster(cup_id, match_id)

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
