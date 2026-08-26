import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from bs4 import BeautifulSoup

from src.etl.stats_crawler import (
    fetch_match_roster, resolve_week_label, upsert_roster_registration,
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
