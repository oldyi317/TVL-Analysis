import logging
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from src.etl.schedule_crawler import (
    next_year_path,
    parse_schedule_page,
    scrape_schedule_pages,
    upsert_schedule_match,
)
from src.utils.constants import TVL_BASE_URL
from tests.conftest import connect_seeded

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MEN = "schedule_2026-10-15.html"
WOMEN = "wschedule_2026-10-15.html"
MEN_NEXT = "schedule_115_2026-10-15.html"
WOMEN_NEXT = "wschedule_115_2026-10-15.html"
PAGES = {"/schedule": MEN, "/wschedule": WOMEN, "/schedule/115": MEN_NEXT, "/wschedule/115": WOMEN_NEXT}
SCORE_COLUMNS = [
    "home_set1", "home_set2", "home_set3", "home_set4", "home_set5", "home_total",
    "away_set1", "away_set2", "away_set3", "away_set4", "away_set5", "away_total",
    "home_sets_won", "away_sets_won",
]


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text
        self.encoding = "utf-8"

    def raise_for_status(self):
        pass


def _html(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def _count_match_rows(html):
    return len(re.findall(r'class="[^"]*\bmatch_row\b', html))


def _men_rows():
    rows, _ = parse_schedule_page(_html(MEN), "M")
    return {r["game_id"]: r for r in rows}


@pytest.fixture
def conn(tmp_db_path):
    connection = connect_seeded(tmp_db_path)
    yield connection
    connection.close()


def _fake_get(url, **kwargs):
    path = url.replace(TVL_BASE_URL, "")
    return _FakeResponse(_html(PAGES[path]))


# ── 解析 ────────────────────────────────────────────────────


@pytest.mark.parametrize("fixture, gender, expected", [(MEN, "M", 65), (WOMEN, "F", 53)])
def test_parses_every_match_row_on_page(fixture, gender, expected):
    html = _html(fixture)
    rows, on_page = parse_schedule_page(html, gender)
    assert on_page == _count_match_rows(html) == expected
    assert len(rows) == on_page
    assert len({r["game_id"] for r in rows}) == on_page
    assert all(r["gender"] == gender for r in rows)


def test_first_men_row_fields():
    row = _men_rows()[264]
    assert row["match_date"] == "2025-11-01"
    assert row["start_time"] == "18:30"
    assert row["venue"] == "國立成功大學體育館"
    assert row["home_team"] == "雲林美津濃"
    assert row["away_team"] == "屏東台電"
    assert row["game_label"] == "Game 3"
    assert row["is_golden_set"] == 0
    assert row["round_name"] is None
    assert not any(col in row for col in SCORE_COLUMNS)


def test_round_name_parsed_when_page_shows_it():
    html = _html(MEN).replace('<h5 class="fs12 mt-2 mb-1"></h5>', '<h5 class="fs12 mt-2 mb-1">例行賽 Week 1</h5>', 1)
    rows, _ = parse_schedule_page(html, "M")
    assert rows[0]["round_name"] == "例行賽 Week 1"
    assert rows[1]["round_name"] is None


def test_team_names_kept_as_official_labels():
    row = _men_rows()[265]
    assert (row["home_team"], row["away_team"]) == ("桃園臺灣產險", "獅子王")


def test_golden_set_row_from_label():
    row = _men_rows()[328]
    assert row["game_label"] == "Game 116-1 (黃金決勝局)"
    assert row["is_golden_set"] == 1
    assert row["match_date"] == "2026-03-22"


def test_women_page_uses_wgame_links():
    rows, _ = parse_schedule_page(_html(WOMEN), "F")
    first = rows[0]
    assert first["game_id"] == 228
    assert (first["home_team"], first["away_team"]) == ("新北中纖", "義力營造")
    assert first["start_time"] == "13:00"


def test_blank_or_non_numeric_score_cells_are_not_errors():
    html = _html(MEN)
    html = re.sub(r'(fs26">)\d+(</h6>)', r"\1\2", html)
    html = re.sub(r'(fs22[^"]*">)\d+(</h6>)', r"\1—\2", html)
    rows, on_page = parse_schedule_page(html, "M")
    assert len(rows) == on_page == 65
    assert rows[0]["venue"] == "國立成功大學體育館"


@pytest.mark.parametrize("fixture, gender, expected", [(MEN_NEXT, "M", 8), (WOMEN_NEXT, "F", 4)])
def test_unplayed_rows_on_next_year_page(fixture, gender, expected):
    html = _html(fixture)
    rows, on_page = parse_schedule_page(html, gender)
    assert len(rows) == on_page == _count_match_rows(html) == expected
    first = rows[0]
    assert first["match_date"] == "2026-10-24"
    assert first["venue"] == "高雄市鳳山體育館"
    assert first["start_time"] in ("13:00", "18:30")
    assert first["is_golden_set"] == 0
    assert not any(col in first for col in SCORE_COLUMNS)


def test_next_year_link_only_on_default_page():
    assert next_year_path(_html(MEN)) == "/schedule/115"
    assert next_year_path(_html(WOMEN)) == "/wschedule/115"
    assert next_year_path(_html(MEN_NEXT)) is None
    assert next_year_path(_html(WOMEN_NEXT)) is None


def test_row_without_date_is_skipped_with_warning(caplog):
    html = _html(MEN).replace('<h5 class="fs16 mt-2 mb-1">11/01</h5>', '<h5 class="fs16 mt-2 mb-1"></h5>', 1)
    with caplog.at_level(logging.WARNING):
        rows, on_page = parse_schedule_page(html, "M")
    assert on_page == 65 and len(rows) == 64
    assert "264" in caplog.text


# ── 寫入 ────────────────────────────────────────────────────


def test_upsert_inserts_unplayed_row_with_null_scores(conn):
    row = _men_rows()[264]
    assert upsert_schedule_match(conn, row, cup_id=21) is True
    db = conn.execute(
        f"SELECT cup_id, match_date, start_time, venue, round_name, game_label, home_team, away_team, "
        f"home_team_id, away_team_id, {', '.join(SCORE_COLUMNS)} FROM matches WHERE game_id=264 AND gender='M'"
    ).fetchone()
    assert db[:10] == (21, "2025-11-01", "18:30", "國立成功大學體育館", None, "Game 3", "雲林美津濃", "屏東台電", 2, 1)
    assert db[10:] == (None,) * len(SCORE_COLUMNS)


def test_upsert_never_touches_scores_or_round_name(conn):
    conn.execute(
        """INSERT INTO matches (game_id, gender, cup_id, match_date, start_time, venue, round_name, game_label,
           home_team, away_team, home_set1, away_set1, home_total, away_total, home_sets_won, away_sets_won)
           VALUES (264, 'M', 21, '2025-10-31', '10:00', '舊場館', '例行賽 Week 1', 'Game 3',
                   '雲林美津濃', '屏東台電', 25, 20, 75, 60, 3, 0)"""
    )
    assert upsert_schedule_match(conn, _men_rows()[264], cup_id=21) is False
    db = conn.execute(
        "SELECT match_date, start_time, venue, round_name, home_set1, away_set1, home_total, away_total, "
        "home_sets_won, away_sets_won, home_team_id FROM matches WHERE game_id=264 AND gender='M'"
    ).fetchone()
    assert db == ("2025-11-01", "18:30", "國立成功大學體育館", "例行賽 Week 1", 25, 20, 75, 60, 3, 0, 2)
    assert conn.execute("SELECT count(*) FROM matches").fetchone()[0] == 1


def test_upsert_keeps_existing_values_when_page_lacks_them(conn):
    row = dict(_men_rows()[328], start_time=None, venue=None, game_label=None, is_golden_set=0)
    upsert_schedule_match(conn, row, cup_id=21)
    conn.execute(
        "UPDATE matches SET start_time='18:30', venue='既有場館', round_name='總決賽 Week 19', "
        "game_label='Game 116-1 (黃金決勝局)', is_golden_set=1 WHERE game_id=328 AND gender='M'"
    )
    upsert_schedule_match(conn, row, cup_id=21)
    db = conn.execute(
        "SELECT start_time, venue, round_name, game_label, is_golden_set FROM matches WHERE game_id=328"
    ).fetchone()
    assert db == ("18:30", "既有場館", "總決賽 Week 19", "Game 116-1 (黃金決勝局)", 1)


def test_upsert_refreshes_golden_flag_with_label(conn):
    row = _men_rows()[328]
    upsert_schedule_match(conn, dict(row, game_label="Game 116", is_golden_set=0), cup_id=21)
    upsert_schedule_match(conn, row, cup_id=21)
    assert conn.execute("SELECT game_label, is_golden_set FROM matches WHERE game_id=328").fetchone() == (
        "Game 116-1 (黃金決勝局)", 1,
    )


def test_unknown_team_writes_null_and_warns(conn, caplog):
    row = dict(_men_rows()[264], home_team="不存在隊")
    with caplog.at_level(logging.WARNING):
        upsert_schedule_match(conn, row, cup_id=21)
    home_id, away_id = conn.execute("SELECT home_team_id, away_team_id FROM matches WHERE game_id=264").fetchone()
    assert (home_id, away_id) == (None, 1)
    assert "不存在隊" in caplog.text and "264" in caplog.text


# ── 整頁流程 ─────────────────────────────────────────────────


def test_scrape_pages_end_to_end(conn, caplog):
    with patch("src.etl.schedule_crawler.requests.get", side_effect=_fake_get), caplog.at_level(logging.INFO):
        result = scrape_schedule_pages(conn)
    assert result["M"] == {"parsed": 73, "on_page": 73, "inserted": 73, "updated": 0}
    assert result["F"] == {"parsed": 57, "on_page": 57, "inserted": 57, "updated": 0}
    assert "/schedule 解析出 65 場／頁面上 65 場" in caplog.text
    assert "/schedule/115 解析出 8 場／頁面上 8 場" in caplog.text
    assert conn.execute("SELECT count(*) FROM matches WHERE home_sets_won IS NULL").fetchone()[0] == 130
    assert conn.execute("SELECT count(*), min(game_id) FROM matches WHERE cup_id=22 AND gender='M'").fetchone() == (8, 329)
    new_team = conn.execute("SELECT home_team, home_team_id FROM matches WHERE game_id=329 AND gender='M'").fetchone()
    assert new_team == ("三福產業", 11)
    with patch("src.etl.schedule_crawler.requests.get", side_effect=_fake_get):
        again = scrape_schedule_pages(conn, genders=("M",))
    assert again["M"]["updated"] == 73 and again["M"]["inserted"] == 0


def test_unknown_season_skips_page_and_logs_error(conn, caplog):
    conn.execute("DELETE FROM seasons")
    with patch("src.etl.schedule_crawler.requests.get", side_effect=_fake_get), caplog.at_level(logging.ERROR):
        result = scrape_schedule_pages(conn, genders=("M",))
    assert result["M"] == {"parsed": 73, "on_page": 73, "inserted": 0, "updated": 0}
    assert conn.execute("SELECT count(*) FROM matches").fetchone()[0] == 0
    assert "賽季" in caplog.text


def test_request_failure_skips_page_without_raising(conn, caplog):
    with patch(
        "src.etl.schedule_crawler.requests.get", side_effect=requests.RequestException("boom")
    ), caplog.at_level(logging.ERROR):
        result = scrape_schedule_pages(conn, genders=("F",))
    assert result["F"] == {"parsed": 0, "on_page": 0, "inserted": 0, "updated": 0}
    assert "boom" in caplog.text


@pytest.mark.skip(reason="待賽程公布，見 #28")
def test_unplayed_row_with_undecided_time_or_venue():
    """2026-27 已有未開打列樣本（schedule_115 fixture，比分顯示 0/0）；
    時間或場館「待定」的列尚無樣本，拿到後補：缺值寫 NULL、不得讓整列被跳過。"""
