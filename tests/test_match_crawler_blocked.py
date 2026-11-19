import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from src.etl import match_crawler
from src.utils.db_config import get_connection
from tests.conftest import connect_seeded

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _response(status_code: int, text: str = "") -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = text.encode("utf-8")
    return response


def _rescan(monkeypatch, tmp_db_path, pages: dict[str, requests.Response]) -> None:
    """跑一次只掃 game_id 301 的全季重掃；pages 以官網路徑對應回應。"""
    connect_seeded(tmp_db_path).close()
    monkeypatch.setattr(sys, "argv", ["match_crawler", "--range-start", "301", "--range-end", "302", "--delay", "0"])
    with (
        patch("src.etl.match_crawler.get_connection", lambda: get_connection(db_path=tmp_db_path)),
        patch("src.etl.match_crawler.requests.get", lambda url, **_: pages[url.removeprefix(match_crawler.BASE_URL)]),
    ):
        match_crawler.main()


def _stored_matches(tmp_db_path) -> int:
    conn = sqlite3.connect(tmp_db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    finally:
        conn.close()


@pytest.fixture
def played_page():
    return _response(200, (FIXTURES / "match_game_sample.html").read_text(encoding="utf-8"))


def test_rescan_stores_scores_from_pages_that_respond_200(monkeypatch, tmp_db_path, played_page):
    _rescan(monkeypatch, tmp_db_path, {"/game/301": played_page, "/wgame/301": _response(200, "<html></html>")})

    assert _stored_matches(tmp_db_path) == 1


def test_rescan_reports_pages_requested(monkeypatch, tmp_db_path, capsys, played_page):
    _rescan(monkeypatch, tmp_db_path, {"/game/301": played_page, "/wgame/301": _response(200, "<html></html>")})

    assert "本次請求：2 頁" in capsys.readouterr().out


def test_rescan_blocked_by_403_fails_and_stores_nothing(monkeypatch, tmp_db_path, played_page):
    blocked = _response(403, "<html>Attention Required! | Cloudflare</html>")

    with pytest.raises(requests.HTTPError):
        _rescan(monkeypatch, tmp_db_path, {"/game/301": played_page, "/wgame/301": blocked})

    assert _stored_matches(tmp_db_path) == 0
