import sqlite3
import sys
from unittest.mock import patch

import pytest
import requests

from src.etl import match_crawler
from src.utils.db_config import get_connection
from tests.conftest import connect_seeded


def _response(status_code: int, text: str = "") -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = text.encode("utf-8")
    return response


def _match_html(
    *,
    match_date: str,
    game_label: str,
    home_team: str,
    away_team: str,
    home_sets: list[int],
    away_sets: list[int],
) -> str:
    def cells(scores: list[int]) -> str:
        padded = [str(s) for s in scores] + [""] * (5 - len(scores))
        return "".join(f"<td>{c}</td>" for c in padded) + f"<td>{sum(scores)}</td>"

    return (
        "<html><body>"
        f'<div class="game_header"><span>{match_date} 16:30:00</span>'
        f"<span>屏東體育館</span><span>例行賽 Week 4</span><span>{game_label}</span></div>"
        '<table class="match_table">'
        "<tr><th>隊伍</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>總分</th></tr>"
        f"<tr><td>{home_team}</td>{cells(home_sets)}</tr>"
        f"<tr><td>{away_team}</td>{cells(away_sets)}</tr>"
        "</table></body></html>"
    )


def _seed_unplayed(conn: sqlite3.Connection, game_id: int, gender: str, cup_id: int, match_date: str, home: str, away: str) -> None:
    conn.execute(
        """
        INSERT INTO matches (game_id, gender, cup_id, match_date, home_team, away_team)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (game_id, gender, cup_id, match_date, home, away),
    )


def _run_by_date(monkeypatch, tmp_db_path, target_date: str, pages: dict[str, requests.Response]) -> list[str]:
    requested: list[str] = []

    def fake_get(url, **_):
        path = url.removeprefix(match_crawler.BASE_URL)
        requested.append(path)
        return pages[path]

    monkeypatch.setattr(sys, "argv", ["match_crawler", "--date", target_date, "--delay", "0"])
    with (
        patch("src.etl.match_crawler.get_connection", lambda: get_connection(db_path=tmp_db_path)),
        patch("src.etl.match_crawler.requests.get", fake_get),
    ):
        match_crawler.main()
    return requested


def _scores(tmp_db_path, game_id: int, gender: str) -> tuple:
    conn = get_connection(db_path=tmp_db_path)
    try:
        return conn.execute(
            "SELECT home_set1, away_set1, home_total, away_total, home_sets_won, away_sets_won "
            "FROM matches WHERE game_id = ? AND gender = ?",
            (game_id, gender),
        ).fetchone()
    finally:
        conn.close()


def test_by_date_fetches_only_target_date_current_season_with_group_paths(monkeypatch, tmp_db_path):
    conn = connect_seeded(tmp_db_path)
    _seed_unplayed(conn, 341, "M", 22, "2026-11-15", "屏東台電男排", "雲林美津濃")
    _seed_unplayed(conn, 288, "F", 22, "2026-11-15", "新北中纖", "義力營造")
    _seed_unplayed(conn, 340, "M", 22, "2026-11-14", "屏東台電男排", "雲林美津濃")
    _seed_unplayed(conn, 250, "M", 21, "2026-11-15", "屏東台電男排", "雲林美津濃")
    conn.commit()
    conn.close()

    pages = {
        "/game/341": _response(
            200,
            _match_html(
                match_date="2026-11-15",
                game_label="Game 341",
                home_team="屏東台電男排",
                away_team="雲林美津濃",
                home_sets=[25, 25, 25],
                away_sets=[20, 18, 22],
            ),
        ),
        "/wgame/288": _response(
            200,
            _match_html(
                match_date="2026-11-15",
                game_label="Game 288",
                home_team="新北中纖",
                away_team="義力營造",
                home_sets=[25, 22, 25, 25],
                away_sets=[19, 25, 21, 20],
            ),
        ),
    }

    requested = _run_by_date(monkeypatch, tmp_db_path, "2026-11-15", pages)

    assert set(requested) == {"/game/341", "/wgame/288"}
    assert len(requested) == 2
    assert _scores(tmp_db_path, 341, "M") == (25, 20, 75, 60, 3, 0)
    assert _scores(tmp_db_path, 288, "F") == (25, 19, 97, 85, 3, 1)
    assert _scores(tmp_db_path, 340, "M") == (None, None, None, None, None, None)
    assert _scores(tmp_db_path, 250, "M") == (None, None, None, None, None, None)


def test_by_date_with_no_matches_sends_zero_requests(monkeypatch, tmp_db_path):
    conn = connect_seeded(tmp_db_path)
    _seed_unplayed(conn, 341, "M", 22, "2026-11-15", "屏東台電男排", "雲林美津濃")
    _seed_unplayed(conn, 301, "M", 21, "2025-12-01", "屏東台電男排", "雲林美津濃")
    conn.commit()
    conn.close()

    assert _run_by_date(monkeypatch, tmp_db_path, "2026-11-19", {}) == []
    assert _run_by_date(monkeypatch, tmp_db_path, "2025-12-01", {}) == []


def test_by_date_blocked_by_403_fails_and_leaves_db_unchanged(monkeypatch, tmp_db_path):
    conn = connect_seeded(tmp_db_path)
    _seed_unplayed(conn, 341, "M", 22, "2026-11-15", "屏東台電男排", "雲林美津濃")
    _seed_unplayed(conn, 288, "F", 22, "2026-11-15", "新北中纖", "義力營造")
    conn.commit()
    conn.close()

    pages = {
        "/game/341": _response(
            200,
            _match_html(
                match_date="2026-11-15",
                game_label="Game 341",
                home_team="屏東台電男排",
                away_team="雲林美津濃",
                home_sets=[25, 25, 25],
                away_sets=[20, 18, 22],
            ),
        ),
        "/wgame/288": _response(403, "<html>Attention Required! | Cloudflare</html>"),
    }

    with pytest.raises(requests.HTTPError):
        _run_by_date(monkeypatch, tmp_db_path, "2026-11-15", pages)

    assert _scores(tmp_db_path, 341, "M") == (None, None, None, None, None, None)
    assert _scores(tmp_db_path, 288, "F") == (None, None, None, None, None, None)


@pytest.mark.parametrize("bad_date", ["2026/11/15", "2026-13-01", "not-a-date"])
def test_by_date_rejects_invalid_date_format(monkeypatch, tmp_db_path, bad_date):
    connect_seeded(tmp_db_path).close()

    with pytest.raises(SystemExit) as exc_info:
        _run_by_date(monkeypatch, tmp_db_path, bad_date, {})

    assert exc_info.value.code != 0


