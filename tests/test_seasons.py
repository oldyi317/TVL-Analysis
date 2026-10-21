from datetime import date

import pytest

from src.utils.seasons import Season, get_season, resolve_date_in_season
from tests.conftest import connect_seeded


def test_resolve_date_cup20(tmp_db_path):
    season = get_season(connect_seeded(tmp_db_path), 20)
    assert resolve_date_in_season(10, 19, season) == "2024-10-19"
    assert resolve_date_in_season(3, 22, season) == "2025-03-22"


def test_resolve_date_cup21(tmp_db_path):
    season = get_season(connect_seeded(tmp_db_path), 21)
    assert resolve_date_in_season(11, 1, season) == "2025-11-01"
    assert resolve_date_in_season(3, 22, season) == "2026-03-22"


def test_resolve_date_outside_season_returns_none_and_warns(caplog):
    season = Season(99, "test", date(2024, 10, 1), date(2025, 3, 31))
    with caplog.at_level("WARNING"):
        assert resolve_date_in_season(5, 10, season) is None
    assert any(r.levelname == "WARNING" and "99" in r.getMessage() for r in caplog.records)


def test_get_season_unknown_cup_raises(tmp_db_path):
    with pytest.raises(RuntimeError):
        get_season(connect_seeded(tmp_db_path), 999)


def test_resolve_feb29_in_non_leap_season_returns_none(tmp_db_path, caplog):
    season = get_season(connect_seeded(tmp_db_path), 21)
    with caplog.at_level("WARNING"):
        assert resolve_date_in_season(2, 29, season) is None
    assert any(r.levelname == "WARNING" for r in caplog.records)
