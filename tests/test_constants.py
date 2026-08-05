import importlib

import src.utils.constants as constants


def test_season_year_for_month_default_season():
    assert constants.season_year_for_month(11) == 2025
    assert constants.season_year_for_month(12) == 2025
    assert constants.season_year_for_month(1) == 2026
    assert constants.season_year_for_month(6) == 2026


def test_season_year_for_month_custom_season():
    assert constants.season_year_for_month(11, season="2026-27") == 2026
    assert constants.season_year_for_month(3, season="2026-27") == 2027


def test_team_alias_merged_into_constants():
    assert constants.TEAM_ALIAS["桃園臺灣產險"] == "桃園臺產"
    assert constants.TEAM_ALIAS["臺中獅子王"] == "獅子王"
    assert constants.TEAM_ALIAS["臺北鯨華"] == "臺北鯨華"


def test_ext_base_env_override(monkeypatch):
    monkeypatch.setenv("EXT_BASE", "http://example.com")
    importlib.reload(constants)
    assert constants.EXT_BASE == "http://example.com"
    monkeypatch.delenv("EXT_BASE", raising=False)
    importlib.reload(constants)
    assert constants.EXT_BASE == "http://114.35.229.141"


def test_season_env_override(monkeypatch):
    monkeypatch.setenv("SEASON", "2026-27")
    importlib.reload(constants)
    assert constants.SEASON == "2026-27"
    monkeypatch.delenv("SEASON", raising=False)
    importlib.reload(constants)
    assert constants.SEASON == "2025-26"
