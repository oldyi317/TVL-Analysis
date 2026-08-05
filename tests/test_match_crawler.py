from sqlalchemy import text

from src.etl.match_crawler import normalize_team, upsert_match


def _sample_match(**overrides) -> dict:
    base = dict(
        game_id=301, gender="M", season="2025-26", match_date="2026-01-05",
        venue="台南", round_name="例行賽 Week 5", game_label="Game 301",
        is_golden_set=0, home_team="屏東台電", away_team="雲林美津濃",
        home_set1=25, home_set2=25, home_set3=25, home_set4=None, home_set5=None,
        home_total=75, away_set1=20, away_set2=18, away_set3=22,
        away_set4=None, away_set5=None, away_total=60,
        home_sets_won=3, away_sets_won=0,
    )
    base.update(overrides)
    return base


def test_normalize_team_maps_game_page_alias():
    assert normalize_team("桃園臺灣產險") == "桃園臺產"
    assert normalize_team("臺中獅子王") == "獅子王"
    assert normalize_team("屏東台電") == "屏東台電"


def test_upsert_match_is_idempotent_on_rerun(sqlite_engine):
    match = _sample_match()
    upsert_match(sqlite_engine, match)
    upsert_match(sqlite_engine, match)

    with sqlite_engine.begin() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM matches")).scalar_one()
    assert n == 1


def test_upsert_match_updates_corrected_score(sqlite_engine):
    match = _sample_match()
    upsert_match(sqlite_engine, match)

    corrected = _sample_match(home_total=74)
    upsert_match(sqlite_engine, corrected)

    with sqlite_engine.begin() as conn:
        total = conn.execute(text("SELECT home_total FROM matches")).scalar_one()
    assert total == 74


def test_upsert_match_does_not_touch_other_season_rows(sqlite_engine):
    match = _sample_match(season="2025-26")
    upsert_match(sqlite_engine, match)

    upsert_match(sqlite_engine, _sample_match(season="2026-27"))

    with sqlite_engine.begin() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM matches")).scalar_one()
        old_total = conn.execute(
            text("SELECT home_total FROM matches WHERE season = '2025-26'")
        ).scalar_one()
    assert total == 2
    assert old_total == 75
