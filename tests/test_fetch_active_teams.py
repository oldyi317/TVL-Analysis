import sqlite3
from pathlib import Path

import pytest

from src.app import helpers
from src.utils.teams import apply_team_seed

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


@pytest.fixture
def seeded_db(tmp_db_path, monkeypatch):
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    apply_team_seed(conn)
    conn.commit()
    conn.close()
    monkeypatch.setattr(helpers, "DB_PATH", tmp_db_path)
    helpers.load_data.clear()
    yield tmp_db_path
    helpers.load_data.clear()


def test_fetch_active_teams_excludes_inactive_legacy(seeded_db):
    df = helpers.fetch_active_teams("M")
    assert list(df["team_id"]) == [1, 2, 4, 5, 7]
    assert not {"彰化三大有線", "臺北Conti", "連莊"} & set(df["team_name"])


def test_fetch_active_teams_filters_gender(seeded_db):
    assert list(helpers.fetch_active_teams("F")["team_id"]) == [3, 4, 5, 7]
