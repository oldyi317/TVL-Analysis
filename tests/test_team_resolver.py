import re
import sqlite3
from pathlib import Path

import pytest

from src.utils.constants import EXT_TEAM_MAPS
from src.etl.stats_crawler import ext_team_map_for
from src.utils.teams import UnknownTeamError, apply_team_seed, resolve_ext_team, resolve_team

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    apply_team_seed(connection)
    yield connection
    connection.close()


OFFICIAL_SPELLINGS = [
    ("屏東台電", (1, "M")), ("屏東台電男排", (1, "M")), ("台灣電力公司男子排球隊", (1, "M")),
    ("高雄台電", (4, "F")), ("高雄台電女排", (4, "F")), ("台灣電力公司女子排球隊", (4, "F")),
    ("雲林美津濃", (2, "M")), ("美津濃男子排球隊", (2, "M")),
    ("臺北鯨華", (3, "F")), ("臺北鯨華女子排球隊", (3, "F")),
    ("新北中纖", (5, "F")), ("義力營造", (7, "F")),
    ("桃園臺產", (5, "M")), ("桃園臺灣產險", (5, "M")), ("桃園臺產隼鷹排球隊", (5, "M")),
    ("龍邁建設", (5, "M")), ("龍脈建設", (5, "M")),
    ("獅子王", (7, "M")), ("臺中獅子王", (7, "M")),
    ("臺北國北獅", (4, "M")), ("臺北鳳凰", (4, "M")),
    ("三福產業", (11, "M")), ("新竹可利工程師", (12, "M")),
    ("彰化三大有線", (8, "M")), ("臺北Conti", (9, "M")), ("連莊", (10, "M")),
]


@pytest.mark.parametrize("name,expected", OFFICIAL_SPELLINGS)
def test_resolve_team_known_names(conn, name, expected):
    assert resolve_team(conn, name) == expected


def test_resolve_team_gender_match_passes(conn):
    assert resolve_team(conn, "屏東台電", "M") == (1, "M")


def test_resolve_team_gender_mismatch_raises(conn):
    with pytest.raises(UnknownTeamError, match="性別"):
        resolve_team(conn, "屏東台電", "F")


def test_taoyuan_variants_resolve_same(conn):
    assert resolve_team(conn, "桃園臺灣產險") == resolve_team(conn, "桃園臺產") == (5, "M")


@pytest.mark.parametrize("ext_id,expected", list(EXT_TEAM_MAPS[21].items()))
def test_resolve_ext_team(conn, ext_id, expected):
    assert resolve_ext_team(conn, ext_id) == expected


def test_resolve_strips_whitespace(conn):
    assert resolve_team(conn, "  屏東台電\n") == (1, "M")


@pytest.mark.parametrize("bad", ["不存在隊", "", "   ", None])
def test_resolve_team_unknown_raises(conn, bad):
    with pytest.raises(UnknownTeamError):
        resolve_team(conn, bad)


def test_resolve_team_does_not_match_ext_stats_alias(conn):
    with pytest.raises(UnknownTeamError):
        resolve_team(conn, "1")


def test_resolve_ext_team_unknown_raises(conn):
    with pytest.raises(UnknownTeamError):
        resolve_ext_team(conn, 99)


def test_seed_is_idempotent(conn):
    before = (
        conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM team_aliases").fetchone()[0],
    )
    apply_team_seed(conn)
    after = (
        conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM team_aliases").fetchone()[0],
    )
    assert before == after == (14, 39)


def test_ext_team_map_20_targets_exist_in_teams(conn):
    existing = set(conn.execute("SELECT team_id, gender FROM teams").fetchall())
    assert set(EXT_TEAM_MAPS[20].values()) <= existing


def test_ext_team_map_20_pins_ten_teams(conn):
    expected = {
        1: ("屏東台電", "M"), 2: ("雲林美津濃", "M"), 3: ("彰化三大有線", "M"), 4: ("臺北Conti", "M"),
        5: ("龍邁建設", "M"), 6: ("連莊", "M"), 7: ("高雄台電", "F"), 8: ("臺北鯨華", "F"),
        9: ("新北中纖", "F"), 10: ("義力營造", "F"),
    }
    actual = {}
    for ext_id, (team_id, gender) in EXT_TEAM_MAPS[20].items():
        name = conn.execute(
            "SELECT team_name FROM teams WHERE team_id = ? AND gender = ?", (team_id, gender)
        ).fetchone()[0]
        actual[ext_id] = (name, gender)
    assert actual == expected


def test_ext_team_map_for_unknown_cup_raises():
    with pytest.raises(RuntimeError, match="99.*EXT_TEAM_MAPS"):
        ext_team_map_for(99)


def test_ext_team_map_for_returns_cup_map():
    assert ext_team_map_for(21) is EXT_TEAM_MAPS[21]
    assert ext_team_map_for(20) is EXT_TEAM_MAPS[20]


def test_short_name_fits_badge(conn):
    rows = conn.execute(
        "SELECT team_id, gender, short_name FROM teams "
        "WHERE is_active = 1 AND (short_name IS NULL OR length(short_name) > 6)"
    ).fetchall()
    assert rows == []


def test_active_teams_official_name_resolves_to_self(conn):
    rows = conn.execute(
        "SELECT team_id, gender, official_name FROM teams WHERE is_active = 1"
    ).fetchall()
    assert len(rows) == 10
    for team_id, gender, official in rows:
        assert official is not None
        assert resolve_team(conn, official) == (team_id, gender)


def test_colors_valid_and_unique_per_gender(conn):
    rows = conn.execute("SELECT gender, color FROM teams").fetchall()
    assert len(rows) == 14
    assert all(c and re.fullmatch(r"#[0-9A-Fa-f]{6}", c) for _, c in rows)
    for gender in ("M", "F"):
        colors = [c for g, c in rows if g == gender]
        assert len(colors) == len(set(colors))


def test_apply_team_seed_requires_migration(tmp_path):
    old = sqlite3.connect(tmp_path / "old.db")
    old.execute("CREATE TABLE teams (team_id INTEGER, team_name TEXT, gender TEXT)")
    with pytest.raises(RuntimeError, match="migrate_add_team_identity"):
        apply_team_seed(old)
    old.close()
