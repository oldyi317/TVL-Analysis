import re
import sqlite3
from pathlib import Path

import pytest

from src.etl.match_crawler import TEAM_ALIAS
from src.utils.constants import (
    EXT_TEAM_MAP,
    LEGACY_TEAMS,
    MATCH_TEAM_ALIASES,
    OPP_SHORT_TO_TEAM,
    TEAM_NAME_SHORT,
)
from src.utils.teams import UnknownTeamError, apply_team_seed, resolve_ext_team, resolve_team

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"

LEGACY_EXPECTED = {"彰化三大有線": (8, "M"), "臺北Conti": (9, "M"), "連莊": (10, "M")}


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    apply_team_seed(connection)
    yield connection
    connection.close()


def _expected_for_short(short: str) -> tuple[int, str]:
    return OPP_SHORT_TO_TEAM[short]


ALL_NAME_CASES = (
    [(k, _expected_for_short(v)) for k, v in TEAM_NAME_SHORT.items()]
    + [(v, _expected_for_short(v)) for v in TEAM_NAME_SHORT.values()]
    + [(k, v) for k, v in OPP_SHORT_TO_TEAM.items()]
    + [(k, _expected_for_short(v)) for k, v in TEAM_ALIAS.items()]
    + [(k, _expected_for_short(v)) for k, v in MATCH_TEAM_ALIASES.items()]
    + [(k, LEGACY_EXPECTED[k]) for k in LEGACY_TEAMS]
)


def test_legacy_expectation_covers_constant():
    assert set(LEGACY_EXPECTED) == set(LEGACY_TEAMS)


@pytest.mark.parametrize("name,expected", ALL_NAME_CASES)
def test_resolve_team_known_names(conn, name, expected):
    assert resolve_team(conn, name) == expected


def test_taoyuan_variants_resolve_same(conn):
    assert resolve_team(conn, "桃園臺灣產險") == resolve_team(conn, "桃園臺產") == (5, "M")


@pytest.mark.parametrize("ext_id,expected", list(EXT_TEAM_MAP.items()))
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


def test_constants_are_subset_of_db_aliases(conn):
    db_aliases = {r[0] for r in conn.execute("SELECT alias FROM team_aliases WHERE source <> 'ext_stats'")}
    const_names = (
        set(TEAM_NAME_SHORT) | set(TEAM_NAME_SHORT.values()) | set(OPP_SHORT_TO_TEAM)
        | set(TEAM_ALIAS) | set(MATCH_TEAM_ALIASES) | set(LEGACY_TEAMS)
    )
    missing = const_names - db_aliases
    assert not missing, f"常數內有、DB team_aliases 缺少：{sorted(missing)}"


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
    assert before == after == (12, 32)


def test_alias_count_matches_parametrized_cases():
    assert len({n for n, _ in ALL_NAME_CASES}) == 23


def test_short_name_equals_team_name(conn):
    assert conn.execute("SELECT COUNT(*) FROM teams WHERE short_name IS NOT team_name").fetchone()[0] == 0


def test_active_teams_official_name_resolves_to_self(conn):
    rows = conn.execute(
        "SELECT team_id, gender, official_name FROM teams WHERE is_active = 1"
    ).fetchall()
    assert len(rows) == 9
    for team_id, gender, official in rows:
        assert official is not None
        assert resolve_team(conn, official) == (team_id, gender)


def test_colors_valid_and_unique_per_gender(conn):
    rows = conn.execute("SELECT gender, color FROM teams").fetchall()
    assert len(rows) == 12
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
