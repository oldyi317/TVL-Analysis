from dataclasses import replace

import pytest

from src.etl.player_stats_store import (
    NoSeasonRoster, Registration, SeasonRosterMatcher, StatLine,
    fill_missing_registration, write_page_registration, write_stat_lines,
)
from tests.conftest import connect_seeded, seed_player, seed_season_roster


REG = Registration(player_id=7, team_id=2, gender="M", cup_id=22,
                   week_label="例行賽 Week 1", week_start_date="2026-10-24")
STATS = dict(match_date="2026-10-24", opponent="龍邁建設", attack_total=3,
             attack_points=4, block_points=5, serve_total=6, serve_points=7,
             receive_total=8, receive_excellent=9, dig_total=10, dig_excellent=11,
             set_total=12, set_excellent=13, total_points=14, is_golden_set=1)


@pytest.fixture
def conn(tmp_db_path):
    db = connect_seeded(tmp_db_path)
    db.execute("PRAGMA foreign_keys = ON")
    yield db
    db.close()


@pytest.fixture
def roster(conn):
    seed_player(conn, 7, "A　林")
    seed_player(conn, 8, "陳乙")
    seed_season_roster(conn, 7, 2, jersey=17, position="OH")
    seed_season_roster(conn, 8, 2, jersey=18, position="OP")
    return SeasonRosterMatcher(conn, 22)


def test_season_roster_prefers_name(roster):
    """同隊姓名優先於背號。"""
    assert roster.match(2, "M", " a\xa0林 ", 18) == 7
    assert roster.matched_by_jersey == {}


def test_season_roster_reports_jersey_match(roster, caplog):
    """以同隊背號對上並記進報告。"""
    assert roster.match(2, "M", "別名", 17) == 7
    entry = "  外部 別名　本地 A　林　雲林美津濃　背號 17"
    assert roster.matched_by_jersey == {entry: None}
    roster.log_reports()
    assert f"姓名不同、以背號對上（1 筆）：\n{entry}" in caplog.messages


def test_season_roster_reports_unmatched(roster, caplog):
    """別隊的姓名與背號對不上時回傳 None、發出警告並記進報告。"""
    assert roster.match(5, "M", "陳乙", 18) is None
    assert "[略過] 龍邁建設 背號 18 外部球員 陳乙 對不上賽季報名名單" in caplog.messages
    entry = "  龍邁建設　背號 18　外部姓名 陳乙"
    assert roster.skipped == {entry: None}
    roster.log_reports()
    assert f"略過（對不上賽季報名名單，1 筆）：\n{entry}" in caplog.messages


def test_missing_season_roster_raises_at_construction(conn, roster):
    """該季沒有賽季報名名單，建構時丟 NoSeasonRoster 並指出賽季。"""
    with pytest.raises(NoSeasonRoster, match="21"):
        SeasonRosterMatcher(conn, 21)


def registrations(conn):
    return conn.execute(
        """SELECT registration_id, jersey_number, position, week_start_date, source
           FROM roster_registrations"""
    ).fetchall()


def test_page_registration_replaces_backfill(conn, roster):
    """頁面週登錄覆寫背號、位置、起始日與來源。"""
    rid = fill_missing_registration(conn, REG)
    reg = replace(REG, week_start_date="2026-10-25")
    assert write_page_registration(conn, reg, jersey_number=2, position="MB") == rid
    assert registrations(conn) == [(rid, 2, "MB", "2026-10-25", "match_page")]


def test_page_registration_is_idempotent(conn):
    """重複寫頁面週登錄只有一列，整季無賽季報名名單也能寫。"""
    seed_player(conn, 7, "甲")
    rid = write_page_registration(conn, REG, jersey_number=2, position="OP")
    assert write_page_registration(conn, REG, jersey_number=2, position="OP") == rid
    assert registrations(conn) == [(rid, 2, "OP", REG.week_start_date, "match_page")]


def test_backfill_preserves_existing_registration(conn, roster):
    """補登錄保留已有週登錄的所有欄位。"""
    rid = write_page_registration(conn, REG, jersey_number=2, position="MB")
    before = registrations(conn)
    assert fill_missing_registration(conn, replace(REG, week_start_date="2026-10-31")) == rid
    assert registrations(conn) == before


def test_backfill_uses_season_roster_across_teams(conn, roster):
    """補登錄從賽季報名名單取背號位置，不限球隊。"""
    rid = fill_missing_registration(conn, replace(REG, team_id=5))
    assert registrations(conn) == [(rid, 17, "OH", REG.week_start_date, "backfill")]


@pytest.mark.parametrize("cup_id", [22, 21])
def test_backfill_without_roster_entry(conn, roster, cup_id):
    """該球員查無賽季報名名單或整季無名單時，補登錄的背號與位置為 NULL。"""
    seed_player(conn, 9, "丙")
    rid = fill_missing_registration(conn, replace(REG, player_id=9, cup_id=cup_id))
    assert registrations(conn) == [(rid, None, None, REG.week_start_date, "backfill")]


def test_stat_lines_preserve_fields_and_order(conn, roster):
    """單場數據逐欄寫入並保留順序，未給出賽局數時寫 NULL。"""
    first = fill_missing_registration(conn, REG)
    second = fill_missing_registration(conn, replace(REG, player_id=8))
    write_stat_lines(conn, [])
    assert not conn.execute("SELECT 1 FROM player_match_stats").fetchall()
    write_stat_lines(conn, [(second, StatLine(**STATS, sets_played=2)),
                            (first, StatLine(**STATS))])
    rows = conn.execute(
        """SELECT registration_id, match_date, opponent, sets_played, attack_total, attack_points,
                  block_points, serve_total, serve_points, receive_total, receive_excellent,
                  dig_total, dig_excellent, set_total, set_excellent, total_points, is_golden_set
           FROM player_match_stats ORDER BY stat_id"""
    ).fetchall()
    assert rows == [
        (rid, "2026-10-24", "龍邁建設", sets, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1)
        for rid, sets in [(second, 2), (first, None)]
    ]


def test_registration_requires_cup_id():
    """週登錄漏給 cup_id 時建構即失敗。"""
    with pytest.raises(TypeError, match="cup_id"):
        Registration(player_id=7, team_id=2, gender="M", week_label=REG.week_label,
                     week_start_date=REG.week_start_date)


def test_store_uses_outer_transaction_and_never_adds_players(conn):
    """外層 rollback 後三表皆空，比對成功與失敗都不新增球員。"""
    conn.execute("BEGIN")
    seed_player(conn, 7, "甲")
    seed_season_roster(conn, 7, 2, jersey=17)
    before = conn.execute("SELECT COUNT(*) FROM players").fetchone()
    matcher = SeasonRosterMatcher(conn, 22)
    assert matcher.match(2, "M", "甲", None) == 7
    assert matcher.match(2, "M", "乙", None) is None
    matcher.log_reports()
    rid = fill_missing_registration(conn, REG)
    assert write_page_registration(conn, REG, jersey_number=17, position="OH") == rid
    write_stat_lines(conn, [(rid, StatLine(**STATS))])
    assert conn.execute("SELECT COUNT(*) FROM players").fetchone() == before
    conn.rollback()
    for table in ("players", "roster_registrations", "player_match_stats"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)
