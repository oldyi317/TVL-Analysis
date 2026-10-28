import sqlite3
from pathlib import Path

from src.etl import stats_crawler
from src.etl.stats_crawler import PlayerResolver, collect_player_rows, resolve_registration_for_stats

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


def _make_conn(tmp_db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute("INSERT INTO teams (team_id, team_name, gender) VALUES (5, '新北中纖', 'F')")
    conn.execute(
        "INSERT INTO matches (game_id, gender, cup_id, match_date, round_name, home_team, away_team) "
        "VALUES (1, 'F', 21, '2025-11-01', '例行賽 Week 1', '新北中纖', '義力營造')"
    )
    conn.execute("INSERT INTO players (name, gender) VALUES ('張瓈文', 'F')")
    conn.commit()
    return conn


def test_resolve_registration_reuses_existing_match_page_row(tmp_db_path):
    conn = _make_conn(tmp_db_path)
    try:
        pid = conn.execute("SELECT player_id FROM players").fetchone()[0]
        conn.execute(
            """INSERT INTO roster_registrations
               (player_id, team_id, gender, cup_id, week_label, jersey_number, position, source)
               VALUES (?, 5, 'F', 21, '例行賽 Week 1', 2, 'OP', 'match_page')""",
            (pid,),
        )
        conn.commit()

        rid = resolve_registration_for_stats(conn, pid, 5, "F", "2025-11-01")

        row = conn.execute(
            "SELECT source, jersey_number FROM roster_registrations WHERE registration_id = ?",
            (rid,),
        ).fetchone()
        assert row == ("match_page", 2), "已有真實登錄時，不得覆蓋成 backfill"
    finally:
        conn.close()


def test_resolve_registration_creates_backfill_when_missing(tmp_db_path):
    conn = _make_conn(tmp_db_path)
    try:
        pid = conn.execute("SELECT player_id FROM players").fetchone()[0]

        rid = resolve_registration_for_stats(conn, pid, 5, "F", "2025-11-01")

        row = conn.execute(
            "SELECT source, jersey_number, position, week_label FROM roster_registrations WHERE registration_id = ?",
            (rid,),
        ).fetchone()
        assert row == ("backfill", None, None, "例行賽 Week 1")
    finally:
        conn.close()


def test_resolve_registration_is_idempotent(tmp_db_path):
    conn = _make_conn(tmp_db_path)
    try:
        pid = conn.execute("SELECT player_id FROM players").fetchone()[0]

        rid_first = resolve_registration_for_stats(conn, pid, 5, "F", "2025-11-01")
        rid_second = resolve_registration_for_stats(conn, pid, 5, "F", "2025-11-01")

        assert rid_first == rid_second
        count = conn.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def test_resolve_registration_scopes_by_cup_id(tmp_db_path):
    """他季（cup_id=20）已有同 player/team/week_label 登錄時，
    當季解析不得誤抓他季那筆，應另建當季 backfill。"""
    conn = _make_conn(tmp_db_path)
    try:
        pid = conn.execute("SELECT player_id FROM players").fetchone()[0]
        conn.execute(
            """INSERT INTO roster_registrations
               (player_id, team_id, gender, cup_id, week_label, jersey_number, position, source)
               VALUES (?, 5, 'F', 20, '例行賽 Week 1', 9, 'OH', 'match_page')""",
            (pid,),
        )
        conn.commit()

        rid = resolve_registration_for_stats(conn, pid, 5, "F", "2025-11-01", cup_id=21)

        row = conn.execute(
            "SELECT cup_id, source FROM roster_registrations WHERE registration_id = ?",
            (rid,),
        ).fetchone()
        assert row == (21, "backfill"), "應建立當季新登錄，而非誤用他季登錄"
        count = conn.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()[0]
        assert count == 2
    finally:
        conn.close()


def test_collect_player_rows_drops_records_without_round_name_and_writes_no_registration(tmp_db_path, monkeypatch, caplog):
    """#61：統計路徑遇到 matches 尚無 round_name 的場次，該筆不寫、也不補 backfill 登錄。"""
    conn = _make_conn(tmp_db_path)
    try:
        def record(match_date):
            return {
                "match_date": match_date, "opponent": "義力營造", "sets_played": 3,
                "attack_total": 0, "attack_points": 0, "block_points": 0,
                "serve_total": 0, "serve_points": 0, "receive_total": 0, "receive_excellent": 0,
                "dig_total": 0, "dig_excellent": 0, "set_total": 0, "set_excellent": 0,
                "total_points": 0, "is_golden_set": 0,
            }

        monkeypatch.setattr(stats_crawler, "fetch_player_stats",
                            lambda *a: [record("2025-11-01"), record("2025-11-08")])
        resolver = PlayerResolver(conn, 21)
        with caplog.at_level("WARNING", logger="src.etl.stats_crawler"):
            rows, skipped = collect_player_rows(None, resolver, 1, 5, "F", {"ext_player_id": 1, "name": "張瓈文"})

        assert [r[1] for r in rows] == ["2025-11-01"]
        assert skipped == 0
        labels = [r[0] for r in conn.execute("SELECT week_label FROM roster_registrations").fetchall()]
        assert labels == ["例行賽 Week 1"]
        assert any("2025-11-08" in r.getMessage() and "round_name" in r.getMessage() for r in caplog.records)
    finally:
        conn.close()
