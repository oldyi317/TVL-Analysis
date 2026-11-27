"""單場數據三個寫入路徑的整表行為鎖測試。"""

import json
import os
import sqlite3
from pathlib import Path

from src.etl import match_stats_writer, season_backfill, stats_crawler
from src.etl.official_box_score import parse_box_score
from src.utils.constants import EXT_TEAM_MAPS
from src.utils.db_config import get_connection
from tests.conftest import connect_seeded

FIXTURES = Path(__file__).parent / "fixtures"
LOCK_DIR = FIXTURES / "write_path_lock"
UPDATE_ENV = "WRITE_PATH_LOCK_UPDATE"

_PLAYER_SORT_FIELDS = ("name", "gender")
_REG_SORT_FIELDS = (
    "cup_id",
    "week_label",
    "team_id",
    "gender",
    "player_name",
    "player_gender",
    "week_start_date",
    "jersey_number",
    "position",
    "source",
)
_STAT_SORT_FIELDS = (
    "cup_id",
    "week_label",
    "team_id",
    "player_name",
    "player_gender",
    "match_date",
    "is_golden_set",
    "opponent",
    "sets_played",
    "attack_total",
    "attack_points",
    "block_points",
    "serve_total",
    "serve_points",
    "receive_total",
    "receive_excellent",
    "dig_total",
    "dig_excellent",
    "set_total",
    "set_excellent",
    "total_points",
)


def _sort_key(row: dict, fields: tuple[str, ...]) -> tuple:
    """建立可處理 None 值的決定性排序鍵。"""
    return tuple((row[f] is None, row[f]) for f in fields)


def _open_seeded(db_path: Path) -> sqlite3.Connection:
    conn = connect_seeded(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _dump_tables(conn: sqlite3.Connection) -> dict:
    """以自然鍵與全欄位 SQL 讀取三張核心表並排序輸出。"""
    player_rows = [
        {"name": name, "gender": gender}
        for name, gender in conn.execute("SELECT name, gender FROM players").fetchall()
    ]
    player_rows.sort(key=lambda r: _sort_key(r, _PLAYER_SORT_FIELDS))

    reg_cols = (
        "player_name",
        "player_gender",
        "team_id",
        "gender",
        "cup_id",
        "week_label",
        "week_start_date",
        "jersey_number",
        "position",
        "source",
    )
    reg_raw = conn.execute(
        """SELECT p.name, p.gender, r.team_id, r.gender, r.cup_id, r.week_label,
                  r.week_start_date, r.jersey_number, r.position, r.source
           FROM roster_registrations r
           LEFT JOIN players p ON p.player_id = r.player_id"""
    ).fetchall()
    reg_rows = [dict(zip(reg_cols, row)) for row in reg_raw]
    reg_rows.sort(key=lambda r: _sort_key(r, _REG_SORT_FIELDS))

    stat_cols = (
        "player_name",
        "player_gender",
        "team_id",
        "cup_id",
        "week_label",
        "match_date",
        "opponent",
        "sets_played",
        "attack_total",
        "attack_points",
        "block_points",
        "serve_total",
        "serve_points",
        "receive_total",
        "receive_excellent",
        "dig_total",
        "dig_excellent",
        "set_total",
        "set_excellent",
        "total_points",
        "is_golden_set",
    )
    stat_raw = conn.execute(
        """SELECT p.name, p.gender, r.team_id, r.cup_id, r.week_label,
                  s.match_date, s.opponent, s.sets_played,
                  s.attack_total, s.attack_points, s.block_points,
                  s.serve_total, s.serve_points,
                  s.receive_total, s.receive_excellent,
                  s.dig_total, s.dig_excellent,
                  s.set_total, s.set_excellent, s.total_points,
                  s.is_golden_set
           FROM player_match_stats s
           LEFT JOIN roster_registrations r ON r.registration_id = s.registration_id
           LEFT JOIN players p ON p.player_id = r.player_id"""
    ).fetchall()
    stat_rows = [dict(zip(stat_cols, row)) for row in stat_raw]
    stat_rows.sort(key=lambda r: _sort_key(r, _STAT_SORT_FIELDS))

    return {
        "players": {"count": len(player_rows), "rows": player_rows},
        "roster_registrations": {"count": len(reg_rows), "rows": reg_rows},
        "player_match_stats": {"count": len(stat_rows), "rows": stat_rows},
    }


def _assert_lock(conn: sqlite3.Connection, scenario: str) -> None:
    """寫出或比對指定情境的 JSON 基準檔。"""
    snapshot = _dump_tables(conn)
    rendered = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path = LOCK_DIR / f"{scenario}.json"
    if os.environ.get(UPDATE_ENV) == "1":
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        return
    assert path.exists(), f"缺少基準檔：{path}（請先以 {UPDATE_ENV}=1 產生）"
    assert snapshot == json.loads(path.read_text(encoding="utf-8"))


def _seed_player_with_roster(
    conn: sqlite3.Connection,
    *,
    name: str,
    gender: str,
    team_id: int,
    cup_id: int,
    jersey_number: int | None,
    position: str | None,
) -> int:
    pid = conn.execute(
        "INSERT INTO players (name, gender) VALUES (?, ?)", (name, gender)
    ).lastrowid
    conn.execute(
        """INSERT INTO season_rosters (player_id, team_id, gender, cup_id, jersey_number, position)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (pid, team_id, gender, cup_id, jersey_number, position),
    )
    return pid


def _stat_record(
    match_date: str,
    opponent: str,
    sets_played: int,
    values: tuple[int, ...],
    is_golden_set: int = 0,
) -> dict:
    """組出十二個技術欄皆不相同的統計紀錄字典。"""
    fields = (
        "attack_total",
        "attack_points",
        "block_points",
        "serve_total",
        "serve_points",
        "receive_total",
        "receive_excellent",
        "dig_total",
        "dig_excellent",
        "set_total",
        "set_excellent",
        "total_points",
    )
    assert len(values) == len(fields)
    record = {
        "match_date": match_date,
        "opponent": opponent,
        "sets_played": sets_played,
        "is_golden_set": is_golden_set,
    }
    record.update(zip(fields, values))
    return record


def test_lock_official_writer(tmp_db_path, monkeypatch):
    """情境 1：官網寫入器，涵蓋一般列、0-0 只寫週登錄、以背號對上與黃金決勝局。"""
    html_w281 = (FIXTURES / "boxscore_wgame_281.html").read_text(encoding="utf-8")
    html_m331 = (FIXTURES / "boxscore_game_331.html").read_text(encoding="utf-8")

    conn = _open_seeded(tmp_db_path)
    matches = [
        (281, "F", 22, "2026-10-24", "例行賽 Week 1", 0, "臺北鯨華", "高雄台電", 3, 4, 3, 1, 92, 89),
        (330, "M", 22, "2026-10-24", "例行賽 Week 1", 0, "屏東台電", "雲林美津濃", 1, 2, None, None, None, None),
        (331, "M", 22, "2026-10-25", "例行賽 Week 1", 0, "龍邁建設", "雲林美津濃", 5, 2, 0, 3, 56, 75),
        (3310, "M", 22, "2026-10-25", "例行賽 Week 1", 1, "龍邁建設", "雲林美津濃", 5, 2, 0, 3, 56, 75),
    ]
    conn.executemany(
        """INSERT INTO matches (game_id, gender, cup_id, match_date, round_name, is_golden_set,
                                home_team, away_team, home_team_id, away_team_id,
                                home_sets_won, away_sets_won, home_total, away_total)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        matches,
    )

    for gender, html, team_ids in (
        ("F", html_w281, (3, 4)),
        ("M", html_m331, (5, 2)),
    ):
        teams = parse_box_score(html)["teams"]
        for team, team_id in zip(teams, team_ids):
            for player in team["players"]:
                # 讓臺北鯨華 4 號改用不同本地姓名，驗證官網路徑能以同隊背號對上
                local_name = (
                    "金彩娜（賽季名）"
                    if gender == "F" and team_id == 3 and player["jersey_number"] == 4
                    else player["name"]
                )
                _seed_player_with_roster(
                    conn,
                    name=local_name,
                    gender=gender,
                    team_id=team_id,
                    cup_id=22,
                    jersey_number=player["jersey_number"],
                    position=player["position"],
                )
    conn.commit()

    monkeypatch.setattr(match_stats_writer.time, "sleep", lambda _s: None)
    pages = {
        ("F", 281): html_w281,
        ("M", 331): html_m331,
        ("M", 3310): html_m331,
    }
    match_stats_writer.write_match_stats(conn, lambda g, gid: pages[(g, gid)], delay=0)

    _assert_lock(conn, "official_writer")
    conn.close()


def test_lock_season_backfill(tmp_db_path, monkeypatch):
    """情境 2：歷史回填，無賽季報名名單，走全域姓名比對與新增球員。"""
    match_ids = [1, 2, 3, 101]
    cup19_dir = FIXTURES / "cup19"
    pages = {
        mid: (cup19_dir / f"match_{mid}_response.html").read_text(encoding="utf-8")
        for mid in match_ids
    }

    conn = _open_seeded(tmp_db_path)
    # 預先埋入同組既有球員與跨組同名球員，鎖住全域姓名比對與分組隔離行為
    conn.execute("INSERT INTO players (name, gender) VALUES ('郭柏毅', 'M')")
    conn.execute("INSERT INTO players (name, gender) VALUES ('陳朝璽', 'F')")
    conn.commit()

    monkeypatch.setattr(season_backfill.time, "sleep", lambda _s: None)
    monkeypatch.setattr(season_backfill, "fetch_match_page", lambda _cup_id, mid: pages.get(mid))

    season_backfill.run_backfill(conn, 19, match_ids=match_ids)

    _assert_lock(conn, "season_backfill")
    conn.close()


def test_lock_stats_crawler_with_roster(tmp_db_path, monkeypatch):
    """情境 3：外部統計爬蟲、該季有賽季報名名單。"""
    cup_id = 20
    team_id = 5
    gender = "F"
    ext_team_id = next(
        ext for ext, target in EXT_TEAM_MAPS[cup_id].items() if target == (team_id, gender)
    )

    conn = _open_seeded(tmp_db_path)
    conn.executemany(
        """INSERT INTO matches (game_id, gender, cup_id, match_date, round_name, home_team, away_team)
           VALUES (?, 'F', ?, ?, ?, ?, ?)""",
        [
            (1, cup_id, "2024-11-02", "例行賽 Week 1", "新北中纖", "義力營造"),
            (2, cup_id, "2024-11-03", "例行賽 Week 1", "義力營造", "新北中纖"),
            (3, cup_id, "2024-11-09", "例行賽 Week 2", "新北中纖", "義力營造"),
        ],
    )
    for name, jersey, position in (
        ("張瓈文", 2, "OP"),
        ("Darin Pinsuwan", 16, "OH"),
        ("杜家馨", 8, "MB"),
    ):
        _seed_player_with_roster(
            conn,
            name=name,
            gender=gender,
            team_id=team_id,
            cup_id=cup_id,
            jersey_number=jersey,
            position=position,
        )
    conn.commit()

    monkeypatch.setattr(stats_crawler.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        stats_crawler, "fetch_match_list", lambda _cup: [{"match_id": 1, "label": "第1週-女 1"}]
    )
    # 故意讓張瓈文在 Week 1 的出賽位置為 OH（不同於賽季報名名單 OP），驗證後續補登錄不覆寫 match_page
    roster_rows = [
        {
            "match_date": "2024-11-02",
            "title_text": "女子組 第1週",
            "team_id": team_id,
            "team_gender": gender,
            "jersey_number": 2,
            "name": "張瓈文",
            "position": "OH",
        },
        {
            "match_date": "2024-11-02",
            "title_text": "女子組 第1週",
            "team_id": team_id,
            "team_gender": gender,
            "jersey_number": 16,
            "name": "Darin",
            "position": "OH",
        },
        {
            "match_date": "2024-11-02",
            "title_text": "女子組 第1週",
            "team_id": team_id,
            "team_gender": gender,
            "jersey_number": 77,
            "name": "陌生人",
            "position": "L",
        },
    ]
    monkeypatch.setattr(stats_crawler, "fetch_match_roster", lambda _c, _s, _mid: roster_rows)
    stats_crawler.crawl_all_rosters(conn, cup_id=cup_id)
    conn.close()

    ext_players = [
        {"ext_player_id": 101, "name": "張瓈文", "jersey_number": 2},
        {"ext_player_id": 102, "name": "Darin", "jersey_number": 16},
        {"ext_player_id": 103, "name": "杜家馨", "jersey_number": 8},
        {"ext_player_id": 104, "name": "陌生人", "jersey_number": 77},
        {"ext_player_id": 105, "name": "無背號陌生人", "jersey_number": None},
    ]
    stats_by_ext_pid = {
        101: [
            _stat_record("2024-11-02", "義力營造", 4, (31, 14, 3, 17, 2, 23, 11, 21, 13, 8, 5, 19)),
            _stat_record("2024-11-09", "義力營造", 3, (29, 12, 4, 15, 1, 18, 9, 24, 16, 7, 6, 17)),
        ],
        102: [
            _stat_record("2024-11-02", "義力營造", 4, (37, 18, 5, 19, 3, 27, 14, 22, 10, 9, 2, 26)),
            _stat_record("2024-11-09", "義力營造", 5, (41, 20, 6, 21, 4, 33, 15, 25, 12, 11, 7, 30)),
        ],
        103: [
            _stat_record("2024-11-03", "義力營造", 3, (28, 11, 7, 14, 2, 16, 8, 19, 10, 5, 3, 20)),
            _stat_record(
                "2024-11-03",
                "義力營造",
                1,
                (12, 6, 2, 7, 1, 9, 4, 11, 5, 8, 3, 10),
                is_golden_set=1,
            ),
        ],
    }

    monkeypatch.setattr(stats_crawler, "get_connection", lambda: get_connection(db_path=tmp_db_path))
    monkeypatch.setattr(
        stats_crawler,
        "fetch_player_list",
        lambda tid, cup=None: ext_players if tid == ext_team_id and cup == cup_id else [],
    )
    monkeypatch.setattr(
        stats_crawler,
        "fetch_player_stats",
        lambda _season, _tid, ext_pid: stats_by_ext_pid.get(ext_pid, []),
    )
    stats_crawler.main(cup_id=cup_id)

    week2_roster_rows = [
        {
            "match_date": "2024-11-09",
            "title_text": "女子組 第2週",
            "team_id": team_id,
            "team_gender": gender,
            "jersey_number": 2,
            "name": "張瓈文",
            "position": "L",
        },
        {
            "match_date": "2024-11-09",
            "title_text": "女子組 第2週",
            "team_id": team_id,
            "team_gender": gender,
            "jersey_number": 99,
            "name": "Darin Pinsuwan",
            "position": "OH",
        },
    ]
    monkeypatch.setattr(
        stats_crawler, "fetch_match_list", lambda _cup: [{"match_id": 2, "label": "第2週-女 3"}]
    )
    monkeypatch.setattr(stats_crawler, "fetch_match_roster", lambda _c, _s, _mid: week2_roster_rows)

    verify_conn = get_connection(db_path=tmp_db_path)
    try:
        stats_crawler.crawl_all_rosters(verify_conn, cup_id=cup_id)
        _assert_lock(verify_conn, "stats_crawler_with_roster")
    finally:
        verify_conn.close()


def test_lock_stats_crawler_without_roster(tmp_db_path, monkeypatch):
    """情境 4：外部統計爬蟲、該季沒有賽季報名名單。"""
    cup_id = 21
    team_id = 5
    gender = "F"
    ext_team_id = next(
        ext for ext, target in EXT_TEAM_MAPS[cup_id].items() if target == (team_id, gender)
    )

    conn = _open_seeded(tmp_db_path)
    conn.executemany(
        """INSERT INTO matches (game_id, gender, cup_id, match_date, round_name, home_team, away_team)
           VALUES (?, 'F', ?, ?, ?, ?, ?)""",
        [
            (1, cup_id, "2025-11-01", "例行賽 Week 1", "新北中纖", "義力營造"),
            (2, cup_id, "2025-11-08", "例行賽 Week 2", "義力營造", "新北中纖"),
        ],
    )
    conn.execute("INSERT INTO players (name, gender) VALUES ('張瓈文', 'F')")
    conn.commit()

    monkeypatch.setattr(stats_crawler.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        stats_crawler, "fetch_match_list", lambda _cup: [{"match_id": 1, "label": "第1週-女 1"}]
    )
    roster_rows = [
        {
            "match_date": "2025-11-01",
            "title_text": "女子組 第1週",
            "team_id": team_id,
            "team_gender": gender,
            "jersey_number": 2,
            "name": "張瓈文",
            "position": "OP",
        },
        {
            "match_date": "2025-11-01",
            "title_text": "女子組 第1週",
            "team_id": team_id,
            "team_gender": gender,
            "jersey_number": 11,
            "name": "名單新球員",
            "position": "S",
        },
    ]
    monkeypatch.setattr(stats_crawler, "fetch_match_roster", lambda _c, _s, _mid: roster_rows)
    stats_crawler.crawl_all_rosters(conn, cup_id=cup_id)
    conn.close()

    ext_players = [
        {"ext_player_id": 201, "name": "張瓈文", "jersey_number": 2},
        {"ext_player_id": 202, "name": "名單新球員", "jersey_number": 11},
        {"ext_player_id": 203, "name": "統計新球員", "jersey_number": 9},
    ]
    stats_by_ext_pid = {
        201: [
            _stat_record("2025-11-01", "義力營造", 4, (35, 16, 4, 18, 2, 24, 12, 20, 11, 7, 5, 22)),
            _stat_record("2025-11-08", "義力營造", 3, (26, 10, 3, 14, 1, 19, 8, 17, 9, 6, 4, 15)),
        ],
        202: [
            _stat_record("2025-11-01", "義力營造", 4, (13, 5, 2, 16, 1, 7, 3, 18, 9, 88, 34, 8)),
        ],
        203: [
            _stat_record("2025-11-08", "義力營造", 5, (42, 19, 6, 20, 3, 31, 14, 27, 15, 12, 4, 28)),
        ],
    }

    monkeypatch.setattr(stats_crawler, "get_connection", lambda: get_connection(db_path=tmp_db_path))
    monkeypatch.setattr(
        stats_crawler,
        "fetch_player_list",
        lambda tid, cup=None: ext_players if tid == ext_team_id and cup == cup_id else [],
    )
    monkeypatch.setattr(
        stats_crawler,
        "fetch_player_stats",
        lambda _season, _tid, ext_pid: stats_by_ext_pid.get(ext_pid, []),
    )
    stats_crawler.main(cup_id=cup_id)

    verify_conn = get_connection(db_path=tmp_db_path)
    try:
        _assert_lock(verify_conn, "stats_crawler_without_roster")
    finally:
        verify_conn.close()
