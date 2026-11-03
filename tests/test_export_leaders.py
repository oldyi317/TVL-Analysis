from tests.conftest import seed_player, seed_registration, seed_stat

MATCH_SQL = """
    INSERT INTO matches (game_id, gender, cup_id, match_date, home_team, away_team, home_team_id, away_team_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""


def stat(total_points):
    return (3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, total_points)


def build(conn):
    seed_player(conn, 1, "王小明")
    seed_player(conn, 2, "李大華")
    seed_player(conn, 3, "陳小美", gender="F")
    seed_registration(conn, 1, 1, 1, position="OH")
    seed_registration(conn, 2, 2, 2)
    seed_registration(conn, 3, 3, 4, gender="F")
    conn.execute(MATCH_SQL, (1, "M", 21, "2025-11-01", "屏東台電", "雲林美津濃", 1, 2))
    conn.execute(MATCH_SQL, (2, "M", 21, "2025-11-08", "雲林美津濃", "屏東台電", 2, 1))
    conn.execute(MATCH_SQL, (3, "M", 20, "2024-11-23", "屏東台電", "雲林美津濃", 1, 2))
    seed_stat(conn, 1, stat(12), "2025-11-01", "雲林美津濃")
    seed_stat(conn, 2, stat(7), "2025-11-01", "屏東台電")
    seed_stat(conn, 1, stat(8), "2025-11-08", "雲林美津濃")
    seed_stat(conn, 2, stat(9), "2025-11-08", "屏東台電")


def test_group_with_stats_has_summary_and_total_points_board(export_json):
    payload, _ = export_json(build, "leaders")
    entry = payload["leaders"][0]
    assert (entry["cup_id"], entry["group"], entry["games"], entry["through_date"]) == (21, "men", 2, "2025-11-08")
    board = entry["boards"][0]
    assert board["key"] == "total_points"
    assert [(r["rank"], r["name"], r["team_short"], r["value"]) for r in board["rows"]] == [
        (1, "王小明", "屏東台電", 20),
        (2, "李大華", "雲林美津濃", 16),
    ]
    assert board["rows"][0]["position"] == "OH"
    assert board["rows"][0]["title"] == "得分王"


def test_only_groups_with_stats_have_entries_men_before_women(export_json):
    def with_women(conn):
        build(conn)
        conn.execute(MATCH_SQL, (4, "F", 21, "2025-11-01", "高雄台電", "新北中纖", 4, 5))
        seed_stat(conn, 3, stat(5), "2025-11-01", "新北中纖")

    payload, _ = export_json(with_women, "leaders")
    assert [(e["cup_id"], e["group"]) for e in payload["leaders"]] == [(21, "men"), (21, "women")]
    assert payload["leaders"][1]["games"] == 1
