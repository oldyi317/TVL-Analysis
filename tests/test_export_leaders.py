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
    assert [b["key"] for b in entry["boards"] if b["kind"] == "count"] == [
        "total_points", "attack_points", "block_points", "serve_points",
        "receive_excellent", "dig_excellent", "set_excellent"]
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


def test_golden_set_rows_are_excluded_from_counts_rates_and_thresholds(export_json):
    """黃金決勝局統計列不進計數、比率，也不算出賽一場或局數。"""
    def build_golden(conn):
        seed_player(conn, 1, "王小明")
        seed_registration(conn, 1, 1, 1, position="OH")
        conn.execute(MATCH_SQL, (1, "M", 21, "2025-11-01", "屏東台電", "雲林美津濃", 1, 2))
        conn.execute(MATCH_SQL, (2, "M", 21, "2025-11-08", "雲林美津濃", "屏東台電", 2, 1))
        # (sets, atk_tot, atk_pts, blk, srv_tot, srv_pts, rcv_tot, rcv_exc, dig_tot, dig_exc, set_tot, set_exc, pts)
        seed_stat(conn, 1, (3, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5), "2025-11-01", "雲林美津濃")
        seed_stat(conn, 1, (1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4), "2025-11-08", "雲林美津濃")
        seed_stat(conn, 1, (1, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6), "2025-11-08", "雲林美津濃", golden=1)

    payload, _ = export_json(build_golden, "leaders")
    boards = {b["key"]: b for b in payload["leaders"][0]["boards"]}
    assert boards["total_points"]["rows"][0]["value"] == 9
    assert boards["attack_points"]["rows"][0]["value"] == 9
    # 正規場次 4 局、攻擊 14 次；含黃金決勝局才會是 5 局、20 次而跨過門檻
    assert boards["attack_success_rate"]["rows"] == []
    assert boards["points_per_game"]["rows"] == []


def test_golden_set_only_appearance_does_not_count_as_a_game(export_json):
    def build_golden(conn):
        seed_player(conn, 1, "王小明")
        seed_registration(conn, 1, 1, 1, position="OH")
        conn.execute(MATCH_SQL, (1, "M", 21, "2025-11-01", "屏東台電", "雲林美津濃", 1, 2))
        conn.execute(MATCH_SQL, (2, "M", 21, "2025-11-08", "雲林美津濃", "屏東台電", 2, 1))
        # 局數未知時門檻是出賽 2 場
        seed_stat(conn, 1, (None, 20, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10), "2025-11-01", "雲林美津濃")
        seed_stat(conn, 1, (None, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1), "2025-11-08", "雲林美津濃", golden=1)

    payload, _ = export_json(build_golden, "leaders")
    boards = {b["key"]: b for b in payload["leaders"][0]["boards"]}
    assert boards["attack_success_rate"]["rows"] == []


def test_each_group_has_four_volume_rates_without_golden_set_rows(export_json):
    def build_volume(conn):
        seed_player(conn, 1, "王小明")
        seed_player(conn, 2, "陳小美", gender="F")
        seed_registration(conn, 1, 1, 1, position="S")
        seed_registration(conn, 2, 2, 4, gender="F", position="OH")
        conn.execute(MATCH_SQL, (1, "M", 21, "2025-11-01", "屏東台電", "雲林美津濃", 1, 2))
        conn.execute(MATCH_SQL, (2, "F", 21, "2025-11-01", "高雄台電", "新北中纖", 4, 5))
        # (sets, atk_tot, atk_pts, blk, srv_tot, srv_pts, rcv_tot, rcv_exc, dig_tot, dig_exc, set_tot, set_exc, pts)
        seed_stat(conn, 1, (5, 20, 10, 0, 0, 0, 10, 4, 10, 5, 30, 12, 10), "2025-11-01", "雲林美津濃")
        seed_stat(conn, 1, (1, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10), "2025-11-01", "雲林美津濃", golden=1)
        seed_stat(conn, 2, (5, 30, 12, 0, 0, 0, 10, 6, 10, 2, 20, 10, 12), "2025-11-01", "新北中纖")

    payload, _ = export_json(build_volume, "leaders")
    men, women = payload["leaders"]
    assert [v["key"] for v in men["volume_rates"]] == [
        "attack_success_rate", "receive_excellent_rate", "dig_excellent_rate", "set_excellent_rate"]
    assert men["volume_rates"][0] == {
        "key": "attack_success_rate", "median_count": 20, "median_rate": 50.0,
        "points": [{"player_id": 1, "name": "王小明", "team_id": 1, "team_short": "屏東台電", "count": 20, "rate": 50.0}],
    }
    assert [p["player_id"] for p in women["volume_rates"][0]["points"]] == [2]
    assert women["volume_rates"][3]["points"] == []
