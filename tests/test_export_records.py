from tests.conftest import seed_player, seed_registration, seed_stat

ZERO = (None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def stat(total_points, attack=0, block=0, serve=0):
    return (3, 0, attack, block, 0, serve, 0, 0, 0, 0, 0, 0, total_points)


def men_boards(payload):
    entry = payload["records"][0]
    assert entry["group"] == "men"
    return {b["key"]: b["season_records"] for b in entry["boards"]}


def build_two_seasons(conn):
    seed_player(conn, 1, "王小明")
    seed_player(conn, 2, "李大華")
    seed_registration(conn, 1, 1, 1, cup_id=20, position="OH")
    seed_registration(conn, 2, 1, 1, cup_id=21)
    seed_registration(conn, 3, 2, 2, cup_id=21)
    seed_stat(conn, 1, stat(30, attack=20, block=6, serve=4), "2024-11-23")
    seed_stat(conn, 2, stat(12, attack=8, block=2, serve=2), "2025-11-01")
    seed_stat(conn, 3, stat(15, attack=15), "2025-11-01", "屏東台電")


def test_season_records_rank_across_seasons_with_row_fields(export_json):
    payload, _ = export_json(build_two_seasons, "records")
    rows = men_boards(payload)["total_points"]
    assert [(r["rank"], r["name"], r["value"], r["cup_id"], r["season"], r["games"], r["team_short"])
            for r in rows] == [
        (1, "王小明", 30, 20, "2024-25", 1, "屏東台電"),
        (2, "李大華", 15, 21, "2025-26", 1, "雲林美津濃"),
        (3, "王小明", 12, 21, "2025-26", 1, "屏東台電"),
    ]
    assert rows[0]["player_id"] == 1 and rows[0]["team_id"] == 1
    assert [r["value"] for r in men_boards(payload)["attack_points"]] == [20, 15, 8]
    assert [r["value"] for r in men_boards(payload)["block_points"]] == [6, 2]
    assert [r["value"] for r in men_boards(payload)["serve_points"]] == [4, 2]


def test_ties_and_zero_values_follow_rank_rules(export_json):
    def build(conn):
        seed_player(conn, 1, "丙")
        seed_player(conn, 2, "乙")
        seed_player(conn, 3, "甲")
        for pid in (1, 2, 3):
            seed_registration(conn, pid, pid, 1)
        seed_stat(conn, 1, stat(10), "2025-11-01")
        seed_stat(conn, 2, stat(10), "2025-11-01")
        seed_stat(conn, 3, stat(8), "2025-11-01")

    payload, _ = export_json(build, "records")
    boards = men_boards(payload)
    assert [(r["rank"], r["name"]) for r in boards["total_points"]] == [(1, "丙"), (1, "乙"), (3, "甲")]
    assert boards["block_points"] == []


def test_tenth_place_ties_are_all_listed(export_json):
    def build(conn):
        for pid in range(1, 14):
            seed_player(conn, pid, f"球員{pid:02d}")
            seed_registration(conn, pid, pid, 1)
            seed_stat(conn, pid, stat(100 - pid if pid < 10 else 50 if pid < 13 else 40), "2025-11-01")

    payload, _ = export_json(build, "records")
    assert [r["rank"] for r in men_boards(payload)["total_points"]] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10]


def test_golden_set_rows_are_excluded(export_json):
    def build(conn):
        seed_player(conn, 1, "王小明")
        seed_registration(conn, 1, 1, 1)
        seed_stat(conn, 1, stat(5), "2025-11-01")
        seed_stat(conn, 1, stat(9), "2025-11-08", golden=1)

    payload, _ = export_json(build, "records")
    row = men_boards(payload)["total_points"][0]
    assert (row["value"], row["games"]) == (5, 1)
    assert payload["records"][0]["coverage"]["through_date"] == "2025-11-01"


def test_games_count_only_matches_with_any_nonzero_stat(export_json):
    def build(conn):
        seed_player(conn, 1, "王小明")
        seed_registration(conn, 1, 1, 1)
        seed_stat(conn, 1, stat(5), "2025-11-01")
        seed_stat(conn, 1, ZERO, "2025-11-08")
        seed_stat(conn, 1, (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), "2025-11-15")
        seed_stat(conn, 1, (None, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0), "2025-11-22")

    payload, _ = export_json(build, "records")
    assert men_boards(payload)["total_points"][0]["games"] == 2


def test_team_uses_season_name_and_last_registration_with_stats(export_json):
    def build(conn):
        seed_player(conn, 1, "王小明")
        seed_registration(conn, 1, 1, 1, week_start="2025-11-01")
        seed_registration(conn, 2, 1, 2, week_start="2025-11-08")
        seed_stat(conn, 1, stat(5), "2025-11-01")
        seed_stat(conn, 2, stat(6), "2025-11-08")
        conn.execute("INSERT INTO team_season_names VALUES (2, 'M', 21, '舊隊全名', '舊隊')")

    payload, _ = export_json(build, "records")
    row = men_boards(payload)["total_points"][0]
    assert (row["value"], row["games"], row["team_id"], row["team_short"]) == (11, 2, 2, "舊隊")


def test_only_latest_season_is_in_progress(export_json):
    def build(conn):
        seed_player(conn, 1, "王小明")
        seed_registration(conn, 1, 1, 1, cup_id=21)
        seed_registration(conn, 2, 1, 1, cup_id=22)
        seed_stat(conn, 1, stat(5), "2025-11-01")
        seed_stat(conn, 2, stat(4), "2026-11-01")

    payload, _ = export_json(build, "records")
    assert [(r["cup_id"], r["in_progress"]) for r in men_boards(payload)["total_points"]] == [(21, False), (22, True)]


def test_coverage_comes_from_seasons_with_data_per_group(export_json):
    def build(conn):
        build_two_seasons(conn)
        seed_player(conn, 3, "陳小美", gender="F")
        seed_registration(conn, 4, 3, 4, gender="F", cup_id=21)
        seed_stat(conn, 4, stat(7), "2025-12-06")

    payload, _ = export_json(build, "records")
    men, women = payload["records"]
    assert (men["group"], women["group"]) == ("men", "women")
    assert men["coverage"] == {"first_season": "2024-25", "season_count": 2, "through_date": "2025-11-01"}
    assert women["coverage"] == {"first_season": "2025-26", "season_count": 1, "through_date": "2025-12-06"}
    assert [r["name"] for r in women["boards"][0]["season_records"]] == ["陳小美"]
    assert all(r["name"] != "陳小美" for b in men["boards"] for r in b["season_records"])


def test_group_without_stats_has_null_coverage_and_no_boards(export_json):
    payload, _ = export_json(build_two_seasons, "records")
    women = payload["records"][1]
    assert women == {"group": "women", "coverage": None, "boards": []}


def test_stats_from_seasons_not_yet_published_are_excluded(export_json):
    def build(conn):
        build_two_seasons(conn)
        seed_registration(conn, 4, 1, 1, cup_id=19, week_start="2023-11-04")
        seed_stat(conn, 4, stat(99), "2023-11-04")

    payload, _ = export_json(build, "records")
    assert all(r["cup_id"] != 19 for b in payload["records"][0]["boards"] for r in b["season_records"])
    assert payload["records"][0]["coverage"]["first_season"] == "2024-25"
