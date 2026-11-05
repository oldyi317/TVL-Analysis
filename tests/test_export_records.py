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


def men_cumulative(payload):
    return {b["key"]: b["cumulative_records"] for b in payload["records"][0]["boards"]}


def test_cumulative_sums_every_season_per_player(export_json):
    payload, _ = export_json(build_two_seasons, "records")
    rows = men_cumulative(payload)["total_points"]
    assert [(r["rank"], r["name"], r["value"], r["season_count"], r["games"]) for r in rows] == [
        (1, "王小明", 42, 2, 2),
        (2, "李大華", 15, 1, 1),
    ]
    assert [r["value"] for r in men_cumulative(payload)["attack_points"]] == [28, 15]
    assert [r["value"] for r in men_cumulative(payload)["block_points"]] == [8]
    assert [r["value"] for r in men_cumulative(payload)["serve_points"]] == [6]


def test_cumulative_takes_team_and_season_from_latest_season_after_transfer(export_json):
    def build(conn):
        seed_player(conn, 1, "王小明")
        seed_registration(conn, 1, 1, 1, cup_id=20, week_start="2024-11-23")
        seed_registration(conn, 2, 1, 2, cup_id=21, week_start="2025-11-01")
        seed_stat(conn, 1, stat(30), "2024-11-23")
        seed_stat(conn, 2, stat(5), "2025-11-01")
        conn.execute("INSERT INTO team_season_names VALUES (2, 'M', 21, '舊隊全名', '舊隊')")

    payload, _ = export_json(build, "records")
    row = men_cumulative(payload)["total_points"][0]
    assert (row["value"], row["cup_id"], row["season"], row["team_id"], row["team_short"]) == (
        35, 21, "2025-26", 2, "舊隊")


def test_cumulative_is_in_progress_only_when_it_includes_current_season(export_json):
    def build(conn):
        seed_player(conn, 1, "王小明")
        seed_player(conn, 2, "李大華")
        seed_registration(conn, 1, 1, 1, cup_id=21)
        seed_registration(conn, 2, 1, 1, cup_id=22)
        seed_registration(conn, 3, 2, 2, cup_id=21)
        seed_stat(conn, 1, stat(5), "2025-11-01")
        seed_stat(conn, 2, stat(4), "2026-11-01")
        seed_stat(conn, 3, stat(6), "2025-11-01")

    payload, _ = export_json(build, "records")
    assert [(r["name"], r["in_progress"]) for r in men_cumulative(payload)["total_points"]] == [
        ("王小明", True), ("李大華", False)]


def test_cumulative_skips_seasons_without_any_game(export_json):
    def build(conn):
        seed_player(conn, 1, "王小明")
        seed_registration(conn, 1, 1, 1, cup_id=21)
        seed_registration(conn, 2, 1, 2, cup_id=22)
        seed_stat(conn, 1, stat(5), "2025-11-01")
        seed_stat(conn, 2, ZERO, "2026-11-01")

    payload, _ = export_json(build, "records")
    row = men_cumulative(payload)["total_points"][0]
    assert (row["season_count"], row["games"], row["season"], row["team_id"], row["in_progress"]) == (
        1, 1, "2025-26", 1, False)


def test_cumulative_team_is_last_team_with_stats_when_transferring_mid_latest_season(export_json):
    def build(conn):
        seed_player(conn, 1, "王小明")
        seed_registration(conn, 1, 1, 1, cup_id=20, week_start="2024-11-23")
        seed_registration(conn, 2, 1, 1, cup_id=21, week_start="2025-11-01")
        seed_registration(conn, 3, 1, 2, cup_id=21, week_start="2025-11-08")
        seed_stat(conn, 1, stat(30), "2024-11-23")
        seed_stat(conn, 2, stat(5), "2025-11-01")
        seed_stat(conn, 3, stat(6), "2025-11-08")

    payload, _ = export_json(build, "records")
    row = men_cumulative(payload)["total_points"][0]
    assert (row["value"], row["season_count"], row["games"], row["cup_id"], row["team_id"]) == (41, 2, 3, 21, 2)


MATCH_SQL = """
    INSERT INTO matches (game_id, gender, cup_id, match_date, home_team, away_team, home_team_id, away_team_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""


def men_match(payload):
    return {b["key"]: b["match_records"] for b in payload["records"][0]["boards"]}


def test_match_records_list_each_match_with_date_season_opponent_and_game(export_json):
    def build(conn):
        build_two_seasons(conn)
        conn.execute(MATCH_SQL, (5, "M", 20, "2024-11-23", "屏東台電", "雲林美津濃", 1, 2))
        conn.execute(MATCH_SQL, (9, "M", 21, "2025-11-01", "雲林美津濃", "屏東台電", 2, 1))

    payload, _ = export_json(build, "records")
    rows = men_match(payload)["total_points"]
    assert [(r["rank"], r["name"], r["value"], r["season"], r["match_date"], r["team_short"], r["opponent"],
             r["game_id"]) for r in rows] == [
        (1, "王小明", 30, "2024-25", "2024-11-23", "屏東台電", "雲林美津濃", 5),
        (2, "李大華", 15, "2025-26", "2025-11-01", "雲林美津濃", "屏東台電", 9),
        (3, "王小明", 12, "2025-26", "2025-11-01", "屏東台電", "雲林美津濃", 9),
    ]
    assert (rows[0]["player_id"], rows[0]["cup_id"], rows[0]["team_id"]) == (1, 20, 1)
    assert [r["value"] for r in men_match(payload)["block_points"]] == [6, 2]


def test_match_records_list_same_player_in_several_matches_of_one_season(export_json):
    def build(conn):
        seed_player(conn, 1, "王小明")
        seed_player(conn, 2, "李大華")
        seed_registration(conn, 1, 1, 1)
        seed_registration(conn, 2, 2, 2)
        seed_stat(conn, 1, stat(20), "2025-11-01")
        seed_stat(conn, 1, stat(25), "2025-11-08")
        seed_stat(conn, 2, stat(20), "2025-11-08", "屏東台電")

    payload, _ = export_json(build, "records")
    assert [(r["rank"], r["name"], r["match_date"]) for r in men_match(payload)["total_points"]] == [
        (1, "王小明", "2025-11-08"), (2, "李大華", "2025-11-08"), (2, "王小明", "2025-11-01")]
    assert men_match(payload)["serve_points"] == []


def test_match_records_exclude_golden_set(export_json):
    def build(conn):
        seed_player(conn, 1, "王小明")
        seed_registration(conn, 1, 1, 1)
        seed_stat(conn, 1, stat(5), "2025-11-01")
        seed_stat(conn, 1, stat(9), "2025-11-01", golden=1)

    payload, _ = export_json(build, "records")
    assert [r["value"] for r in men_match(payload)["total_points"]] == [5]


def test_match_record_without_match_page_has_null_game_and_raw_opponent(export_json):
    def build(conn):
        seed_player(conn, 1, "王小明")
        seed_registration(conn, 1, 1, 1)
        seed_stat(conn, 1, stat(5), "2025-11-01")
        seed_stat(conn, 1, stat(7), "2025-11-08", "已解散隊")

    payload, _ = export_json(build, "records")
    assert [(r["value"], r["opponent"], r["game_id"]) for r in men_match(payload)["total_points"]] == [
        (7, "已解散隊", None), (5, "雲林美津濃", None)]


def test_match_record_opponent_uses_season_name(export_json):
    def build(conn):
        seed_player(conn, 1, "王小明")
        seed_registration(conn, 1, 1, 1)
        seed_stat(conn, 1, stat(5), "2025-11-01")
        conn.execute("INSERT INTO team_season_names VALUES (2, 'M', 21, '舊隊全名', '舊隊')")

    payload, _ = export_json(build, "records")
    assert men_match(payload)["total_points"][0]["opponent"] == "舊隊"


def test_match_records_flag_current_season(export_json):
    def build(conn):
        seed_player(conn, 1, "王小明")
        seed_registration(conn, 1, 1, 1, cup_id=21)
        seed_registration(conn, 2, 1, 1, cup_id=22)
        seed_stat(conn, 1, stat(5), "2025-11-01")
        seed_stat(conn, 2, stat(4), "2026-11-01")

    payload, _ = export_json(build, "records")
    assert [(r["cup_id"], r["in_progress"]) for r in men_match(payload)["total_points"]] == [(21, False), (22, True)]


def test_match_records_same_player_same_value_lists_latest_match_first(export_json):
    def build(conn):
        seed_player(conn, 1, "王小明")
        seed_registration(conn, 1, 1, 1, cup_id=20, week_start="2024-11-23")
        seed_registration(conn, 2, 1, 1, cup_id=21)
        seed_stat(conn, 1, stat(20), "2024-11-23")
        seed_stat(conn, 2, stat(20), "2025-11-01")
        seed_stat(conn, 2, stat(20), "2025-11-08")

    payload, _ = export_json(build, "records")
    assert [(r["rank"], r["match_date"]) for r in men_match(payload)["total_points"]] == [
        (1, "2025-11-08"), (1, "2025-11-01"), (1, "2024-11-23")]
