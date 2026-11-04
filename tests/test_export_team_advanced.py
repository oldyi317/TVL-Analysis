import pytest

from tests.conftest import seed_player, seed_registration, seed_stat

MATCH_SQL = """
    INSERT INTO matches (game_id, gender, cup_id, match_date, is_golden_set, home_team, away_team,
                         home_team_id, away_team_id, home_set1, home_set2, home_set3, home_set4,
                         away_set1, away_set2, away_set3, away_set4)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
THREE_SETS = (25, 25, 25, None, 20, 22, 23, None)
FOUR_SETS = (25, 25, 20, 25, 10, 12, 25, 15)


def match(conn, game_id, home, away, sets=THREE_SETS, gender="M", cup_id=21, date="2025-11-01", golden=0):
    (home_name, home_id), (away_name, away_id) = home, away
    conn.execute(MATCH_SQL, (game_id, gender, cup_id, date, golden, home_name, away_name, home_id, away_id, *sets))


def stat(attack=(0, 0), block=0, serve=(0, 0), receive=(0, 0), dig=(0, 0), sets=(0, 0), sets_played=3):
    return (sets_played, attack[1], attack[0], block, serve[1], serve[0], receive[1], receive[0],
            dig[1], dig[0], sets[1], sets[0], 0)


PHOENIX, TAIPOWER, MIZUNO, LONGMAI = (("臺北鳳凰", 4), ("屏東台電", 1), ("雲林美津濃", 2), ("龍邁建設", 5))
KAOHSIUNG, JINGHUA = ("高雄台電", 4), ("臺北鯨華", 3)


def player(conn, pid, team_id, gender="M", cup_id=21):
    seed_player(conn, pid, f"球員{pid}", gender=gender)
    seed_registration(conn, pid, pid, team_id, cup_id=cup_id, gender=gender)


@pytest.fixture
def export(export_json):
    return lambda build: export_json(build, "team_advanced")[0]["team_advanced"]


def team(payload, cup_id, group, team_id):
    scope = next(s for s in payload if (s["cup_id"], s["group"]) == (cup_id, group))
    return next(t for t in scope["teams"] if t["team_id"] == team_id)


def test_same_team_id_men_and_women_are_aggregated_separately(export):
    def build(conn):
        match(conn, 1, PHOENIX, TAIPOWER, THREE_SETS)
        match(conn, 1, KAOHSIUNG, JINGHUA, FOUR_SETS, gender="F")
        player(conn, 1, 4)
        player(conn, 2, 4, gender="F")
        seed_stat(conn, 1, stat(attack=(5, 10), block=3), opponent="屏東台電")
        seed_stat(conn, 2, stat(attack=(4, 20), block=2), opponent="臺北鯨華")

    payload = export(build)
    men, women = team(payload, 21, "men", 4), team(payload, 21, "women", 4)
    assert men["metrics"]["attack_success_rate"] == {"numerator": 5, "denominator": 10, "value": 50.0, "rank": 1}
    assert men["metrics"]["blocks_per_set"]["value"] == 1.0
    assert women["metrics"]["attack_success_rate"]["value"] == 20.0
    assert women["metrics"]["blocks_per_set"] == {"numerator": 2, "denominator": 4, "value": 0.5, "rank": 1}


def test_metric_is_team_sum_divided_not_player_average_and_zero_denominator_is_null(export):
    def build(conn):
        match(conn, 1, PHOENIX, TAIPOWER)
        player(conn, 1, 4)
        player(conn, 2, 4)
        seed_stat(conn, 1, stat(attack=(1, 1)), opponent="屏東台電")
        seed_stat(conn, 2, stat(attack=(1, 9)), opponent="屏東台電")

    metrics = team(export(build), 21, "men", 4)["metrics"]
    assert metrics["attack_success_rate"]["value"] == 20.0
    assert metrics["serve_point_rate"] == {"numerator": 0, "denominator": 0, "value": None, "rank": None}


def test_blocks_per_set_uses_score_sets_when_sets_played_is_missing(export):
    def build(conn):
        match(conn, 1, PHOENIX, TAIPOWER, FOUR_SETS, cup_id=22, date="2026-10-17")
        player(conn, 1, 4, cup_id=22)
        seed_stat(conn, 1, stat(block=2, sets_played=None), match_date="2026-10-17", opponent="屏東台電")

    blocks = team(export(build), 22, "men", 4)["metrics"]["blocks_per_set"]
    assert blocks == {"numerator": 2, "denominator": 4, "value": 0.5, "rank": 1}


def test_golden_set_match_and_stat_rows_are_excluded(export):
    def build(conn):
        match(conn, 1, PHOENIX, TAIPOWER, THREE_SETS)
        match(conn, 2, PHOENIX, TAIPOWER, (15, None, None, None, 10, None, None, None), golden=1)
        player(conn, 1, 4)
        seed_stat(conn, 1, stat(attack=(5, 10), block=3), opponent="屏東台電")
        seed_stat(conn, 1, stat(attack=(5, 5), block=3), opponent="屏東台電", golden=1)

    phoenix = team(export(build), 21, "men", 4)
    assert phoenix["games"] == 1
    assert phoenix["metrics"]["attack_success_rate"]["value"] == 50.0
    assert phoenix["metrics"]["blocks_per_set"] == {"numerator": 3, "denominator": 3, "value": 1.0, "rank": 1}


def test_group_rank_ties_share_rank_and_next_rank_skips(export):
    def build(conn):
        match(conn, 1, PHOENIX, TAIPOWER)
        match(conn, 2, MIZUNO, LONGMAI)
        for pid, team_id, attack, opponent in (
            (1, 4, (5, 10), "屏東台電"), (2, 1, (5, 10), "臺北鳳凰"),
            (3, 2, (4, 10), "龍邁建設"), (4, 5, (0, 0), "雲林美津濃"),
        ):
            player(conn, pid, team_id)
            seed_stat(conn, pid, stat(attack=attack), opponent=opponent)

    payload = export(build)
    ranks = {tid: team(payload, 21, "men", tid)["metrics"]["attack_success_rate"]["rank"] for tid in (4, 1, 2, 5)}
    assert ranks == {4: 1, 1: 1, 2: 3, 5: None}


def test_games_counts_matches_with_stats_per_team(export):
    def build(conn):
        match(conn, 1, PHOENIX, TAIPOWER, date="2025-11-01")
        match(conn, 2, MIZUNO, PHOENIX, date="2025-11-08")
        match(conn, 3, PHOENIX, LONGMAI, date="2025-11-15")
        player(conn, 1, 4)
        seed_stat(conn, 1, stat(), match_date="2025-11-01", opponent="屏東台電")
        seed_stat(conn, 1, stat(), match_date="2025-11-08", opponent="雲林美津濃")

    payload = export(build)
    assert [s["group"] for s in payload] == ["men"]
    assert team(payload, 21, "men", 4)["games"] == 2


def test_points_opponent_error_is_derived_per_game_and_summed(export):
    def build(conn):
        # 第 1 場：鳳凰 75 分、技術得分 5+3+2=10 → 65；第 2 場：鳳凰 75 分、技術得分 4+1=5 → 70
        match(conn, 1, PHOENIX, TAIPOWER, THREE_SETS, date="2025-11-01")
        match(conn, 2, MIZUNO, PHOENIX, (20, 20, 20, None, 25, 25, 25, None), date="2025-11-08")
        player(conn, 1, 4)
        seed_stat(conn, 1, stat(attack=(5, 10), block=3, serve=(2, 4)), match_date="2025-11-01", opponent="屏東台電")
        seed_stat(conn, 1, stat(attack=(4, 10), block=1), match_date="2025-11-08", opponent="雲林美津濃")

    phoenix = team(export(build), 21, "men", 4)
    assert phoenix["points"] == {"attack": 9, "block": 4, "serve": 2, "opponent_error": 135, "total": 150}


def test_negative_opponent_error_warns_and_is_not_corrected(export, caplog):
    def build(conn):
        match(conn, 1, PHOENIX, TAIPOWER, THREE_SETS)
        player(conn, 1, 4)
        seed_stat(conn, 1, stat(attack=(80, 100)), opponent="屏東台電")

    with caplog.at_level("WARNING"):
        phoenix = team(export(build), 21, "men", 4)
    assert phoenix["points"]["opponent_error"] == 75 - 80
    assert any("game_id=1" in r.message and "team_id=4" in r.message and "-5" in r.message for r in caplog.records)


def test_points_exclude_golden_set_match_and_stat_rows(export):
    def build(conn):
        match(conn, 1, PHOENIX, TAIPOWER, THREE_SETS)
        match(conn, 2, PHOENIX, TAIPOWER, (15, None, None, None, 10, None, None, None), golden=1)
        player(conn, 1, 4)
        seed_stat(conn, 1, stat(attack=(5, 10), block=3), opponent="屏東台電")
        seed_stat(conn, 1, stat(attack=(9, 9), block=9, serve=(9, 9)), opponent="屏東台電", golden=1)

    assert team(export(build), 21, "men", 4)["points"] == {
        "attack": 5, "block": 3, "serve": 0, "opponent_error": 67, "total": 75}


def test_points_same_team_id_men_and_women_are_separate(export):
    def build(conn):
        match(conn, 1, PHOENIX, TAIPOWER, THREE_SETS)
        match(conn, 1, KAOHSIUNG, JINGHUA, FOUR_SETS, gender="F")
        player(conn, 1, 4)
        player(conn, 2, 4, gender="F")
        seed_stat(conn, 1, stat(attack=(5, 10)), opponent="屏東台電")
        seed_stat(conn, 2, stat(attack=(4, 20)), opponent="臺北鯨華")

    payload = export(build)
    assert team(payload, 21, "men", 4)["points"]["total"] == 75
    assert team(payload, 21, "women", 4)["points"] == {
        "attack": 4, "block": 0, "serve": 0, "opponent_error": 91, "total": 95}
