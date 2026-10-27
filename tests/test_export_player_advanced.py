import pytest

from tests.conftest import seed_player, seed_registration, seed_stat


RATIO_KEYS = [
    "attack_success_rate", "receive_excellent_rate", "serve_point_rate", "dig_excellent_rate",
    "points_per_game", "set_excellent_rate", "blocks_per_set", "defense_load", "defense_efficiency",
]
PR_KEYS = [
    "attack_success_rate", "receive_excellent_rate", "serve_point_rate", "dig_excellent_rate",
    "set_excellent_rate", "blocks_per_set", "defense_efficiency",
]


@pytest.fixture
def export(export_json):
    return lambda build: export_json(build, "player_advanced")[0]


def _oh(conn, player_id, name, reg_id, team_id, sets, atk_pts, atk_tot=20, gender="M", cup_id=21):
    seed_player(conn, player_id, name, gender)
    seed_registration(conn, reg_id, player_id, team_id=team_id, position="OH", gender=gender, cup_id=cup_id)
    # (sets, atk_tot, atk_pts, blk, srv_tot, srv_pts, rcv_tot, rcv_exc, dig_tot, dig_exc, set_tot, set_exc, pts)
    seed_stat(conn, reg_id, (sets, atk_tot, atk_pts, 2, 10, 1, 10, 5, 8, 4, 4, 2, atk_pts + 3))


def test_ratios_follow_old_site_formulas_and_count_golden_set_rows(export):
    """九項比率沿用舊站公式（百分比、分母為 0 給 0）；黃金決勝局列照算且算一場。"""
    def build(conn):
        seed_player(conn, 1, "男甲")
        seed_registration(conn, 1, 1, team_id=1, position="OH")
        seed_stat(conn, 1, (3, 10, 5, 1, 4, 2, 6, 3, 8, 4, 0, 0, 8), match_date="2025-11-01")
        seed_stat(conn, 1, (2, 20, 10, 2, 5, 1, 4, 2, 6, 3, 10, 5, 13), match_date="2025-11-08")
        seed_stat(conn, 1, (1, 3, 2, 0, 1, 1, 0, 0, 2, 1, 0, 0, 3), match_date="2025-11-08", golden=1)

    payload = export(build)

    [entry] = payload["players"]
    assert entry["player_id"] == 1 and entry["cup_id"] == 21 and entry["group"] == "men"
    assert entry["position"] == "OH"
    assert entry["games"] == 3 and entry["sets_played"] == 6
    assert entry["insufficient_sets"] is False
    assert entry["ratios"] == {
        "attack_success_rate": 51.52,   # 17 / 33
        "receive_excellent_rate": 50.0,  # 5 / 10
        "serve_point_rate": 40.0,        # 4 / 10
        "dig_excellent_rate": 50.0,      # 8 / 16
        "points_per_game": 8.0,          # 24 / 3
        "set_excellent_rate": 50.0,      # 5 / 10
        "blocks_per_set": 0.5,           # 3 / 6
        "defense_load": 26,              # 10 + 16
        "defense_efficiency": 50.0,      # 13 / 26
    }
    assert set(entry["pr"]) == set(PR_KEYS)


def test_pr_ranks_within_same_position_and_ties_take_average_rank(export):
    """PR 值 = 同位置內的平均名次 / 人數 × 100（同分取平均名次），不同位置互不影響。"""
    def build(conn):
        _oh(conn, 1, "甲", 1, 1, sets=10, atk_pts=5)    # 25%
        _oh(conn, 2, "乙", 2, 1, sets=10, atk_pts=10)   # 50%
        _oh(conn, 3, "丙", 3, 2, sets=10, atk_pts=10)   # 50%
        _oh(conn, 4, "丁", 4, 2, sets=10, atk_pts=15)   # 75%
        seed_player(conn, 5, "戊")
        seed_registration(conn, 5, 5, team_id=4, position="MB")
        seed_stat(conn, 5, (10, 20, 19, 2, 10, 1, 10, 5, 8, 4, 4, 2, 22))  # 95%，但是 MB

    players = export(build)["players"]

    pr = {p["player_id"]: p["pr"]["attack_success_rate"] for p in players}
    assert pr == {1: 25.0, 2: 62.5, 3: 62.5, 4: 100.0, 5: 100.0}
    assert all(p["pr"]["serve_point_rate"] == 62.5 for p in players if p["position"] == "OH")
    assert all(0 < value <= 100 for p in players for value in p["pr"].values())


def test_players_under_five_sets_get_null_pr_with_flag_and_stay_out_of_pool(export):
    """總局數未達 5：PR 值全空並標記局數不足，比率照常輸出，且不影響其他人的 PR 值。"""
    def build(conn):
        _oh(conn, 1, "甲", 1, 1, sets=5, atk_pts=5)
        _oh(conn, 2, "乙", 2, 1, sets=4, atk_pts=15)

    players = export(build)["players"]

    short = next(p for p in players if p["player_id"] == 2)
    assert short["insufficient_sets"] is True
    assert short["pr"] == dict.fromkeys(PR_KEYS)
    assert short["ratios"]["attack_success_rate"] == 75.0
    full = next(p for p in players if p["player_id"] == 1)
    assert full["insufficient_sets"] is False
    assert full["pr"]["attack_success_rate"] == 100.0


def test_player_without_position_keeps_ratios_but_no_pr(export):
    """補登球員位置為空：有比率、PR 值為空，也不進任何位置的計算池。"""
    def build(conn):
        _oh(conn, 1, "甲", 1, 1, sets=10, atk_pts=5)
        seed_player(conn, 2, "乙")
        seed_registration(conn, 2, 2, team_id=1, source="backfill")
        seed_stat(conn, 2, (10, 20, 15, 2, 10, 1, 10, 5, 8, 4, 4, 2, 18))

    payload = export(build)

    none_pos = next(p for p in payload["players"] if p["player_id"] == 2)
    assert none_pos["position"] is None and none_pos["insufficient_sets"] is False
    assert none_pos["pr"] == dict.fromkeys(PR_KEYS)
    assert none_pos["ratios"]["attack_success_rate"] == 75.0
    assert [d["position"] for d in payload["distributions"]] == ["OH"]
    assert payload["distributions"][0]["player_count"] == 1


def test_pr_pools_are_split_by_group_and_season(export):
    """同位置但不同組別或不同賽季的球員各自成池。"""
    def build(conn):
        _oh(conn, 1, "男甲", 1, 1, sets=10, atk_pts=5)
        _oh(conn, 2, "男乙", 2, 1, sets=10, atk_pts=15)
        _oh(conn, 3, "女甲", 3, 3, sets=10, atk_pts=1, gender="F")
        _oh(conn, 4, "男丙去年", 4, 1, sets=10, atk_pts=19, cup_id=20)
        seed_registration(conn, 5, 1, team_id=1, position="OH", cup_id=20)  # 男甲去年有登錄、沒統計

    payload = export(build)

    rows = {(p["player_id"], p["cup_id"]): p for p in payload["players"]}
    assert set(rows) == {(1, 21), (2, 21), (3, 21), (4, 20)}
    assert rows[(1, 21)]["pr"]["attack_success_rate"] == 50.0
    assert rows[(2, 21)]["pr"]["attack_success_rate"] == 100.0
    assert rows[(3, 21)]["group"] == "women" and rows[(3, 21)]["pr"]["attack_success_rate"] == 100.0
    assert rows[(4, 20)]["pr"]["attack_success_rate"] == 100.0
    assert [(d["cup_id"], d["group"], d["position"]) for d in payload["distributions"]] == [
        (20, "men", "OH"), (21, "men", "OH"), (21, "women", "OH"),
    ]
    assert [p["cup_id"] for p in payload["players"]] == [20, 21, 21, 21]


def test_distribution_lists_pool_values_sorted_with_median(export):
    """分佈 = 計算池內每位球員的指標值（升冪）與中位數，九項指標都有，前端不自己算。"""
    def build(conn):
        _oh(conn, 1, "甲", 1, 1, sets=10, atk_pts=15)
        _oh(conn, 2, "乙", 2, 1, sets=10, atk_pts=5)
        _oh(conn, 3, "丙", 3, 2, sets=10, atk_pts=10)
        _oh(conn, 4, "丁", 4, 2, sets=3, atk_pts=19)  # 局數不足，不進分佈

    [dist] = export(build)["distributions"]

    assert dist["player_count"] == 3
    assert set(dist["metrics"]) == set(RATIO_KEYS)
    assert dist["metrics"]["attack_success_rate"] == {"median": 50.0, "values": [25.0, 50.0, 75.0]}
    assert dist["metrics"]["defense_load"] == {"median": 18, "values": [18, 18, 18]}


def test_percentile_ranks_match_pandas_rank_pct_on_ties():
    """PR 值的手算名次必須與舊站 pandas rank(pct=True) 完全一致（含同分與單人）。"""
    pd = pytest.importorskip("pandas")
    from src.export.player_advanced import percentile_ranks

    for values in ([3.0], [0.0, 0.0], [5.5, 1.0, 5.5, 2.0, 5.5, 9.0, 0.0], [1, 2, 2, 3, 3, 3, 4, 10, 10]):
        expected = pd.Series(values).rank(pct=True).mul(100).round(1).tolist()
        assert percentile_ranks(values) == expected, values


@pytest.mark.parametrize("sets", [(None, None), (5, None)])
def test_unknown_sets_preserve_null_and_use_games_for_pr(export, sets):
    def build(conn):
        _oh(conn, 1, "甲", 1, 1, sets=sets[0], atk_pts=5)
        seed_stat(conn, 1, (sets[1], 20, 10, 2, 10, 1, 10, 5, 8, 4, 4, 2, 13), match_date="2025-11-08")
        _oh(conn, 2, "乙", 2, 1, sets=None, atk_pts=15)

    payload = export(build)
    full, short = payload["players"]
    assert full["sets_played"] is None
    assert full["ratios"]["blocks_per_set"] is None
    assert full["pr"]["blocks_per_set"] is None
    assert full["insufficient_sets"] is False
    assert full["pr"]["attack_success_rate"] == 100.0
    assert short["insufficient_sets"] is True
    assert all(value is None for value in short["pr"].values())
    metrics = payload["distributions"][0]["metrics"]
    assert metrics["blocks_per_set"] == {"median": None, "values": []}
    assert metrics["attack_success_rate"]["values"] == [37.5]
