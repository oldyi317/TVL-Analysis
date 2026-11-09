import pytest

from tests.conftest import seed_player, seed_registration, seed_season_roster, seed_stat


RATIO_KEYS = [
    "attack_success_rate", "receive_excellent_rate", "serve_point_rate", "dig_excellent_rate",
    "points_per_game", "set_excellent_rate", "blocks_per_set", "blocks_per_game", "defense_load",
    "defense_efficiency",
]
PR_KEYS = [
    "attack_success_rate", "receive_excellent_rate", "serve_point_rate", "dig_excellent_rate",
    "set_excellent_rate", "blocks_per_set", "blocks_per_game", "defense_efficiency",
]
TEAM_NAMES = {1: "屏東台電", 2: "雲林美津濃", 4: "高雄台電", 5: "新北中纖"}
OPPONENT = {1: 2, 2: 1, 4: 5, 5: 4}
BELOW = (3, 3, 3, 0, 3, 0, 3, 0, 3, 0, 0, 0, 3)  # 各項次數都遠低於門檻


@pytest.fixture
def export(export_json):
    return lambda build: export_json(build, "player_advanced")[0]


def _match(conn, game_id=1, home=1, away=2, date="2025-11-01", gender="M", cup_id=21, sets=10):
    """例行賽一場，兩隊各打 sets 局；sets=None 表示官網尚無比分。"""
    won = (None, None) if sets is None else (3, sets - 3)
    conn.execute(
        """INSERT INTO matches (game_id, gender, cup_id, match_date, home_team, away_team, home_team_id,
                                away_team_id, round_name, home_sets_won, away_sets_won)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, '例行賽 Week 1', ?, ?)""",
        (game_id, gender, cup_id, date, TEAM_NAMES[home], TEAM_NAMES[away], home, away, *won),
    )


def _oh(conn, player_id, name, reg_id, team_id, atk_pts, atk_tot=20, gender="M", cup_id=21, date="2025-11-01"):
    seed_player(conn, player_id, name, gender)
    seed_registration(conn, reg_id, player_id, team_id=team_id, position="OH", gender=gender, cup_id=cup_id)
    # (sets, atk_tot, atk_pts, blk, srv_tot, srv_pts, rcv_tot, rcv_exc, dig_tot, dig_exc, set_tot, set_exc, pts)
    seed_stat(conn, reg_id, (3, atk_tot, atk_pts, 2, 10, 1, 10, 5, 10, 4, 4, 2, atk_pts + 3),
              match_date=date, opponent=TEAM_NAMES[OPPONENT[team_id]])


def test_ratios_follow_old_site_formulas_and_count_golden_set_rows(export):
    """十項比率沿用舊站公式（百分比、分母為 0 給 0）；每局攔網分母是球隊已打局數；黃金決勝局列照算且算一場。"""
    def build(conn):
        seed_player(conn, 1, "男甲")
        seed_registration(conn, 1, 1, team_id=1, position="OH")
        _match(conn, 1, date="2025-11-01", sets=3)
        _match(conn, 2, date="2025-11-08", sets=3)
        seed_stat(conn, 1, (3, 10, 5, 1, 4, 2, 6, 3, 8, 4, 0, 0, 8), match_date="2025-11-01")
        seed_stat(conn, 1, (2, 20, 10, 2, 5, 1, 4, 2, 6, 3, 10, 5, 13), match_date="2025-11-08")
        seed_stat(conn, 1, (1, 3, 2, 0, 1, 1, 0, 0, 2, 1, 0, 0, 3), match_date="2025-11-08", golden=1)

    payload = export(build)

    [entry] = payload["players"]
    assert entry["player_id"] == 1 and entry["cup_id"] == 21 and entry["group"] == "men"
    assert entry["position"] == "OH"
    assert entry["games"] == 3 and entry["team_sets"] == 7
    assert entry["below_threshold"] is False
    assert "sets_played" not in entry and "insufficient_sets" not in entry
    assert entry["ratios"] == {
        "attack_success_rate": 51.52,   # 17 / 33
        "receive_excellent_rate": 50.0,  # 5 / 10
        "serve_point_rate": 40.0,        # 4 / 10
        "dig_excellent_rate": 50.0,      # 8 / 16
        "points_per_game": 8.0,          # 24 / 3
        "set_excellent_rate": 50.0,      # 5 / 10
        "blocks_per_set": 0.43,          # 3 / 球隊全季 7 局（含黃金決勝局 1 局）
        "blocks_per_game": 1.0,          # 3 / 3
        "defense_load": 26,              # 10 + 16
        "defense_efficiency": 50.0,      # 13 / 26
    }
    assert set(entry["pr"]) == set(PR_KEYS)
    assert "player_count" not in payload["distributions"][0]


def test_pr_ranks_within_same_position_and_ties_take_average_rank(export):
    """PR 值 = 同位置內的平均名次 / 人數 × 100（同分取平均名次），不同位置互不影響。"""
    def build(conn):
        _match(conn)
        _oh(conn, 1, "甲", 1, 1, atk_pts=5)    # 25%
        _oh(conn, 2, "乙", 2, 1, atk_pts=10)   # 50%
        _oh(conn, 3, "丙", 3, 2, atk_pts=10)   # 50%
        _oh(conn, 4, "丁", 4, 2, atk_pts=15)   # 75%
        seed_player(conn, 5, "戊")
        seed_registration(conn, 5, 5, team_id=2, position="MB")
        seed_stat(conn, 5, (10, 20, 19, 2, 10, 1, 10, 5, 10, 4, 4, 2, 22), opponent="屏東台電")  # 95%，但是 MB

    players = export(build)["players"]

    pr = {p["player_id"]: p["pr"]["attack_success_rate"] for p in players}
    assert pr == {1: 25.0, 2: 62.5, 3: 62.5, 4: 100.0, 5: 100.0}
    assert all(p["pr"]["dig_excellent_rate"] == 62.5 for p in players if p["position"] == "OH")
    assert all(0 < value <= 100 for p in players for value in p["pr"].values() if value is not None)


def test_player_below_every_threshold_gets_null_pr_with_flag_and_stays_out_of_pool(export):
    """八項 PR 指標次數都未達門檻：PR 值全空並標記未達門檻，比率照常輸出，且不影響其他人的 PR 值。"""
    def build(conn):
        _match(conn)
        _oh(conn, 1, "甲", 1, 1, atk_pts=5)
        seed_player(conn, 2, "乙")
        seed_registration(conn, 2, 2, team_id=1, position="OH")
        seed_stat(conn, 2, (3, 3, 3, 0, 3, 0, 3, 0, 3, 0, 0, 0, 3))

    players = export(build)["players"]

    short = next(p for p in players if p["player_id"] == 2)
    assert short["below_threshold"] is True
    assert short["pr"] == dict.fromkeys(PR_KEYS)
    assert short["ratios"]["attack_success_rate"] == 100.0
    full = next(p for p in players if p["player_id"] == 1)
    assert full["below_threshold"] is False
    assert full["pr"]["attack_success_rate"] == 100.0


def test_player_without_position_keeps_ratios_but_no_pr(export):
    """補登球員位置為空：有比率、PR 值為空，也不進任何位置的計算池。"""
    def build(conn):
        _match(conn)
        _oh(conn, 1, "甲", 1, 1, atk_pts=5)
        seed_player(conn, 2, "乙")
        seed_registration(conn, 2, 2, team_id=1, source="backfill")
        seed_stat(conn, 2, (10, 20, 15, 2, 10, 1, 10, 5, 10, 4, 4, 2, 18))

    payload = export(build)

    none_pos = next(p for p in payload["players"] if p["player_id"] == 2)
    assert none_pos["position"] is None and none_pos["below_threshold"] is False
    assert none_pos["pr"] == dict.fromkeys(PR_KEYS)
    assert none_pos["ratios"]["attack_success_rate"] == 75.0
    assert [d["position"] for d in payload["distributions"]] == ["OH"]
    assert payload["distributions"][0]["metrics"]["attack_success_rate"]["values"] == [25.0]


def test_blank_registration_position_falls_back_to_season_roster_position(export):
    """週登錄位置空白、同隊賽季報名名單有位置：以報名名單位置進 PR 計算池與分佈。"""
    def build(conn):
        _match(conn)
        _oh(conn, 1, "甲", 1, 1, atk_pts=5)
        seed_player(conn, 2, "乙")
        seed_registration(conn, 2, 2, team_id=1)
        seed_season_roster(conn, 2, team_id=1, cup_id=21, position="OH")
        seed_stat(conn, 2, (10, 20, 15, 2, 10, 1, 10, 5, 10, 4, 4, 2, 18))

    payload = export(build)

    fallback = next(p for p in payload["players"] if p["player_id"] == 2)
    assert fallback["position"] == "OH"
    assert fallback["pr"]["attack_success_rate"] == 100.0
    [dist] = payload["distributions"]
    assert dist["position"] == "OH" and dist["metrics"]["attack_success_rate"]["values"] == [25.0, 75.0]


def test_pr_pools_are_split_by_group_and_season(export):
    """同位置但不同組別或不同賽季的球員各自成池。"""
    def build(conn):
        _match(conn)
        _match(conn, 2, home=4, away=5, gender="F")
        _match(conn, 3, cup_id=20, date="2024-11-23")
        _oh(conn, 1, "男甲", 1, 1, atk_pts=5)
        _oh(conn, 2, "男乙", 2, 1, atk_pts=15)
        _oh(conn, 3, "女甲", 3, 4, atk_pts=1, gender="F")
        _oh(conn, 4, "男丙去年", 4, 1, atk_pts=19, cup_id=20, date="2024-11-23")
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
    """分佈 = 該指標達門檻的球員值（升冪）與中位數，十項指標都有，前端不自己算。"""
    def build(conn):
        _match(conn)
        _oh(conn, 1, "甲", 1, 1, atk_pts=15)
        _oh(conn, 2, "乙", 2, 1, atk_pts=5)
        _oh(conn, 3, "丙", 3, 2, atk_pts=10)
        seed_player(conn, 4, "丁")
        seed_registration(conn, 4, 4, team_id=2, position="OH")
        seed_stat(conn, 4, BELOW, opponent="屏東台電")  # 全項未達門檻，不進分佈

    [dist] = export(build)["distributions"]

    assert set(dist["metrics"]) == set(RATIO_KEYS)
    assert dist["metrics"]["attack_success_rate"] == {"median": 50.0, "values": [25.0, 50.0, 75.0]}
    assert dist["metrics"]["defense_load"] == {"median": 20, "values": [20, 20, 20]}


def test_per_metric_pool_at_export_level_only_counts_metric_qualifiers(export):
    """同隊兩人：乙攻擊次數未達但防守達標 → 乙只有防守到位率有 PR，也只進該項分佈。"""
    def build(conn):
        _match(conn)
        _oh(conn, 1, "甲", 1, 1, atk_pts=10)  # 攻擊 20 次、防守 10 次 → 兩項皆達標
        seed_player(conn, 2, "乙")
        seed_registration(conn, 2, 2, team_id=1, position="OH")
        seed_stat(conn, 2, (3, 19, 19, 0, 0, 0, 0, 0, 10, 8, 0, 0, 19))

    payload = export(build)

    first, second = payload["players"]
    assert first["pr"]["attack_success_rate"] == 100.0 and second["pr"]["attack_success_rate"] is None
    assert (first["pr"]["dig_excellent_rate"], second["pr"]["dig_excellent_rate"]) == (50.0, 100.0)
    assert second["below_threshold"] is False
    metrics = payload["distributions"][0]["metrics"]
    assert metrics["attack_success_rate"]["values"] == [50.0]
    assert metrics["dig_excellent_rate"]["values"] == [40.0, 80.0]


def test_missing_team_sets_keep_blocks_per_set_null_but_other_metrics_work(export):
    """官網尚無比分、球隊已打局數查無：每局攔網為 None 且不進 PR 與分佈，其他指標照常。"""
    def build(conn):
        _match(conn, 1, date="2025-11-01", sets=None)
        _match(conn, 2, date="2025-11-08", sets=None)
        _oh(conn, 1, "甲", 1, 1, atk_pts=10)
        seed_stat(conn, 1, (3, 20, 10, 2, 10, 1, 10, 5, 10, 4, 4, 2, 13), match_date="2025-11-08",
                  opponent="雲林美津濃")

    payload = export(build)

    [entry] = payload["players"]
    assert entry["team_sets"] == 0
    assert entry["ratios"]["blocks_per_set"] is None and entry["pr"]["blocks_per_set"] is None
    assert entry["ratios"]["blocks_per_game"] == 2.0 and entry["pr"]["blocks_per_game"] == 100.0
    assert entry["pr"]["attack_success_rate"] == 100.0
    metrics = payload["distributions"][0]["metrics"]
    assert metrics["blocks_per_set"] == {"median": None, "values": []}
    assert metrics["blocks_per_game"] == {"median": 2.0, "values": [2.0]}


def test_percentile_ranks_match_pandas_rank_pct_on_ties():
    """PR 值的手算名次必須與舊站 pandas rank(pct=True) 完全一致（含同分與單人）。"""
    pd = pytest.importorskip("pandas")
    from src.export.player_advanced import percentile_ranks

    for values in ([3.0], [0.0, 0.0], [5.5, 1.0, 5.5, 2.0, 5.5, 9.0, 0.0], [1, 2, 2, 3, 3, 3, 4, 10, 10]):
        expected = pd.Series(values).rank(pct=True).mul(100).round(1).tolist()
        assert percentile_ranks(values) == expected, values


def test_blocks_per_set_divides_by_team_sets_and_ignores_sets_played():
    from src.export.player_advanced import compute_ratios

    totals = {"sets_played": 99, "games": 4, "block_points": 6, "attack_total": 0, "attack_points": 0,
              "receive_total": 0, "receive_excellent": 0, "serve_total": 0, "serve_points": 0,
              "dig_total": 0, "dig_excellent": 0, "total_points": 0, "set_total": 0, "set_excellent": 0}
    assert compute_ratios(totals, 12)["blocks_per_set"] == 0.5
    assert compute_ratios(totals, 0)["blocks_per_set"] is None
    assert compute_ratios(totals)["blocks_per_set"] is None


def prow(player_id, team_id=1, position="OH", **changes):
    base = dict.fromkeys(("attack_total", "attack_points", "block_points", "serve_total", "serve_points",
                          "receive_total", "receive_excellent", "dig_total", "dig_excellent", "set_total",
                          "set_excellent", "total_points"), 0)
    return {"player_id": player_id, "team_id": team_id, "position": position, "games": 5, "sets_played": None,
            **base, **changes}


def test_pr_is_per_metric_only_among_players_meeting_that_metric_threshold():
    """球隊 10 局：攻擊門檻 20、防守 10。乙攻擊未達但防守達標 → 乙只有防守到位率有 PR，也只進該項分佈。"""
    from src.export.player_advanced import compute_advanced, distributions

    rows = [prow(1, attack_total=20, attack_points=10, dig_total=10, dig_excellent=2),
            prow(2, attack_total=19, attack_points=19, dig_total=10, dig_excellent=8),
            prow(3, attack_total=20, attack_points=15, dig_total=0)]
    results = {r["player_id"]: r for r in compute_advanced(rows, {1: 10})}

    assert results[1]["pr"]["attack_success_rate"] == 50.0
    assert results[3]["pr"]["attack_success_rate"] == 100.0
    assert results[2]["pr"]["attack_success_rate"] is None
    assert results[2]["pr"]["dig_excellent_rate"] == 100.0
    assert results[1]["pr"]["dig_excellent_rate"] == 50.0
    assert results[3]["pr"]["dig_excellent_rate"] is None
    metrics = distributions(list(results.values()))["OH"]
    assert metrics["attack_success_rate"] == [50.0, 75.0]
    assert metrics["dig_excellent_rate"] == [20.0, 80.0]


def test_below_threshold_only_when_all_eight_pr_metrics_unmet_and_team_sets_reported():
    from src.export.player_advanced import compute_advanced

    rows = [prow(1, attack_total=9), prow(2, attack_total=20), prow(3, team_id=2, attack_total=10)]
    results = {r["player_id"]: r for r in compute_advanced(rows, {1: 10})}

    assert [results[i]["below_threshold"] for i in (1, 2, 3)] == [True, False, False]
    assert [results[i]["team_sets"] for i in (1, 2, 3)] == [10, 10, 0]
    assert not {"sets_played", "insufficient_sets"} & set(results[1])


def test_blocks_per_set_uses_team_sets_and_is_none_without_them():
    from src.export.player_advanced import compute_advanced, distributions

    rows = [prow(1, attack_total=20, block_points=5), prow(2, team_id=2, attack_total=20, block_points=5)]
    results = {r["player_id"]: r for r in compute_advanced(rows, {1: 10})}

    assert results[1]["ratios"]["blocks_per_set"] == 0.5
    assert results[2]["ratios"]["blocks_per_set"] is None and results[2]["pr"]["blocks_per_set"] is None
    assert distributions(list(results.values()))["OH"]["blocks_per_set"] == [0.5]


def test_positions_with_only_below_threshold_players_have_no_distribution():
    from src.export.player_advanced import compute_advanced, distributions

    rows = [prow(1, position="MB", attack_total=3), prow(2, position="OH", attack_total=20)]
    assert list(distributions(compute_advanced(rows, {1: 10}))) == ["OH"]


def test_leaders_boards_and_pr_pools_agree_on_who_qualifies():
    """一致性鎖：同一組 rows 與球隊局數下，排行榜各比率榜的上榜者 = 該項 PR 值不為空者。"""
    from src.export.leaders import RATE_BOARDS, compute_leaders
    from src.export.player_advanced import compute_advanced

    team_sets = {1: 10, 2: 20, 3: 0}
    rows = []
    for pid, team in enumerate([1, 1, 2, 2, 3, 3, 1, 2], start=1):
        for variant in range(4):
            rows.append(prow(
                pid * 10 + variant, team_id=team, position="S" if variant % 2 else "OH",
                attack_total=[9, 19, 20, 39, 40][(pid + variant) % 5], attack_points=5,
                serve_total=[12, 13, 25, 26][(pid + variant) % 4], serve_points=2,
                receive_total=[12, 13, 25, 26][(pid * 3 + variant) % 4], receive_excellent=3,
                dig_total=[9, 10, 19, 20][(pid + 2 * variant) % 4], dig_excellent=4,
                set_total=[49, 50, 99, 100][(pid + variant) % 4], set_excellent=7,
                block_points=2, total_points=9))
    rows.append(prow(900, team_id=1, position="OH", set_total=100, set_excellent=7))
    rows.append(prow(901, team_id=1, position="S", set_total=100, set_excellent=7))
    named = [{**r, "name": f"球員{r['player_id']}", "team_short": f"隊{r['team_id']}"} for r in rows]
    results = compute_advanced(rows, team_sets)
    pr = {r["player_id"]: r for r in results}
    boards = {b["key"]: {r["player_id"] for r in b["rows"]} for b in compute_leaders(named, team_sets, RATE_BOARDS)}

    for spec in RATE_BOARDS:
        if spec.key in ("points_per_game", "blocks_per_set"):
            continue
        in_pool = {pid for pid, r in pr.items() if r["pr"][spec.key] is not None}
        assert boards[spec.key] == in_pool and in_pool, spec.key
    assert 900 not in boards["set_excellent_rate"] and 901 in boards["set_excellent_rate"]

    ppg_pool = {r["player_id"] for r in results if "points_per_game" in r["eligible"] and r["position"] is not None}
    assert boards["points_per_game"] == ppg_pool and ppg_pool
    bps_pool = {pid for pid, r in pr.items() if r["pr"]["blocks_per_set"] is not None}
    assert boards["blocks_per_set"] == bps_pool and bps_pool
    assert any(r["team_id"] == 3 for r in rows) and not any(pr[p]["team_sets"] == 0 for p in bps_pool)


def test_team_sets_cover_whole_season_with_golden_set_counted_as_one(export):
    """球員頁分母跟著分子走全季：例行賽 3 局 + 季後賽 4 局 + 黃金決勝局 1 局 = 8，每局攔網 = 全季攔網 / 8。"""
    def build(conn):
        _match(conn, 1, date="2025-11-01", sets=3)
        _match(conn, 2, date="2025-11-08", sets=4)
        conn.execute("UPDATE matches SET round_name = '季後賽 準決賽' WHERE game_id = 2")
        seed_player(conn, 1, "甲")
        seed_registration(conn, 1, 1, team_id=1, position="OH")
        seed_stat(conn, 1, (3, 20, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 12), match_date="2025-11-01")
        seed_stat(conn, 1, (4, 20, 10, 3, 0, 0, 0, 0, 0, 0, 0, 0, 13), match_date="2025-11-08")
        seed_stat(conn, 1, (1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2), match_date="2025-11-08", golden=1)

    [entry] = export(build)["players"]

    assert entry["team_sets"] == 8
    assert entry["ratios"]["blocks_per_set"] == 0.75  # 全季攔網 6 / 8
