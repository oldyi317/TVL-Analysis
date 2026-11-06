from src.export.leaders import (
    RATE_BOARDS, compute_leaders, compute_volume_rates, count_team_sets, regular_totals, threshold_text,
)
from src.export.player_seasons import STAT_FIELDS


def row(player_id, name, total_points, position="S"):
    return {"player_id": player_id, "name": name, "team_id": 1, "team_short": "甲隊", "position": position,
            "total_points": total_points, "attack_points": total_points, "block_points": total_points,
            "serve_points": total_points, "receive_excellent": total_points,
            "dig_excellent": total_points, "set_excellent": total_points,
            "sets_played": None, "games": 1, "attack_total": 0, "serve_total": 0,
            "receive_total": 0, "dig_total": 0, "set_total": 0}


def test_tied_values_share_rank_and_next_rank_skips():
    rows = [row(1, "甲", 10), row(2, "乙", 10), row(3, "丙", 8)]
    board = compute_leaders(rows, {1: 10})[0]
    assert [r["rank"] for r in board["rows"]] == [1, 1, 3]


def test_ties_are_ordered_by_name():
    rows = [row(1, "丙", 10), row(2, "乙", 10), row(3, "甲", 10)]
    board = compute_leaders(rows, {1: 10})[0]
    assert [r["name"] for r in board["rows"]] == ["丙", "乙", "甲"]


def test_only_rank_one_gets_title_and_rate_fields_are_null():
    rows = [row(1, "甲", 10), row(2, "乙", 10), row(3, "丙", 8)]
    board = compute_leaders(rows, {1: 10})[0]
    assert [r["title"] for r in board["rows"]] == ["得分王", "得分王", None]
    assert board["rows"][0]["numerator"] is None and board["rows"][0]["denominator"] is None
    assert (board["key"], board["label"], board["kind"], board["threshold"]) == ("total_points", "總得分", "count", None)
    assert board["rows"][0]["value"] == 10


def test_zero_point_players_are_included():
    board = compute_leaders([row(1, "甲", 0)], {1: 10})[0]
    assert [(r["rank"], r["value"]) for r in board["rows"]] == [(1, 0)]


def test_seven_count_boards_and_only_four_champion_titles():
    boards = compute_leaders([row(1, "甲", 10), row(2, "乙", 10), row(3, "丙", 8)], {1: 10})
    assert [b["label"] for b in boards if b["kind"] == "count"] == [
        "總得分", "攻擊得分", "攔網得分", "發球得分", "接發球到位", "防守到位", "舉球到位"]
    for board, title in zip(boards, ["得分王", "攻擊王", "攔網王", "發球王", None, None, None]):
        assert [r["title"] for r in board["rows"]] == [title, title, None]


def rate_row(player_id=1, **changes):
    return {**row(player_id, f"球員{player_id}", 20), "sets_played": 10, "games": 10, **changes}


def loaded(player_id=1, **changes):
    totals = {"attack_total": 40, "serve_total": 40, "receive_total": 40, "dig_total": 40, "set_total": 50}
    return rate_row(player_id, **{**totals, **changes})


def rate_board(rows, key, team_sets=None, apply_makeup=False):
    boards = compute_leaders(rows, {1: 10} if team_sets is None else team_sets, apply_makeup=apply_makeup)
    return next(b for b in boards if b["key"] == key)


def test_rate_boards_apply_threshold_and_order_ties_by_sample_size():
    attack = rate_board([
        rate_row(1, attack_points=5, attack_total=19),
        rate_row(2, attack_points=10, attack_total=20),
        rate_row(3, attack_points=20, attack_total=40),
        rate_row(4, attack_points=10, attack_total=40),
    ], "attack_success_rate")
    assert attack["threshold"] == "門檻：攻擊次數達所屬球隊已打局數 × 2（目前 20 次）"
    assert [(r["player_id"], r["rank"], r["value"], r["numerator"], r["denominator"])
            for r in attack["rows"]] == [(3, 1, 50, 20, 40), (2, 1, 50, 10, 20), (4, 3, 25, 10, 40)]
    assert all(r["title"] is None for r in attack["rows"])


def test_threshold_floats_with_team_sets():
    rows = [rate_row(1, attack_total=25, attack_points=10), rate_row(2, attack_total=24, attack_points=20)]
    attack = lambda sets: next(b for b in compute_leaders(rows, {1: sets}) if b["key"] == "attack_success_rate")
    assert [r["player_id"] for r in attack(12)["rows"]] == [2, 1]
    assert attack(13)["rows"] == []


def test_threshold_uses_floor_below_ten():
    rows = [rate_row(1, attack_total=10, attack_points=5), rate_row(2, attack_total=9, attack_points=4)]
    assert [r["player_id"] for r in rate_board(rows, "attack_success_rate", {1: 3})["rows"]] == [1]


def test_missing_team_uses_floor():
    rows = [rate_row(1, attack_total=10, attack_points=5)]
    assert rate_board(rows, "attack_success_rate", {})["rows"][0]["player_id"] == 1


def test_each_rate_board_uses_its_own_coefficient():
    thresholds = {
        "attack_success_rate": ("attack_total", 40), "serve_point_rate": ("serve_total", 26),
        "receive_excellent_rate": ("receive_total", 26), "dig_excellent_rate": ("dig_total", 20),
        "set_excellent_rate": ("set_total", 100), "points_per_game": ("attack_total", 40),
        "blocks_per_set": ("attack_total", 40),
    }
    for key, (field, needed) in thresholds.items():
        board = rate_board([rate_row(1, **{field: needed}), rate_row(2, **{field: needed - 1})], key, {1: 20})
        assert [r["player_id"] for r in board["rows"]] == [1], key
    both = rate_board([rate_row(1, attack_total=20, serve_total=20), rate_row(2, attack_total=20, serve_total=19)],
                      "points_per_game", {1: 20})
    assert [r["player_id"] for r in both["rows"]] == [1]


def test_all_seven_rates_use_their_own_denominators_and_match_player_page():
    from src.export.player_advanced import compute_ratios

    player = loaded(1)
    boards = [b for b in compute_leaders([player], {1: 10}) if b["kind"] == "rate"]
    assert [b["label"] for b in boards] == [
        "攻擊成功率", "發球得分率", "接發球到位率", "防守到位率", "舉球到位率", "場均得分", "每局攔網"]
    assert [b["threshold"] for b in boards] == [
        "門檻：攻擊次數達所屬球隊已打局數 × 2（目前 20 次）",
        "門檻：發球次數達所屬球隊已打局數 × 1.3（目前 13 次）",
        "門檻：接發球次數達所屬球隊已打局數 × 1.3（目前 13 次）",
        "門檻：防守次數達所屬球隊已打局數 × 1（目前 10 次）",
        "門檻：舉球次數達所屬球隊已打局數 × 5（目前 50 次）",
        "門檻：攻擊＋發球次數達所屬球隊已打局數 × 2（目前 20 次）",
        "門檻：攻擊＋發球次數達所屬球隊已打局數 × 2（目前 20 次）"]
    assert [b["rows"][0]["value"] for b in boards] == [50, 50, 50, 50, 40, 2, 2]
    assert [b["rows"][0]["denominator"] for b in boards] == [40, 40, 40, 40, 50, 10, 10]
    for board in boards[:6]:
        assert board["rows"][0]["value"] == compute_ratios(player)[board["key"]]


def test_blocks_per_set_uses_team_sets_and_is_always_output():
    board = rate_board([rate_row(1, block_points=6, sets_played=None, attack_total=40)], "blocks_per_set", {1: 20})
    assert board["rows"][0]["value"] == 0.3 and board["rows"][0]["denominator"] == 20


def test_blocks_per_set_without_team_sets_is_empty():
    board = rate_board([rate_row(1, block_points=6, attack_total=40)], "blocks_per_set", {})
    assert board["rows"] == []


def test_blocks_per_set_ties_order_by_block_points():
    rows = [rate_row(1, team_id=1, block_points=4, attack_total=40), rate_row(2, team_id=2, block_points=8, attack_total=80)]
    board = rate_board(rows, "blocks_per_set", {1: 20, 2: 40})
    assert [r["player_id"] for r in board["rows"]] == [2, 1] and [r["rank"] for r in board["rows"]] == [1, 1]


def test_threshold_text_formats():
    spec = next(s for s in RATE_BOARDS if s.key == "attack_success_rate")
    assert threshold_text(spec, {1: 10, 2: 14}) == "門檻：攻擊次數達所屬球隊已打局數 × 2（目前 20 到 28 次）"
    assert threshold_text(spec, {1: 12, 2: 12}) == "門檻：攻擊次數達所屬球隊已打局數 × 2（目前 24 次）"
    assert threshold_text(spec, {1: 3, 2: 6}) == "門檻：攻擊次數達所屬球隊已打局數 × 2，季初至少 10 次（目前 10 到 12 次）"
    assert threshold_text(spec, {}) == "門檻：攻擊次數達所屬球隊已打局數 × 2，季初至少 10 次（目前 10 次）"
    serve = next(s for s in RATE_BOARDS if s.key == "serve_point_rate")
    assert threshold_text(serve, {1: 20}) == "門檻：發球次數達所屬球隊已打局數 × 1.3（目前 26 次）"
    ppg = next(s for s in RATE_BOARDS if s.key == "points_per_game")
    assert threshold_text(ppg, {1: 20}) == "門檻：攻擊＋發球次數達所屬球隊已打局數 × 2（目前 40 次）"


def test_rate_values_match_export_precision_before_assigning_tied_ranks():
    attack = rate_board([
        rate_row(1, attack_points=1, attack_total=300),
        rate_row(2, attack_points=1, attack_total=301),
    ], "attack_success_rate")
    assert [(r["player_id"], r["value"], r["rank"]) for r in attack["rows"]] == [
        (2, 0.33, 1), (1, 0.33, 1)]


def test_setting_boards_only_include_setters_while_other_boards_include_all_positions():
    players = [
        loaded(1, name="舉球甲", position="S", set_excellent=30, set_total=50),
        loaded(2, name="舉球零", position="S", set_excellent=0, set_total=50),
        loaded(3, name="主攻", position="OH", set_excellent=40),
        loaded(4, name="攔中", position="MB", set_excellent=10),
        loaded(5, name="副攻", position="OP", set_excellent=10),
        loaded(6, name="自由", position="L", set_excellent=10),
        loaded(7, name="無位置", position=None, set_excellent=10),
    ]
    boards = {b["key"]: b for b in compute_leaders(players, {1: 10})}
    assert [(r["player_id"], r["rank"], r["value"]) for r in boards["set_excellent"]["rows"]] == [
        (1, 1, 30), (2, 2, 0),
    ]
    assert [(r["player_id"], r["rank"], r["value"], r["numerator"], r["denominator"])
            for r in boards["set_excellent_rate"]["rows"]] == [
        (1, 1, 60.0, 30, 50), (2, 2, 0.0, 0, 50),
    ]
    assert boards["set_excellent_rate"]["threshold"] == "門檻：舉球次數達所屬球隊已打局數 × 5（目前 50 次）"
    for key, board in boards.items():
        if key not in ("set_excellent", "set_excellent_rate"):
            assert {r["player_id"] for r in board["rows"]} == {1, 2, 3, 4, 5, 6, 7}


def test_volume_rates_use_the_same_threshold_as_rate_boards():
    players = [
        loaded(1, attack_points=30),
        rate_row(2, attack_total=19, attack_points=5),
        rate_row(3, attack_total=9, receive_total=9, attack_points=5, receive_excellent=5),
        rate_row(4, dig_total=10),
    ]
    team_sets = {1: 10}
    boards = {b["key"]: b for b in compute_leaders(players, team_sets)}
    volume_rates = compute_volume_rates(players, team_sets)
    assert [v["key"] for v in volume_rates] == [
        "attack_success_rate", "receive_excellent_rate", "dig_excellent_rate", "set_excellent_rate"]
    for v in volume_rates:
        assert {p["player_id"] for p in v["points"]} == {r["player_id"] for r in boards[v["key"]]["rows"]}
    assert {p["player_id"] for p in volume_rates[0]["points"]} == {1}
    assert {p["player_id"] for p in volume_rates[2]["points"]} == {1, 4}
    assert next(p for p in volume_rates[0]["points"] if p["player_id"] == 1) == {
        "player_id": 1, "name": "球員1", "team_id": 1, "team_short": "甲隊", "count": 40, "rate": 75.0}


def test_volume_rates_for_setting_only_list_setters():
    players = [
        rate_row(1, position="S", set_excellent=30, set_total=50, attack_total=20),
        rate_row(2, position="OH", set_excellent=40, set_total=50, attack_total=20),
        rate_row(3, position=None, set_excellent=10, set_total=50, attack_total=20),
    ]
    setting = compute_volume_rates(players, {1: 10})[3]
    assert [p["player_id"] for p in setting["points"]] == [1]
    assert {p["player_id"] for p in compute_volume_rates(players, {1: 10})[0]["points"]} == {1, 2, 3}


def test_volume_rate_medians_are_precomputed_for_both_axes():
    players = [
        rate_row(1, attack_points=10, attack_total=20),
        rate_row(2, attack_points=30, attack_total=40),
        rate_row(3, attack_points=10, attack_total=50),
        rate_row(4, attack_points=1, attack_total=3),
    ]
    attack = compute_volume_rates(players, {1: 10})[0]
    # 次數 20、40、50 → 中位數 40；率 50、75、20 → 中位數 50；第 4 人未達門檻不計
    assert (attack["median_count"], attack["median_rate"]) == (40, 50.0)
    even = compute_volume_rates(players[:2], {1: 10})[0]
    assert (even["median_count"], even["median_rate"]) == (30, 62.5)


def test_volume_rate_medians_are_null_without_points():
    empty = compute_volume_rates([rate_row(1)], {1: 10})
    assert all(v["points"] == [] and v["median_count"] is None and v["median_rate"] is None for v in empty)


def entry(player_id, team_id, **stats):
    base = {f: (None if f == "sets_played" else 0) for f in STAT_FIELDS}
    return {"player_id": player_id, "team_id": team_id, "jersey_number": None, "position": None, **base, **stats}


def game(game_id, home_rows, away_rows, home_team_id=1, away_team_id=2, sets=4, round_name="例行賽 Week 1"):
    return {"game_id": game_id, "date": "2025-11-01", "home": home_rows, "away": away_rows,
            "golden_set": None, "home_team_id": home_team_id, "away_team_id": away_team_id,
            "round_name": round_name, "sets": sets}


def test_regular_totals_sum_stat_fields_and_count_games():
    g1 = game(1, [entry(7, 1, attack_total=10, attack_points=4)], [])
    g2 = game(2, [], [entry(7, 1, attack_total=5, attack_points=1)])
    totals = regular_totals([g1, g2])
    assert totals[7]["games"] == 2 and totals[7]["attack_total"] == 15 and totals[7]["attack_points"] == 5
    assert totals[7]["sets_played"] is None


def test_regular_totals_sum_sets_played_when_all_present():
    g1 = game(1, [entry(7, 1, sets_played=3)], [])
    g2 = game(2, [entry(7, 1, sets_played=2)], [])
    assert regular_totals([g1, g2])[7]["sets_played"] == 5


def test_team_sets_sum_both_sides():
    assert count_team_sets([game(1, [], [], 1, 2, sets=4), game(2, [], [], 2, 3, sets=5)]) == {1: 4, 2: 9, 3: 5}


def attack_board(rows, team_sets=None):
    return rate_board(rows, "attack_success_rate", {1: 20} if team_sets is None else team_sets, apply_makeup=True)


def test_makeup_is_off_by_default_and_when_season_incomplete():
    rows = [rate_row(1, attack_total=40, attack_points=20), rate_row(2, attack_total=30, attack_points=25)]
    off = rate_board(rows, "attack_success_rate", {1: 20})
    assert [(r["player_id"], r["makeup"]) for r in off["rows"]] == [(1, False)]
    on = rate_board(rows, "attack_success_rate", {1: 20}, apply_makeup=True)
    assert [(r["player_id"], r["makeup"]) for r in on["rows"]] == [(2, True), (1, False)]


def test_makeup_lists_short_player_first_with_actual_rate():
    rows = [rate_row(1, attack_total=40, attack_points=20), rate_row(2, attack_total=30, attack_points=25)]
    board = attack_board(rows)
    assert [(r["player_id"], r["rank"], r["makeup"]) for r in board["rows"]] == [(2, 1, True), (1, 2, False)]
    assert board["rows"][0]["value"] == 83.33 and board["rows"][0]["denominator"] == 30
    assert board["rows"][0]["numerator"] == 25


def test_makeup_requires_strictly_higher():
    rows = [rate_row(1, attack_total=40, attack_points=20), rate_row(2, attack_total=30, attack_points=20)]
    assert [(r["player_id"], r["makeup"]) for r in attack_board(rows)["rows"]] == [(1, False)]


def test_makeup_ignores_zero_attempts():
    rows = [rate_row(1, attack_total=40, attack_points=20), rate_row(2, attack_total=0, attack_points=0)]
    assert [r["player_id"] for r in attack_board(rows)["rows"]] == [1]


def test_makeup_skips_when_nobody_qualifies():
    rows = [rate_row(1, attack_total=30, attack_points=29)]
    assert attack_board(rows)["rows"] == []


def test_makeup_takes_only_the_best_candidate_and_demotes_tied_leaders():
    rows = [
        rate_row(1, attack_total=40, attack_points=16), rate_row(2, attack_total=40, attack_points=16),
        rate_row(3, attack_total=30, attack_points=28), rate_row(4, attack_total=20, attack_points=19),
        rate_row(5, attack_total=40, attack_points=10),
    ]
    got = [(r["player_id"], r["rank"], r["makeup"]) for r in attack_board(rows)["rows"]]
    assert got == [(3, 1, True), (1, 2, False), (2, 2, False), (5, 4, False)]


def test_makeup_stays_first_when_rounded_value_equals_leader():
    rows = [rate_row(1, attack_total=1_000_000, attack_points=500_039),
            rate_row(2, attack_total=99_999, attack_points=50_004)]
    board = attack_board(rows, {1: 50_000})
    assert [r["value"] for r in board["rows"]] == [50.0, 50.0]
    assert [(r["player_id"], r["rank"], r["makeup"]) for r in board["rows"]] == [(2, 1, True), (1, 2, False)]


def test_makeup_tie_prefers_more_attempts_then_name():
    rows = [
        rate_row(1, attack_total=40, attack_points=12),
        rate_row(2, attack_total=20, attack_points=16), rate_row(3, attack_total=30, attack_points=16),
    ]
    got = [(r["player_id"], r["makeup"]) for r in attack_board(rows)["rows"]]
    assert got == [(3, True), (1, False)]


def test_makeup_uses_each_players_team_threshold():
    rows = [rate_row(1, attack_total=40, attack_points=20), {**rate_row(2, attack_total=30, attack_points=25), "team_id": 2}]
    assert [(r["player_id"], r["makeup"]) for r in attack_board(rows, {1: 20, 2: 15})["rows"]] == [(2, False), (1, False)]
    assert [(r["player_id"], r["makeup"]) for r in attack_board(rows, {1: 20, 2: 20})["rows"]] == [(2, True), (1, False)]


def test_makeup_not_applied_to_points_per_game_and_blocks_per_set():
    rows = [rate_row(1, attack_total=40, block_points=20), rate_row(2, attack_total=30, block_points=29, total_points=29)]
    for key in ("points_per_game", "blocks_per_set"):
        assert [r["player_id"] for r in rate_board(rows, key, {1: 20})["rows"]] == [1]


def test_count_boards_have_makeup_false():
    board = compute_leaders([row(1, "甲", 10)], {1: 10})[0]
    assert board["rows"][0]["makeup"] is False


def test_volume_rates_exclude_makeup_rows():
    rows = [rate_row(1, attack_total=40, attack_points=20), rate_row(2, attack_total=30, attack_points=25)]
    attack = next(v for v in compute_volume_rates(rows, {1: 20}) if v["key"] == "attack_success_rate")
    assert [p["player_id"] for p in attack["points"]] == [1]
