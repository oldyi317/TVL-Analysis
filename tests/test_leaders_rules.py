from src.export.leaders import compute_leaders


def row(player_id, name, total_points, position="S"):
    return {"player_id": player_id, "name": name, "team_id": 1, "team_short": "甲隊", "position": position,
            "total_points": total_points, "attack_points": total_points, "block_points": total_points,
            "serve_points": total_points, "receive_excellent": total_points,
            "dig_excellent": total_points, "set_excellent": total_points,
            "sets_played": None, "games": 1, "attack_total": 0, "serve_total": 0,
            "receive_total": 0, "dig_total": 0, "set_total": 0}


def test_tied_values_share_rank_and_next_rank_skips():
    rows = [row(1, "甲", 10), row(2, "乙", 10), row(3, "丙", 8)]
    board = compute_leaders(rows)[0]
    assert [r["rank"] for r in board["rows"]] == [1, 1, 3]


def test_ties_are_ordered_by_name():
    rows = [row(1, "丙", 10), row(2, "乙", 10), row(3, "甲", 10)]
    board = compute_leaders(rows)[0]
    assert [r["name"] for r in board["rows"]] == ["丙", "乙", "甲"]


def test_only_rank_one_gets_title_and_rate_fields_are_null():
    rows = [row(1, "甲", 10), row(2, "乙", 10), row(3, "丙", 8)]
    board = compute_leaders(rows)[0]
    assert [r["title"] for r in board["rows"]] == ["得分王", "得分王", None]
    assert board["rows"][0]["numerator"] is None and board["rows"][0]["denominator"] is None
    assert (board["key"], board["label"], board["kind"], board["threshold"]) == ("total_points", "總得分", "count", None)
    assert board["rows"][0]["value"] == 10


def test_zero_point_players_are_included():
    board = compute_leaders([row(1, "甲", 0)])[0]
    assert [(r["rank"], r["value"]) for r in board["rows"]] == [(1, 0)]


def test_seven_count_boards_and_only_four_champion_titles():
    boards = compute_leaders([row(1, "甲", 10), row(2, "乙", 10), row(3, "丙", 8)])
    assert [b["label"] for b in boards if b["kind"] == "count"] == [
        "總得分", "攻擊得分", "攔網得分", "發球得分", "接發球到位", "防守到位", "舉球到位"]
    for board, title in zip(boards, ["得分王", "攻擊王", "攔網王", "發球王", None, None, None]):
        assert [r["title"] for r in board["rows"]] == [title, title, None]


def rate_row(player_id=1, **changes):
    return {**row(player_id, f"球員{player_id}", 20), "sets_played": 10, "games": 10,
            "attack_total": 40, "serve_total": 40, "receive_total": 40,
            "dig_total": 40, "set_total": 40, **changes}


def test_rate_boards_apply_both_thresholds_and_order_ties_by_sample_size():
    boards = {b["key"]: b for b in compute_leaders([
        rate_row(1, sets_played=4),
        rate_row(2, attack_total=9),
        rate_row(3, attack_points=10, attack_total=20),
        rate_row(4, attack_points=20, attack_total=40),
        rate_row(5, attack_points=10, attack_total=40),
    ])}
    attack = boards["attack_success_rate"]
    assert attack["threshold"] == "門檻：出賽 5 局、攻擊 10 次"
    assert [(r["player_id"], r["rank"], r["value"], r["numerator"], r["denominator"])
            for r in attack["rows"]] == [(4, 1, 50, 20, 40), (3, 1, 50, 10, 20), (5, 3, 25, 10, 40)]
    assert all(r["title"] is None for r in attack["rows"])


def test_unknown_sets_use_games_and_omit_blocks_board():
    boards = {b["key"]: b for b in compute_leaders([
        rate_row(1, sets_played=None, games=1),
        rate_row(2, sets_played=None, games=2),
    ])}
    assert "blocks_per_set" not in boards
    assert len(boards) == 13
    assert boards["attack_success_rate"]["threshold"] == "門檻：出賽 2 場、攻擊 10 次"
    assert [r["player_id"] for r in boards["attack_success_rate"]["rows"]] == [2]
    assert boards["points_per_game"]["rows"] == []
    assert boards["points_per_game"]["threshold"] == "門檻：出賽 2 場、出賽 10 場"


def test_all_seven_rates_use_their_own_denominators_and_match_player_page():
    from src.export.player_advanced import compute_ratios

    player = rate_row(1)
    boards = [b for b in compute_leaders([player]) if b["kind"] == "rate"]
    assert [b["label"] for b in boards] == [
        "攻擊成功率", "發球得分率", "接發球到位率", "防守到位率", "舉球到位率", "場均得分", "每局攔網"]
    assert [b["threshold"] for b in boards] == [
        "門檻：出賽 5 局、攻擊 10 次", "門檻：出賽 5 局、發球 10 次", "門檻：出賽 5 局、接發球 10 次",
        "門檻：出賽 5 局、防守 10 次", "門檻：出賽 5 局、舉球 10 次", "門檻：出賽 5 局、出賽 10 場",
        "門檻：出賽 5 局、出賽 10 局"]
    assert [b["rows"][0]["value"] for b in boards] == [50, 50, 50, 50, 50, 2, 2]
    for board, denominator in zip(boards, ["attack_total", "serve_total", "receive_total", "dig_total", "set_total", "games", "sets_played"]):
        assert board["rows"][0]["value"] == compute_ratios(player)[board["key"]]
        below = {b["key"]: b for b in compute_leaders([rate_row(1, **{denominator: 9})])}
        assert below[board["key"]]["rows"] == []


def test_rate_values_match_export_precision_before_assigning_tied_ranks():
    attack = next(b for b in compute_leaders([
        rate_row(1, attack_points=1, attack_total=300),
        rate_row(2, attack_points=1, attack_total=301),
    ]) if b["key"] == "attack_success_rate")
    assert [(r["player_id"], r["value"], r["rank"]) for r in attack["rows"]] == [
        (2, 0.33, 1), (1, 0.33, 1)]


def test_mixed_sets_describe_the_unknown_sets_fallback():
    attack = next(b for b in compute_leaders([
        rate_row(1, sets_played=5), rate_row(2, sets_played=None, games=2),
    ]) if b["key"] == "attack_success_rate")
    assert attack["threshold"] == "門檻：出賽 5 局（無局數時 2 場）、攻擊 10 次"
    assert len(attack["rows"]) == 2


def test_setting_boards_only_include_setters_while_other_boards_include_all_positions():
    players = [
        rate_row(1, name="舉球甲", position="S", set_excellent=30, set_total=50),
        rate_row(2, name="舉球零", position="S", set_excellent=0, set_total=10),
        rate_row(3, name="主攻", position="OH", set_excellent=40, set_total=40),
        rate_row(4, name="攔中", position="MB", set_excellent=10, set_total=10),
        rate_row(5, name="副攻", position="OP", set_excellent=10, set_total=10),
        rate_row(6, name="自由", position="L", set_excellent=10, set_total=10),
        rate_row(7, name="無位置", position=None, set_excellent=10, set_total=10),
    ]
    boards = {b["key"]: b for b in compute_leaders(players)}
    assert [(r["player_id"], r["rank"], r["value"]) for r in boards["set_excellent"]["rows"]] == [
        (1, 1, 30), (2, 2, 0),
    ]
    assert [(r["player_id"], r["rank"], r["value"], r["numerator"], r["denominator"])
            for r in boards["set_excellent_rate"]["rows"]] == [
        (1, 1, 60.0, 30, 50), (2, 2, 0.0, 0, 10),
    ]
    assert boards["set_excellent_rate"]["threshold"] == "門檻：出賽 5 局、舉球 10 次"
    for key, board in boards.items():
        if key not in ("set_excellent", "set_excellent_rate"):
            assert {r["player_id"] for r in board["rows"]} == {1, 2, 3, 4, 5, 6, 7}

