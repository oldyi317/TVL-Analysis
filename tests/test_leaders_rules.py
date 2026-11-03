from src.export.leaders import compute_leaders


def row(player_id, name, total_points):
    return {"player_id": player_id, "name": name, "team_id": 1, "team_short": "甲隊", "position": "OH",
            "total_points": total_points, "attack_points": total_points, "block_points": total_points,
            "serve_points": total_points, "receive_excellent": total_points,
            "dig_excellent": total_points, "set_excellent": total_points}


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
    assert [b["label"] for b in boards] == [
        "總得分", "攻擊得分", "攔網得分", "發球得分", "接發球到位", "防守到位", "舉球到位"]
    for board, title in zip(boards, ["得分王", "攻擊王", "攔網王", "發球王", None, None, None]):
        assert [r["title"] for r in board["rows"]] == [title, title, None]
