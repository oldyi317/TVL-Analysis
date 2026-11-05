from src.export.leaders import rank_ordered
from src.export.records import RECORD_LIMIT, compute_records


def row(player_id, name, total_points, cup_id=21):
    return {"player_id": player_id, "name": name, "cup_id": cup_id, "season": "2025-26",
            "team_id": 1, "team_short": "甲隊", "in_progress": False, "match_date": "2025-11-01",
            "opponent": "乙隊", "game_id": None,
            "stats": {"total_points": total_points, "attack_points": total_points,
                      "block_points": total_points, "serve_points": total_points}}


def total_rows(rows):
    return compute_records(rows)[0]["season_records"]


def test_rank_ordered_ties_share_rank_and_next_rank_skips():
    assert rank_ordered([10, 10, 8, 8, 8, 5]) == [1, 1, 3, 3, 3, 6]
    assert rank_ordered([]) == []


def test_four_boards_in_scoring_order_with_labels():
    boards = compute_records([row(1, "甲", 3)])
    assert [(b["key"], b["label"]) for b in boards] == [
        ("total_points", "總得分"), ("attack_points", "攻擊得分"),
        ("block_points", "攔網得分"), ("serve_points", "發球得分")]


def test_ties_share_rank_and_are_ordered_by_name():
    rows = total_rows([row(1, "丙", 10), row(2, "乙", 10), row(3, "甲", 8)])
    assert [(r["rank"], r["name"]) for r in rows] == [(1, "丙"), (1, "乙"), (3, "甲")]


def test_zero_values_are_not_listed():
    assert [r["player_id"] for r in total_rows([row(1, "甲", 0), row(2, "乙", 3)])] == [2]
    assert total_rows([row(1, "甲", 0)]) == []


def test_top_ten_cuts_but_keeps_everyone_tied_at_tenth():
    rows = [row(i, f"球員{i:02d}", 100 - i) for i in range(1, 10)]
    rows += [row(10, "球員10", 50), row(11, "球員11", 50), row(12, "球員12", 50), row(13, "球員13", 40)]
    shown = total_rows(rows)
    assert [r["rank"] for r in shown] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10]
    assert RECORD_LIMIT == 10


def test_same_player_can_hold_two_seasons_and_keeps_row_fields():
    rows = total_rows([row(1, "甲", 10, cup_id=20), row(1, "甲", 10, cup_id=21)])
    assert [(r["rank"], r["cup_id"]) for r in rows] == [(1, 21), (1, 20)]
    assert rows[0]["value"] == 10 and rows[0]["season"] == "2025-26" and rows[0]["games"] == 1
