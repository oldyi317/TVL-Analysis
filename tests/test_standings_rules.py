from src.export.standings import compute_standings

NO_SCORES = (None,) * 10


def game(home, away, home_sets, away_sets, home_pts=(), away_pts=()):
    home_scores = tuple(home_pts) + (None,) * (5 - len(home_pts))
    away_scores = tuple(away_pts) + (None,) * (5 - len(away_pts))
    return (home, away, home_sets, away_sets, *home_scores, *away_scores)


def by_team(rows):
    return {r["team_id"]: r for r in rows}


def order(rows):
    return [r["team_id"] for r in rows]


def test_points_follow_set_record_3_2_1_0():
    results = [game(1, 2, 3, 0), game(1, 3, 3, 2), game(1, 4, 2, 3), game(1, 5, 1, 3)]
    rows = by_team(compute_standings([1, 2, 3, 4, 5], results))
    assert [rows[t]["points"] for t in (1, 2, 3, 4, 5)] == [3 + 2 + 1 + 0, 0, 1, 2, 3]
    assert rows[1]["set_record"] == {"3-0": 1, "3-1": 0, "3-2": 1, "2-3": 1, "1-3": 1, "0-3": 0}


def test_wins_rank_before_points():
    results = [game(1, 9, 3, 2), game(1, 9, 3, 2), game(2, 9, 3, 0), game(9, 2, 3, 2)]
    rows = compute_standings([1, 2, 9], results)
    assert order(rows)[:2] == [1, 2]
    by = by_team(rows)
    assert (by[1]["wins"], by[1]["points"]) == (2, 4)
    assert (by[2]["wins"], by[2]["points"]) == (1, 4)


def test_set_ratio_none_ranks_as_infinity():
    results = [game(1, 3, 3, 0), game(2, 3, 3, 1)]
    rows = compute_standings([1, 2, 3], results)
    assert by_team(rows)[1]["set_ratio"] is None
    assert order(rows) == [1, 2, 3]


def test_point_ratio_breaks_tie_after_set_ratio():
    results = [game(1, 3, 3, 0, (25, 25, 25), (20, 20, 20)), game(2, 4, 3, 0, (25, 25, 25), (10, 10, 10))]
    rows = compute_standings([1, 2, 3, 4], results)
    assert order(rows)[:2] == [2, 1]


def test_team_id_is_last_tie_break():
    rows = compute_standings([7, 3, 5], [game(5, 3, 3, 0), game(3, 5, 3, 0)])
    assert order(rows) == [3, 5, 7]


def test_teams_without_results_have_empty_row_and_rank_last():
    rows = compute_standings([1, 2, 3], [game(1, 2, 3, 0)])
    assert order(rows) == [1, 2, 3]
    row = by_team(rows)[3]
    assert (row["win_pct"], row["set_ratio"], row["point_ratio"], row["last5"]) == (None, None, None, [])
    assert [r["rank"] for r in rows] == [1, 2, 3]


def test_last5_is_oldest_to_newest_capped_at_five():
    results = [game(1, 2, 3, 0)] * 2 + [game(2, 1, 3, 0)] * 4
    row = by_team(compute_standings([1, 2], results))[1]
    assert row["last5"] == ["W", "L", "L", "L", "L"]


def test_none_scores_sum_as_zero_and_none_team_is_skipped():
    results = [game(1, 2, 3, 0, (25, 25, None), (10, None, 10)), game(None, 2, 3, 0)]
    row = by_team(compute_standings([1, 2], results))[1]
    assert (row["points_for"], row["points_against"]) == (50, 20)
    assert row["wins"] == 1
