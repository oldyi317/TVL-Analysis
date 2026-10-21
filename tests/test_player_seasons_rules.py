from src.export.player_seasons import STAT_FIELDS, Registration, SeasonRoster, SeasonTotals, compute_player_seasons


def reg(player, cup, team, jersey, position, week, reg_id):
    return Registration(player, cup, team, jersey, position, week, reg_id)


def season(registrations, season_rosters=(), totals=None, player=1, cup=1):
    result = compute_player_seasons(totals or {}, registrations, season_rosters)
    return next(e for e in result[player] if e["cup_id"] == cup)


def test_position_is_majority_vote():
    regs = [reg(1, 1, 10, 5, "OH", "2026-01-01", 1), reg(1, 1, 10, 5, "MB", "2026-01-08", 2),
            reg(1, 1, 10, 5, "MB", "2026-01-15", 3)]
    assert season(regs)["position"] == "MB"


def test_position_tie_takes_most_recent_regardless_of_input_order():
    regs = [reg(1, 1, 10, 5, "MB", "2026-01-15", 3), reg(1, 1, 10, 5, "OH", "2026-01-01", 1),
            reg(1, 1, 10, 5, "OH", "2026-01-08", 2), reg(1, 1, 10, 5, "MB", "2026-01-02", 4)]
    assert season(regs)["position"] == "MB"


def test_null_week_rows_rank_after_dated_rows():
    regs = [reg(1, 1, 10, 9, "OH", None, 99), reg(1, 1, 11, 7, "MB", "2026-01-01", 1)]
    entry = season(regs)
    assert entry["position"] == "MB"
    assert (entry["team_id"], entry["jersey_number"]) == (11, 7)


def test_same_week_ties_break_by_larger_registration_id():
    regs = [reg(1, 1, 10, 1, "OH", "2026-01-01", 1), reg(1, 1, 11, 2, "MB", "2026-01-01", 2)]
    entry = season(regs)
    assert (entry["team_id"], entry["jersey_number"], entry["position"]) == (11, 2, "MB")


def test_blank_positions_are_not_counted():
    regs = [reg(1, 1, 10, 5, None, "2026-01-15", 3), reg(1, 1, 10, 5, None, "2026-01-08", 2),
            reg(1, 1, 10, 5, "OH", "2026-01-01", 1)]
    assert season(regs)["position"] == "OH"
    assert season([reg(1, 1, 10, 5, None, "2026-01-01", 1)])["position"] is None


def test_latest_registration_gives_team_and_jersey():
    regs = [reg(1, 1, 10, 5, "OH", "2026-01-01", 1), reg(1, 1, 11, None, "OH", "2026-01-08", 2)]
    entry = season(regs)
    assert (entry["team_id"], entry["jersey_number"]) == (11, None)


def test_totals_fill_games_and_stats_else_zero():
    regs = [reg(1, 1, 10, 5, "OH", "2026-01-01", 1), reg(1, 2, 10, 5, "OH", "2026-02-01", 2)]
    totals = {(1, 1): SeasonTotals(4, tuple(range(1, len(STAT_FIELDS) + 1)))}
    result = compute_player_seasons(totals, regs, [])[1]
    assert [e["cup_id"] for e in result] == [1, 2]
    assert result[0]["games"] == 4 and result[0]["sets_played"] == 1 and result[0]["total_points"] == 13
    assert result[1]["games"] == 0 and result[1]["total_points"] == 0


def test_season_roster_fills_missing_jersey_only_for_same_team():
    regs = [reg(1, 1, 10, None, "OH", "2026-01-01", 1), reg(2, 1, 10, None, "OH", "2026-01-01", 2)]
    rosters = [SeasonRoster(1, 1, 10, 8, "MB"), SeasonRoster(2, 1, 99, 8, "MB")]
    result = compute_player_seasons({}, regs, rosters)
    same, other = result[1][0], result[2][0]
    assert (same["jersey_number"], same["season_roster_position"], same["position"]) == (8, "MB", "OH")
    assert (other["jersey_number"], other["season_roster_position"]) == (None, None)


def test_season_roster_does_not_overwrite_existing_jersey():
    result = compute_player_seasons({}, [reg(1, 1, 10, 5, "OH", "2026-01-01", 1)], [SeasonRoster(1, 1, 10, 8, "MB")])
    assert result[1][0]["jersey_number"] == 5


def test_roster_only_player_gets_zero_stat_row_with_no_position():
    result = compute_player_seasons({}, [reg(1, 2, 10, 5, "OH", "2026-02-01", 1)], [SeasonRoster(1, 1, 10, 8, "MB"), SeasonRoster(3, 1, 11, 2, "OH")])
    assert [e["cup_id"] for e in result[1]] == [1, 2]
    added = result[3][0]
    assert added["games"] == 0 and all(added[f] == 0 for f in STAT_FIELDS)
    assert added["position"] is None and added["season_roster_position"] == "OH"
    assert result[1][0]["position"] is None and result[1][0]["season_roster_position"] == "MB"
