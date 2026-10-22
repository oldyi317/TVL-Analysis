SET_COLS = [f"{side}_set{i}" for side in ("home", "away") for i in range(1, 6)]

UNPLAYED_WHERE = " AND ".join(f"{c} IS NULL" for c in SET_COLS)
HAS_RESULT_WHERE = "home_sets_won IS NOT NULL AND away_sets_won IS NOT NULL"


def has_result(home_sets_won, away_sets_won) -> bool:
    return all(x is not None and x == x for x in (home_sets_won, away_sets_won))
