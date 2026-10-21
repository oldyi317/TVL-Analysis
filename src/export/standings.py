"""戰績的純函式：積分、排序與近五場。

不碰 SQL，匯出器只負責讀賽果再呼叫 compute_standings。
"""
import math
from collections.abc import Iterable, Sequence

SET_RECORD_POINTS = {(3, 0): 3, (3, 1): 3, (3, 2): 2, (2, 3): 1}
SET_RECORD_KEYS = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _new_tally() -> dict:
    return {
        "wins": 0, "losses": 0, "points": 0, "sets_won": 0, "sets_lost": 0,
        "points_for": 0, "points_against": 0, "results": [],
        "set_record": dict.fromkeys(SET_RECORD_KEYS, 0),
    }


def _add_result(tally: dict, own_sets: int, opp_sets: int, own_pts: int, opp_pts: int) -> None:
    won = own_sets > opp_sets
    tally["wins" if won else "losses"] += 1
    tally["points"] += SET_RECORD_POINTS.get((own_sets, opp_sets), 0)
    tally["sets_won"] += own_sets
    tally["sets_lost"] += opp_sets
    tally["points_for"] += own_pts
    tally["points_against"] += opp_pts
    tally["results"].append("W" if won else "L")
    key = f"{own_sets}-{opp_sets}"
    if key in tally["set_record"]:
        tally["set_record"][key] += 1


def _sum_scores(values: Sequence) -> int:
    return sum(v for v in values if v is not None)


def _inf_if_none(value: float | None) -> float:
    return math.inf if value is None else value


def _sort_key(team_id: int, tally: dict):
    return (
        -tally["wins"],
        not tally["results"],
        -tally["points"],
        -_inf_if_none(_ratio(tally["sets_won"], tally["sets_lost"])),
        -_inf_if_none(_ratio(tally["points_for"], tally["points_against"])),
        team_id,
    )


def _standings_row(rank: int, team_id: int, tally: dict) -> dict:
    played = tally["wins"] + tally["losses"]
    return {
        "rank": rank,
        "team_id": team_id,
        "wins": tally["wins"],
        "losses": tally["losses"],
        "win_pct": _rounded(_ratio(tally["wins"], played)),
        "set_record": tally["set_record"],
        "points": tally["points"],
        "sets_won": tally["sets_won"],
        "sets_lost": tally["sets_lost"],
        "set_ratio": _rounded(_ratio(tally["sets_won"], tally["sets_lost"])),
        "points_for": tally["points_for"],
        "points_against": tally["points_against"],
        "point_ratio": _rounded(_ratio(tally["points_for"], tally["points_against"])),
        "last5": tally["results"][-5:],
    }


def compute_standings(team_ids: Iterable[int], results: Iterable[Sequence]) -> list[dict]:
    """一組別一賽季的例行賽戰績，回傳已排名的列。

    results 每筆為 (主隊 id, 客隊 id, 主隊局數, 客隊局數, *主隊五局分數, *客隊五局分數)，
    分數允許 None。呼叫者須依 match_date、game_id 由舊到新排好：last5 取自這個順序。
    主客任一隊為 None 的賽果略過；沒有賽果的隊仍會有一列。
    """
    tallies = {team_id: _new_tally() for team_id in team_ids}
    for home, away, home_sets, away_sets, *scores in results:
        if home is None or away is None:
            continue
        home_pts, away_pts = _sum_scores(scores[:5]), _sum_scores(scores[5:])
        _add_result(tallies[home], home_sets, away_sets, home_pts, away_pts)
        _add_result(tallies[away], away_sets, home_sets, away_pts, home_pts)
    ordered = sorted(tallies, key=lambda t: _sort_key(t, tallies[t]))
    return [_standings_row(i, t, tallies[t]) for i, t in enumerate(ordered, 1)]
