"""球隊指標的純函式：全隊分子總和除以分母總和，再算組別內名次。

輸入為單一賽季單一組別的單場數據（box score 的正規列，不含黃金決勝局）與各場比分（局數、兩隊總分與兩隊編號，不含黃金決勝局）。
"""
import logging
from collections.abc import Iterable, Mapping

from src.export.metrics import columns, measure

logger = logging.getLogger(__name__)

METRICS = (
    "attack_success_rate", "serve_point_rate", "receive_excellent_rate",
    "dig_excellent_rate", "set_excellent_rate", "blocks_per_set",
)
TOTAL_FIELDS = columns(METRICS)


def _game_points(skill: Mapping[str, int], total_points: int, game_id: int, team_id: int) -> dict[str, int]:
    """單場得分來源（排除黃金決勝局）：球隊頁得分來源組成與單場頁賽後對戰比較共用。"""
    attack, block, serve = skill["attack_points"], skill["block_points"], skill["serve_points"]
    opponent_error = total_points - attack - block - serve
    if opponent_error < 0:
        logger.warning("對手失誤得分推算為負值：game_id=%s team_id=%s opponent_error=%s",
                       game_id, team_id, opponent_error)
    return {"attack": attack, "block": block, "serve": serve,
            "opponent_error": opponent_error, "total": total_points}


def _team_totals(games: Iterable[Mapping], scores: Mapping[int, Mapping[str, int]]) -> list[dict]:
    totals: dict[int, dict] = {}
    for game in games:
        for side in ("home", "away"):
            rows = game[side]
            if not rows:
                continue
            team = totals.setdefault(rows[0]["team_id"], {
                "team_id": rows[0]["team_id"], "games": 0, "sets": 0, "total": 0, "opponent_error": 0,
                **dict.fromkeys(TOTAL_FIELDS, 0), "trend": [],
            })
            score = scores[game["game_id"]]
            team["games"] += 1
            team["sets"] += score["sets"]
            skill = {field: sum(row[field] for row in rows) for field in TOTAL_FIELDS}
            for field in TOTAL_FIELDS:
                team[field] += skill[field]
            game_points = _game_points(skill, score[f"{side}_points"], game["game_id"], team["team_id"])
            team["trend"].append({
                "game_id": game["game_id"], "date": game["date"],
                "opponent_id": score["away_team_id" if side == "home" else "home_team_id"],
                "points": game_points,
                "metrics": {key: _metric(skill, key, sets=score["sets"]) for key in METRICS},
            })
            team["total"] += game_points["total"]
            team["opponent_error"] += game_points["opponent_error"]
    for team in totals.values():
        team["trend"].sort(key=lambda g: (g["date"], g["game_id"]))
    return sorted(totals.values(), key=lambda t: t["team_id"])


def _metric(totals: Mapping, key: str, sets: int) -> dict:
    result = measure(key, totals, sets=sets)
    return {"numerator": result.numerator, "denominator": result.denominator,
            "value": None if result.value is None else round(result.value, 2)}


def compute_team_advanced(games: Iterable[Mapping], scores: Mapping[int, Mapping[str, int]]) -> list[dict]:
    """名次規則同排行榜：同值同名次、下一名跳號；沒有值的名次為 None。"""
    teams = [
        {"team_id": t["team_id"], "games": t["games"],
         "points": {"attack": t["attack_points"], "block": t["block_points"], "serve": t["serve_points"],
                    "opponent_error": t["opponent_error"], "total": t["total"]},
         "metrics": {key: _metric(t, key, sets=t["sets"]) for key in METRICS},
         "trend": t["trend"]}
        for t in _team_totals(games, scores)
    ]
    for key in METRICS:
        values = [t["metrics"][key]["value"] for t in teams if t["metrics"][key]["value"] is not None]
        for t in teams:
            metric = t["metrics"][key]
            metric["rank"] = None if metric["value"] is None else 1 + sum(v > metric["value"] for v in values)
    return teams
