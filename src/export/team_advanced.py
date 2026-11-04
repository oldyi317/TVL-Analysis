"""球隊指標的純函式：全隊分子總和除以分母總和，再算組別內名次。

輸入為單一賽季單一組別的單場數據（box score 的正規列，不含黃金決勝局）與各場比分（局數、兩隊總分與兩隊編號，不含黃金決勝局）。
"""
import logging
from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)

METRICS = (
    ("attack_success_rate", "attack_points", "attack_total", 100),
    ("serve_point_rate", "serve_points", "serve_total", 100),
    ("receive_excellent_rate", "receive_excellent", "receive_total", 100),
    ("dig_excellent_rate", "dig_excellent", "dig_total", 100),
    ("set_excellent_rate", "set_excellent", "set_total", 100),
    ("blocks_per_set", "block_points", "sets", 1),
)
TOTAL_FIELDS = sorted({f for _, *fields, _ in METRICS for f in fields} - {"sets"})


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
            game_totals = {**skill, "sets": score["sets"]}
            team["trend"].append({
                "game_id": game["game_id"], "date": game["date"],
                "opponent_id": score["away_team_id" if side == "home" else "home_team_id"],
                "metrics": {key: _metric(game_totals, num, den, scale) for key, num, den, scale in METRICS},
            })
            points = score[f"{side}_points"]
            opponent_error = points - skill["attack_points"] - skill["block_points"] - skill["serve_points"]
            if opponent_error < 0:
                logger.warning("對手失誤得分推算為負值：game_id=%s team_id=%s opponent_error=%s",
                               game["game_id"], team["team_id"], opponent_error)
            team["total"] += points
            team["opponent_error"] += opponent_error
    for team in totals.values():
        team["trend"].sort(key=lambda g: (g["date"], g["game_id"]))
    return sorted(totals.values(), key=lambda t: t["team_id"])


def _metric(totals: Mapping, numerator: str, denominator: str, scale: int) -> dict:
    num, den = totals[numerator], totals[denominator]
    return {"numerator": num, "denominator": den, "value": round(num / den * scale, 2) if den else None}


def compute_team_advanced(games: Iterable[Mapping], scores: Mapping[int, Mapping[str, int]]) -> list[dict]:
    """名次規則同排行榜：同值同名次、下一名跳號；沒有值的名次為 None。"""
    teams = [
        {"team_id": t["team_id"], "games": t["games"],
         "points": {"attack": t["attack_points"], "block": t["block_points"], "serve": t["serve_points"],
                    "opponent_error": t["opponent_error"], "total": t["total"]},
         "metrics": {key: _metric(t, num, den, scale) for key, num, den, scale in METRICS},
         "trend": t["trend"]}
        for t in _team_totals(games, scores)
    ]
    for key, *_ in METRICS:
        values = [t["metrics"][key]["value"] for t in teams if t["metrics"][key]["value"] is not None]
        for t in teams:
            metric = t["metrics"][key]
            metric["rank"] = None if metric["value"] is None else 1 + sum(v > metric["value"] for v in values)
    return teams
