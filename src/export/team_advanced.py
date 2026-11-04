"""球隊指標的純函式：全隊分子總和除以分母總和，再算組別內名次。

輸入為單一賽季單一組別的單場數據（box score 的正規列，不含黃金決勝局）與各場比分局數。
"""
from collections.abc import Iterable, Mapping

METRICS = (
    ("attack_success_rate", "attack_points", "attack_total", 100),
    ("serve_point_rate", "serve_points", "serve_total", 100),
    ("receive_excellent_rate", "receive_excellent", "receive_total", 100),
    ("dig_excellent_rate", "dig_excellent", "dig_total", 100),
    ("set_excellent_rate", "set_excellent", "set_total", 100),
    ("blocks_per_set", "block_points", "sets", 1),
)
TOTAL_FIELDS = sorted({f for _, *fields, _ in METRICS for f in fields} - {"sets"})


def _team_totals(games: Iterable[Mapping], sets: Mapping[int, int]) -> list[dict]:
    totals: dict[int, dict] = {}
    for game in games:
        for side in ("home", "away"):
            rows = game[side]
            if not rows:
                continue
            team = totals.setdefault(rows[0]["team_id"], {
                "team_id": rows[0]["team_id"], "games": 0, "sets": 0, **dict.fromkeys(TOTAL_FIELDS, 0),
            })
            team["games"] += 1
            team["sets"] += sets[game["game_id"]]
            for row in rows:
                for field in TOTAL_FIELDS:
                    team[field] += row[field]
    return sorted(totals.values(), key=lambda t: t["team_id"])


def _metric(totals: Mapping, numerator: str, denominator: str, scale: int) -> dict:
    num, den = totals[numerator], totals[denominator]
    return {"numerator": num, "denominator": den, "value": round(num / den * scale, 2) if den else None}


def compute_team_advanced(games: Iterable[Mapping], sets: Mapping[int, int]) -> list[dict]:
    """名次規則同排行榜：同值同名次、下一名跳號；沒有值的名次為 None。"""
    teams = [
        {"team_id": t["team_id"], "games": t["games"],
         "metrics": {key: _metric(t, num, den, scale) for key, num, den, scale in METRICS}}
        for t in _team_totals(games, sets)
    ]
    for key, *_ in METRICS:
        values = [t["metrics"][key]["value"] for t in teams if t["metrics"][key]["value"] is not None]
        for t in teams:
            metric = t["metrics"][key]
            metric["rank"] = None if metric["value"] is None else 1 + sum(v > metric["value"] for v in values)
    return teams
