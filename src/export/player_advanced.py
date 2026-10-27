"""球員進階數據的純函式：比率、PR 值與同位置分佈。

匯出器用這一處，不依賴 pandas。
輸入為球員某賽季的合計（games 與 13 項統計欄，欄名同 player_match_stats）。
"""
from collections.abc import Iterable, Mapping, Sequence

MIN_SETS = 5
MIN_GAMES = 2
RATIO_KEYS = (
    "attack_success_rate", "receive_excellent_rate", "serve_point_rate", "dig_excellent_rate",
    "points_per_game", "set_excellent_rate", "blocks_per_set", "defense_load", "defense_efficiency",
)
PR_KEYS = (
    "attack_success_rate", "receive_excellent_rate", "serve_point_rate", "dig_excellent_rate",
    "set_excellent_rate", "blocks_per_set", "defense_efficiency",
)


def _pct(numerator: float, denominator: float) -> float:
    return numerator / denominator * 100 if denominator > 0 else 0.0


def _per(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def compute_ratios(totals: Mapping) -> dict[str, float | None]:
    """九項比率，沿用舊站公式：率類為百分比，分母為 0 給 0；局數未知時每局攔網為 None。"""
    defense_total = totals["receive_total"] + totals["dig_total"]
    return {
        "attack_success_rate": _pct(totals["attack_points"], totals["attack_total"]),
        "receive_excellent_rate": _pct(totals["receive_excellent"], totals["receive_total"]),
        "serve_point_rate": _pct(totals["serve_points"], totals["serve_total"]),
        "dig_excellent_rate": _pct(totals["dig_excellent"], totals["dig_total"]),
        "points_per_game": _per(totals["total_points"], totals["games"]),
        "set_excellent_rate": _pct(totals["set_excellent"], totals["set_total"]),
        "blocks_per_set": (None if totals["sets_played"] is None
                           else _per(totals["block_points"], totals["sets_played"])),
        "defense_load": defense_total,
        "defense_efficiency": _pct(totals["receive_excellent"] + totals["dig_excellent"], defense_total),
    }


def percentile_ranks(values: Sequence[float]) -> list[float]:
    """等同 pandas Series.rank(pct=True)：同分取平均名次，除以人數後乘 100 取一位小數。"""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    start = 0
    while start < n:
        end = start
        while end + 1 < n and values[order[end + 1]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end + 1) / 2
        for k in range(start, end + 1):
            ranks[order[k]] = round(average_rank / n * 100, 1)
        start = end + 1
    return ranks


def _in_pool(result: Mapping) -> bool:
    return not result["insufficient_sets"] and result["position"] is not None


def compute_advanced(rows: Iterable[Mapping]) -> list[dict]:
    """同組別同賽季的球員合計 → 每人比率、PR 值與局數不足標記。

    PR 值只在同位置且樣本足夠的球員間計算：已知局數須達 MIN_SETS，未知時須達 MIN_GAMES。
    每局攔網在未知局數時為 None，不進該指標的 PR 與分佈。
    """
    results = [
        {
            "player_id": row["player_id"],
            "position": row["position"],
            "games": row["games"],
            "sets_played": row["sets_played"],
            "insufficient_sets": (row["games"] < MIN_GAMES if row["sets_played"] is None
                                  else row["sets_played"] < MIN_SETS),
            "ratios": compute_ratios(row),
            "pr": {key: None for key in PR_KEYS},
        }
        for row in rows
    ]
    pools: dict[str, list[dict]] = {}
    for result in results:
        if _in_pool(result):
            pools.setdefault(result["position"], []).append(result)
    for members in pools.values():
        for key in PR_KEYS:
            eligible = [m for m in members if m["ratios"][key] is not None]
            ranks = percentile_ranks([m["ratios"][key] for m in eligible])
            for member, rank in zip(eligible, ranks):
                member["pr"][key] = rank
    return results


def distributions(results: Iterable[Mapping]) -> dict[str, dict[str, list[float]]]:
    """每位置各指標的分佈：取 PR 值計算池內的球員值，升冪排列。"""
    out: dict[str, dict[str, list[float]]] = {}
    for result in results:
        if not _in_pool(result):
            continue
        metrics = out.setdefault(result["position"], {key: [] for key in RATIO_KEYS})
        for key in RATIO_KEYS:
            if result["ratios"][key] is not None:
                metrics[key].append(result["ratios"][key])
    for metrics in out.values():
        for values in metrics.values():
            values.sort()
    return out
