"""球員進階數據的純函式：比率、PR 值與同位置分佈。

匯出器用這一處，不依賴 pandas。
輸入為球員某賽季的合計（games 與 13 項統計欄，欄名同 player_match_stats）。
"""
from collections.abc import Iterable, Mapping, Sequence

from src.export.metrics import measure
from src.export.thresholds import RATIO_KEYS, eligible_metrics

PR_KEYS = (
    "attack_success_rate", "receive_excellent_rate", "serve_point_rate", "dig_excellent_rate",
    "set_excellent_rate", "blocks_per_set", "blocks_per_game", "defense_efficiency",
)


def compute_ratios(totals: Mapping, sets: int | None = None) -> dict[str, float | None]:
    """十項比率：率類為百分比；球隊已打局數 sets 缺或為 0 時每局攔網為 None。"""
    ratios = {}
    for key in RATIO_KEYS:
        value = measure(key, totals, sets=sets).value
        # 暫時：分母為 0 仍輸出 0.0，#348 會刪
        ratios[key] = 0.0 if value is None and key != "blocks_per_set" else value
    return ratios


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


def compute_advanced(rows: Iterable[Mapping], team_sets: Mapping[int, int]) -> list[dict]:
    """同組別同賽季的球員合計 → 每人比率、PR 值與未達門檻標記。

    PR 值只在同位置、且該指標次數達浮動門檻（與排行榜同一份，見 thresholds）的球員間逐項計算；
    每局攔網的分母是球員所屬球隊已打局數，查無時為 None。
    """
    results = []
    for row in rows:
        sets = team_sets.get(row["team_id"], 0)
        eligible = eligible_metrics(row, sets)
        results.append({
            "player_id": row["player_id"],
            "position": row["position"],
            "games": row["games"],
            "team_sets": sets,
            "eligible": eligible,
            "below_threshold": not eligible & set(PR_KEYS),
            "ratios": compute_ratios(row, sets),
            "pr": {key: None for key in PR_KEYS},
        })
    pools: dict[str, list[dict]] = {}
    for result in results:
        if result["position"] is not None:
            pools.setdefault(result["position"], []).append(result)
    for members in pools.values():
        for key in PR_KEYS:
            ranked = [m for m in members if key in m["eligible"] and m["ratios"][key] is not None]
            ranks = percentile_ranks([m["ratios"][key] for m in ranked])
            for member, rank in zip(ranked, ranks):
                member["pr"][key] = rank
    return results


def distributions(results: Iterable[Mapping]) -> dict[str, dict[str, list[float]]]:
    """每位置各指標的分佈：只收該指標達門檻且有位置的球員值，升冪排列。"""
    out: dict[str, dict[str, list[float]]] = {}
    for result in results:
        if result["position"] is None or result["below_threshold"]:
            continue
        metrics = out.setdefault(result["position"], {key: [] for key in RATIO_KEYS})
        for key in RATIO_KEYS:
            if key in result["eligible"] and result["ratios"][key] is not None:
                metrics[key].append(result["ratios"][key])
    for metrics in out.values():
        for values in metrics.values():
            values.sort()
    return out
