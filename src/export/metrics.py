# 值以分子總和 / 分母總和 * 倍率計算，取位由呼叫端決定。
# 分母為 0 時沒有值；每局項目的分母就是呼叫端給的局數，沒給（None）或 0 時沒有值。
# 沒有分母的防守負荷直接回傳整數分子。
# 局數只讀 sets 參數，忽略 totals 裡的 sets 欄。
# 未知指標或缺少統計欄位時丟出 KeyError。
# gate 與 position 由 thresholds 判定，measure 只計算數值。
# 前端不自己選分子分母、不做除法，只顯示這裡算出來的匯出值（ADR 0017）。
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

SETS = "sets"


@dataclass(frozen=True)
class Metric:
    key: str
    numerator: tuple[str, ...]
    denominator: tuple[str, ...] | None
    scale: int = 1
    gate: tuple[str, ...] = ()
    position: str | None = None


class Measure(NamedTuple):
    numerator: int
    denominator: int | None
    value: float | int | None


METRICS: Mapping[str, Metric] = {metric.key: metric for metric in (
    Metric("attack_success_rate", ("attack_points",), ("attack_total",), 100, gate=("attack_total",)),
    Metric("serve_point_rate", ("serve_points",), ("serve_total",), 100, gate=("serve_total",)),
    Metric("receive_excellent_rate", ("receive_excellent",), ("receive_total",), 100, gate=("receive_total",)),
    Metric("dig_excellent_rate", ("dig_excellent",), ("dig_total",), 100, gate=("dig_total",)),
    Metric("set_excellent_rate", ("set_excellent",), ("set_total",), 100, gate=("set_total",), position="S"),
    Metric("points_per_game", ("total_points",), ("games",), gate=("attack_serve_total",)),
    Metric("points_per_set", ("total_points",), (SETS,), gate=("attack_serve_total",)),
    Metric("attack_points_per_set", ("attack_points",), (SETS,), gate=("attack_total",)),
    Metric("blocks_per_set", ("block_points",), (SETS,), gate=("attack_serve_total",)),
    Metric("serve_points_per_set", ("serve_points",), (SETS,), gate=("serve_total",)),
    Metric("receive_excellent_per_set", ("receive_excellent",), (SETS,), gate=("receive_total",)),
    Metric("blocks_per_game", ("block_points",), ("games",), gate=("attack_serve_total",)),
    Metric("defense_load", ("receive_total", "dig_total"), None, gate=("attack_serve_total",)),
    Metric("defense_efficiency", ("receive_excellent", "dig_excellent"), ("receive_total", "dig_total"),
           100, gate=("receive_total", "dig_total")),
)}


def measure(key: str, totals: Mapping, sets: int | None = None) -> Measure:
    metric = METRICS[key]
    numerator = sum(totals[field] for field in metric.numerator)
    if metric.denominator is None:
        return Measure(numerator, None, numerator)
    if metric.denominator == (SETS,):
        denominator = sets
    else:
        denominator = sum(totals[field] for field in metric.denominator)
    value = numerator / denominator * metric.scale if denominator else None
    return Measure(numerator, denominator, value)


def columns(keys: Iterable[str]) -> tuple[str, ...]:
    fields = set()
    for key in keys:
        metric = METRICS[key]
        fields.update(metric.numerator, metric.denominator or ())
    fields.discard(SETS)
    return tuple(sorted(fields))
