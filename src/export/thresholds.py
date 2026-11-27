from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from src.export.metrics import METRICS, SETS


THRESHOLD_FLOOR = 10

COEFFICIENTS = {
    "attack_total": 2.0,
    "serve_total": 1.75,
    "receive_total": 3.0,
    "dig_total": 1.0,
    "set_total": 8.0,
    "attack_serve_total": 3.0,
}
ATTEMPTS_LABELS = {
    "attack_total": "攻擊次數",
    "serve_total": "發球次數",
    "receive_total": "接發球次數",
    "dig_total": "防守次數",
    "set_total": "舉球次數",
    "attack_serve_total": "攻擊＋發球次數",
}


@dataclass(frozen=True)
class BoardSpec:
    key: str
    label: str
    kind: Literal["count", "rate"]
    title: str | None
    numerator: str | None = None
    denominator: str | None = None
    attempts_key: str | None = None
    scale: int = 1
    position: str | None = None
    makeup: bool = False

    @property
    def attempts_label(self) -> str | None:
        return ATTEMPTS_LABELS[self.attempts_key] if self.attempts_key else None

    @property
    def coefficient(self) -> float | None:
        return COEFFICIENTS[self.attempts_key] if self.attempts_key else None


def _rate_board(key: str, label: str, makeup: bool = False) -> BoardSpec:
    metric = METRICS[key]
    # 比率榜的分子、分母、看的次數都只有一項，多於一項在這裡就會出錯
    numerator, = metric.numerator
    denominator, = metric.denominator
    attempts_key, = metric.gate
    return BoardSpec(key, label, "rate", None, numerator, denominator, attempts_key,
                     metric.scale, metric.position, makeup)


RATE_BOARDS = (
    _rate_board("attack_success_rate", "攻擊成功率", makeup=True),
    _rate_board("serve_point_rate", "發球得分率", makeup=True),
    _rate_board("receive_excellent_rate", "接發球到位率", makeup=True),
    _rate_board("dig_excellent_rate", "防守到位率", makeup=True),
    _rate_board("set_excellent_rate", "舉球到位率", makeup=True),
    _rate_board("points_per_game", "場均得分"),
    _rate_board("points_per_set", "每局得分"),
    _rate_board("attack_points_per_set", "每局攻擊得分"),
    _rate_board("blocks_per_set", "每局攔網"),
    _rate_board("serve_points_per_set", "每局發球得分"),
    _rate_board("receive_excellent_per_set", "每局接發球到位"),
)

RATIO_KEYS = (
    "attack_success_rate", "receive_excellent_rate", "serve_point_rate", "dig_excellent_rate",
    "points_per_game", "set_excellent_rate", "blocks_per_set", "blocks_per_game", "defense_load",
    "defense_efficiency",
)


def _attempts(row: Mapping, key: str) -> int:
    if key == "attack_serve_total":
        return row["attack_total"] + row["serve_total"]
    return row[key]


def attempts(row: Mapping, spec: BoardSpec) -> int:
    return _attempts(row, spec.attempts_key)


def _threshold(key: str, sets: int) -> int:
    return max(THRESHOLD_FLOOR, int(COEFFICIENTS[key] * sets))


def threshold_for(spec: BoardSpec, sets: int) -> int:
    return _threshold(spec.attempts_key, sets)


def eligible_metrics(row: Mapping, sets: int) -> frozenset[str]:
    """球員在球隊已打局數下達門檻的進階指標：位置符合、看的次數任一項達門檻；每局項目另要求局數大於 0。"""
    eligible = set()
    for key in RATIO_KEYS:
        metric = METRICS[key]
        if metric.position and row["position"] != metric.position:
            continue
        if metric.denominator == (SETS,) and sets <= 0:
            continue
        if any(_attempts(row, gate) >= _threshold(gate, sets) for gate in metric.gate):
            eligible.add(key)
    return frozenset(eligible)
