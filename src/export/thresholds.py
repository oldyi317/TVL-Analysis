from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


THRESHOLD_FLOOR = 10


@dataclass(frozen=True)
class BoardSpec:
    key: str
    label: str
    kind: Literal["count", "rate"]
    title: str | None
    numerator: str | None = None
    denominator: str | None = None
    attempts_key: str | None = None
    attempts_label: str | None = None
    coefficient: float | None = None
    position: str | None = None
    makeup: bool = False


RATE_BOARDS = (
    BoardSpec("attack_success_rate", "攻擊成功率", "rate", None, "attack_points", "attack_total",
              "attack_total", "攻擊次數", 2.0, makeup=True),
    BoardSpec("serve_point_rate", "發球得分率", "rate", None, "serve_points", "serve_total",
              "serve_total", "發球次數", 1.75, makeup=True),
    BoardSpec("receive_excellent_rate", "接發球到位率", "rate", None, "receive_excellent", "receive_total",
              "receive_total", "接發球次數", 3.0, makeup=True),
    BoardSpec("dig_excellent_rate", "防守到位率", "rate", None, "dig_excellent", "dig_total",
              "dig_total", "防守次數", 1.0, makeup=True),
    BoardSpec("set_excellent_rate", "舉球到位率", "rate", None, "set_excellent", "set_total",
              "set_total", "舉球次數", 8.0, position="S", makeup=True),
    BoardSpec("points_per_game", "場均得分", "rate", None, "total_points", "games",
              "attack_serve_total", "攻擊＋發球次數", 3.0),
    BoardSpec("blocks_per_set", "每局攔網", "rate", None, "block_points", None,
              "attack_serve_total", "攻擊＋發球次數", 3.0),
)


def attempts(row: Mapping, spec: BoardSpec) -> int:
    if spec.attempts_key == "attack_serve_total":
        return row["attack_total"] + row["serve_total"]
    return row[spec.attempts_key]


def threshold_for(spec: BoardSpec, sets: int) -> int:
    return max(THRESHOLD_FLOOR, int(spec.coefficient * sets))


_SPECS = {spec.key: spec for spec in RATE_BOARDS}
_OWN_COUNT_KEYS = tuple(s.key for s in RATE_BOARDS if s.attempts_key != "attack_serve_total")


def _met(row: Mapping, key: str, sets: int) -> bool:
    spec = _SPECS[key]
    if spec.position and row["position"] != spec.position:
        return False
    return attempts(row, spec) >= threshold_for(spec, sets)


def eligible_metrics(row: Mapping, sets: int) -> frozenset[str]:
    """球員在球隊已打局數下達門檻的進階指標；次數與係數取自 RATE_BOARDS。"""
    eligible = {key for key in _OWN_COUNT_KEYS if _met(row, key, sets)}
    if _met(row, "points_per_game", sets):
        eligible |= {"points_per_game", "blocks_per_game", "defense_load"}
        if sets:
            eligible.add("blocks_per_set")
    if _met(row, "receive_excellent_rate", sets) or _met(row, "dig_excellent_rate", sets):
        eligible.add("defense_efficiency")
    return frozenset(eligible)
