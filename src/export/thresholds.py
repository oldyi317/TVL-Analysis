from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


THRESHOLD_FLOOR = 10
TEAM_SETS = "team_sets"

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


RATE_BOARDS = (
    BoardSpec("attack_success_rate", "攻擊成功率", "rate", None, "attack_points", "attack_total",
              "attack_total", scale=100, makeup=True),
    BoardSpec("serve_point_rate", "發球得分率", "rate", None, "serve_points", "serve_total",
              "serve_total", scale=100, makeup=True),
    BoardSpec("receive_excellent_rate", "接發球到位率", "rate", None, "receive_excellent", "receive_total",
              "receive_total", scale=100, makeup=True),
    BoardSpec("dig_excellent_rate", "防守到位率", "rate", None, "dig_excellent", "dig_total",
              "dig_total", scale=100, makeup=True),
    BoardSpec("set_excellent_rate", "舉球到位率", "rate", None, "set_excellent", "set_total",
              "set_total", scale=100, position="S", makeup=True),
    BoardSpec("points_per_game", "場均得分", "rate", None, "total_points", "games",
              "attack_serve_total"),
    BoardSpec("points_per_set", "每局得分", "rate", None, "total_points", TEAM_SETS,
              "attack_serve_total"),
    BoardSpec("attack_points_per_set", "每局攻擊得分", "rate", None, "attack_points", TEAM_SETS,
              "attack_total"),
    BoardSpec("blocks_per_set", "每局攔網", "rate", None, "block_points", TEAM_SETS,
              "attack_serve_total"),
    BoardSpec("serve_points_per_set", "每局發球得分", "rate", None, "serve_points", TEAM_SETS,
              "serve_total"),
    BoardSpec("receive_excellent_per_set", "每局接發球到位", "rate", None, "receive_excellent", TEAM_SETS,
              "receive_total"),
)


def attempts(row: Mapping, spec: BoardSpec) -> int:
    if spec.attempts_key == "attack_serve_total":
        return row["attack_total"] + row["serve_total"]
    return row[spec.attempts_key]


def threshold_for(spec: BoardSpec, sets: int) -> int:
    return max(THRESHOLD_FLOOR, int(spec.coefficient * sets))


_SPECS = {spec.key: spec for spec in RATE_BOARDS}
_OWN_COUNT_KEYS = tuple(s.key for s in RATE_BOARDS if s.denominator == s.attempts_key)


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
