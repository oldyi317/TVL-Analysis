from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Literal

from src.export.player_advanced import MIN_GAMES, MIN_SETS, compute_ratios


@dataclass(frozen=True)
class BoardSpec:
    key: str
    label: str
    kind: Literal["count", "rate"]
    title: str | None
    numerator: str | None = None
    denominator: str | None = None
    attempts: str | None = None
    position: str | None = None


COUNT_BOARDS = (
    BoardSpec("total_points", "總得分", "count", "得分王"),
    BoardSpec("attack_points", "攻擊得分", "count", "攻擊王"),
    BoardSpec("block_points", "攔網得分", "count", "攔網王"),
    BoardSpec("serve_points", "發球得分", "count", "發球王"),
    BoardSpec("receive_excellent", "接發球到位", "count", None),
    BoardSpec("dig_excellent", "防守到位", "count", None),
    BoardSpec("set_excellent", "舉球到位", "count", None, position="S"),
)


RATE_BOARDS = (
    BoardSpec("attack_success_rate", "攻擊成功率", "rate", None, "attack_points", "attack_total", "攻擊 10 次"),
    BoardSpec("serve_point_rate", "發球得分率", "rate", None, "serve_points", "serve_total", "發球 10 次"),
    BoardSpec("receive_excellent_rate", "接發球到位率", "rate", None, "receive_excellent", "receive_total", "接發球 10 次"),
    BoardSpec("dig_excellent_rate", "防守到位率", "rate", None, "dig_excellent", "dig_total", "防守 10 次"),
    BoardSpec("set_excellent_rate", "舉球到位率", "rate", None, "set_excellent", "set_total", "舉球 10 次", position="S"),
    BoardSpec("points_per_game", "場均得分", "rate", None, "total_points", "games", "出賽 10 場"),
    BoardSpec("blocks_per_set", "每局攔網", "rate", None, "block_points", "sets_played", "出賽 10 局"),
)


def _board(spec: BoardSpec, rows: list[dict]) -> dict:
    threshold = None
    if spec.kind == "rate":
        has_sets = any(r["sets_played"] is not None for r in rows)
        participation = f"出賽 {MIN_SETS} 局" if has_sets else f"出賽 {MIN_GAMES} 場"
        if has_sets and spec.denominator != "sets_played" and any(r["sets_played"] is None for r in rows):
            participation += f"（無局數時 {MIN_GAMES} 場）"
        threshold = f"門檻：{participation}、{spec.attempts}"
    if spec.position is not None:
        rows = [r for r in rows if r["position"] == spec.position]
    if spec.kind == "rate":
        rows = [
            {**r, spec.key: round(compute_ratios(r)[spec.key], 2)}
            for r in rows
            if (r["games"] >= MIN_GAMES if r["sets_played"] is None else r["sets_played"] >= MIN_SETS)
            and r[spec.denominator] is not None and r[spec.denominator] >= 10
        ]
    ordered = sorted(rows, key=lambda r: (
        -r[spec.key], -(r[spec.denominator]) if spec.kind == "rate" else 0, r["name"]
    ))
    leaders = []
    for index, r in enumerate(ordered):
        rank = leaders[-1]["rank"] if index and r[spec.key] == ordered[index - 1][spec.key] else index + 1
        leaders.append({
            "rank": rank,
            "player_id": r["player_id"],
            "name": r["name"],
            "team_id": r["team_id"],
            "team_short": r["team_short"],
            "position": r["position"],
            "value": r[spec.key],
            "numerator": r[spec.numerator] if spec.numerator else None,
            "denominator": r[spec.denominator] if spec.denominator else None,
            "title": spec.title if rank == 1 else None,
        })
    return {"key": spec.key, "label": spec.label, "kind": spec.kind, "threshold": threshold, "rows": leaders}


def compute_leaders(rows: list[dict], boards: Sequence[BoardSpec] = COUNT_BOARDS + RATE_BOARDS) -> list[dict]:
    return [_board(spec, rows) for spec in boards
            if spec.key != "blocks_per_set" or any(r["sets_played"] is not None for r in rows)]


VOLUME_RATE_KEYS = ("attack_success_rate", "receive_excellent_rate", "dig_excellent_rate", "set_excellent_rate")


def _median(values: list[float]) -> float | None:
    return round(median(values), 2) if values else None


def compute_volume_rates(rows: list[dict]) -> list[dict]:
    """量率散點：直接取同技術比率榜的上榜球員，門檻與舉球限 S 都與比率榜同一份邏輯。"""
    specs = {spec.key: spec for spec in RATE_BOARDS}
    result = []
    for key in VOLUME_RATE_KEYS:
        points = [
            {"player_id": r["player_id"], "name": r["name"], "team_id": r["team_id"],
             "team_short": r["team_short"], "count": r["denominator"], "rate": r["value"]}
            for r in _board(specs[key], rows)["rows"]
        ]
        result.append({
            "key": key,
            "median_count": _median([p["count"] for p in points]),
            "median_rate": _median([p["rate"] for p in points]),
            "points": points,
        })
    return result
