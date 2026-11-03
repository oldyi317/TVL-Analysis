from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BoardSpec:
    key: str
    label: str
    kind: Literal["count", "rate"]
    title: str | None


COUNT_BOARDS = (BoardSpec("total_points", "總得分", "count", "得分王"),)


def _count_board(spec: BoardSpec, rows: list[dict]) -> dict:
    ordered = sorted(rows, key=lambda r: (-r[spec.key], r["name"]))
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
            "numerator": None,
            "denominator": None,
            "title": spec.title if rank == 1 else None,
        })
    return {"key": spec.key, "label": spec.label, "kind": spec.kind, "threshold": None, "rows": leaders}


def compute_leaders(rows: list[dict], boards: Sequence[BoardSpec] = COUNT_BOARDS) -> list[dict]:
    return [_count_board(spec, rows) for spec in boards]
