from collections.abc import Iterable, Mapping, Sequence
from statistics import median

from src.export.player_advanced import compute_ratios
from src.export.player_seasons import STAT_FIELDS
from src.export.thresholds import RATE_BOARDS, THRESHOLD_FLOOR, BoardSpec, attempts, threshold_for


COUNT_BOARDS = (
    BoardSpec("total_points", "總得分", "count", "得分王"),
    BoardSpec("attack_points", "攻擊得分", "count", "攻擊王"),
    BoardSpec("block_points", "攔網得分", "count", "攔網王"),
    BoardSpec("serve_points", "發球得分", "count", "發球王"),
    BoardSpec("receive_excellent", "接發球到位", "count", None),
    BoardSpec("dig_excellent", "防守到位", "count", None),
    BoardSpec("set_excellent", "舉球到位", "count", None, position="S"),
)


def threshold_text(spec: BoardSpec, team_sets: Mapping[int, int]) -> str:
    thresholds = [threshold_for(spec, sets) for sets in team_sets.values()] or [THRESHOLD_FLOOR]
    raw_below_floor = not team_sets or any(int(spec.coefficient * sets) < THRESHOLD_FLOOR for sets in team_sets.values())
    low, high = min(thresholds), max(thresholds)
    current = f"{low} 次" if low == high else f"{low} 到 {high} 次"
    floor = f"，季初至少 {THRESHOLD_FLOOR} 次" if raw_below_floor else ""
    setter = "，只計舉球（S）" if spec.position == "S" else ""
    return f"門檻：{spec.attempts_label}達所屬球隊已打局數 × {spec.coefficient:g}{floor}（目前 {current}）{setter}"


def rank_ordered(values: Sequence) -> list[int]:
    """已排序的值轉名次：同值同名次，下一名跳號。"""
    ranks: list[int] = []
    for index, value in enumerate(values):
        ranks.append(ranks[-1] if index and value == values[index - 1] else index + 1)
    return ranks


def makeup_candidate(spec: BoardSpec, rows: list[dict], team_sets: Mapping[int, int],
                     leader_rate: float | None) -> dict | None:
    """補算達標：次數未達門檻者，把差額全算失敗後仍嚴格高於第一名，才能上榜。"""
    if leader_rate is None:
        return None
    candidates = []
    for r in rows:
        count = attempts(r, spec)
        threshold = threshold_for(spec, team_sets.get(r["team_id"], 0))
        if not 0 < count < threshold:
            continue
        makeup_rate = r[spec.numerator] / threshold * 100
        if makeup_rate > leader_rate:
            candidates.append(((-makeup_rate, -count, r["name"]), r))
    return min(candidates, key=lambda c: c[0])[1] if candidates else None


def _board(spec: BoardSpec, rows: list[dict], team_sets: Mapping[int, int], apply_makeup: bool = False) -> dict:
    if spec.kind == "rate":
        threshold = threshold_text(spec, team_sets)
    else:
        threshold = "只計舉球（S）" if spec.position == "S" else None
    if spec.position is not None:
        rows = [r for r in rows if r["position"] == spec.position]
    tiebreak = {}
    makeup_ids: set[int] = set()
    candidate = None
    if spec.kind == "rate":
        qualified = []
        for r in rows:
            sets = team_sets.get(r["team_id"], 0)
            if attempts(r, spec) < threshold_for(spec, sets):
                continue
            if spec.key == "blocks_per_set":
                if not sets:
                    continue
                qualified.append({**r, spec.key: round(compute_ratios(r, sets)[spec.key], 2), "_denominator": sets})
                tiebreak[r["player_id"]] = r["block_points"]
            elif r[spec.denominator]:
                qualified.append({**r, spec.key: round(compute_ratios(r)[spec.key], 2),
                                  "_denominator": r[spec.denominator]})
                tiebreak[r["player_id"]] = r[spec.denominator]
        if spec.makeup and apply_makeup and qualified:
            leader_rate = max(r[spec.numerator] / r[spec.denominator] * 100 for r in qualified)
        else:
            leader_rate = None
        candidate = makeup_candidate(spec, rows, team_sets, leader_rate) if spec.makeup and apply_makeup else None
        if candidate:
            makeup_ids.add(candidate["player_id"])
        rows = qualified
    ordered = sorted(rows, key=lambda r: (-r[spec.key], -tiebreak.get(r["player_id"], 0), r["name"]))
    ranks = rank_ordered([r[spec.key] for r in ordered])
    if candidate:
        ordered.insert(0, {**candidate, spec.key: round(compute_ratios(candidate)[spec.key], 2),
                           "_denominator": candidate[spec.denominator]})
        ranks = [1] + [rank + 1 for rank in ranks]
    leaders = []
    for r, rank in zip(ordered, ranks):
        leaders.append({
            "rank": rank,
            "player_id": r["player_id"],
            "name": r["name"],
            "team_id": r["team_id"],
            "team_short": r["team_short"],
            "position": r["position"],
            "value": r[spec.key],
            "numerator": r[spec.numerator] if spec.numerator else None,
            "denominator": r["_denominator"] if spec.kind == "rate" else None,
            "title": spec.title if rank == 1 else None,
            "makeup": r["player_id"] in makeup_ids,
        })
    return {"key": spec.key, "label": spec.label, "kind": spec.kind, "threshold": threshold, "rows": leaders}


def compute_leaders(rows: list[dict], team_sets: Mapping[int, int],
                    boards: Sequence[BoardSpec] = COUNT_BOARDS + RATE_BOARDS,
                    apply_makeup: bool = False) -> list[dict]:
    return [_board(spec, rows, team_sets, apply_makeup) for spec in boards]


VOLUME_RATE_KEYS = ("attack_success_rate", "receive_excellent_rate", "dig_excellent_rate", "set_excellent_rate")


def _median(values: list[float]) -> float | None:
    return round(median(values), 2) if values else None


def compute_volume_rates(rows: list[dict], team_sets: Mapping[int, int]) -> list[dict]:
    """量率散點：直接取同技術比率榜的上榜球員，門檻與舉球限 S 都與比率榜同一份邏輯。"""
    specs = {spec.key: spec for spec in RATE_BOARDS}
    result = []
    for key in VOLUME_RATE_KEYS:
        points = [
            {"player_id": r["player_id"], "name": r["name"], "team_id": r["team_id"],
             "team_short": r["team_short"], "count": r["denominator"], "rate": r["value"]}
            for r in _board(specs[key], rows, team_sets)["rows"]
        ]
        result.append({
            "key": key,
            "median_count": _median([p["count"] for p in points]),
            "median_rate": _median([p["rate"] for p in points]),
            "points": points,
        })
    return result


def regular_totals(games: Iterable[Mapping]) -> dict[int, dict]:
    totals: dict[int, dict] = {}
    for game in games:
        for row in (*game["home"], *game["away"]):
            total = totals.setdefault(row["player_id"], {"games": 0, **{f: 0 for f in STAT_FIELDS}})
            total["games"] += 1
            for field in STAT_FIELDS:
                if total[field] is None or row[field] is None:
                    total[field] = None
                else:
                    total[field] += row[field]
    return totals


def count_team_sets(games: Iterable[Mapping], include_golden_set: bool = False) -> dict[int, int]:
    sets: dict[int, int] = {}
    for game in games:
        if game["sets"] is None:
            continue
        played = game["sets"] + (1 if include_golden_set and game["golden_set"] is not None else 0)
        for team_id in (game["home_team_id"], game["away_team_id"]):
            sets[team_id] = sets.get(team_id, 0) + played
    return sets
