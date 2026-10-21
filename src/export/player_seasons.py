"""球員賽季列的純函式：主要位置、最新球隊／背號、報名名單補位。

不碰 SQL，匯出器只負責讀取三份輸入再呼叫 compute_player_seasons。
"""
from collections.abc import Iterable, Mapping, Sequence

STAT_FIELDS = [
    "sets_played", "attack_total", "attack_points", "block_points", "serve_total", "serve_points",
    "receive_total", "receive_excellent", "dig_total", "dig_excellent", "set_total", "set_excellent",
    "total_points",
]


def _season_entry(cup_id: int, team_id: int, jersey: int | None, games: int, stats: Sequence[int]) -> dict:
    return {
        "cup_id": cup_id,
        "team_id": team_id,
        "jersey_number": jersey,
        "position": None,
        "season_roster_position": None,
        "games": games,
        **dict(zip(STAT_FIELDS, stats)),
    }


def _newest_first(rows: list[Sequence]) -> list[Sequence]:
    """week_start_date 非 NULL 優先、新到舊，再 registration_id 大到小。"""
    ordered = sorted(rows, key=lambda r: r[6], reverse=True)
    return sorted(ordered, key=lambda r: (r[5] is not None, r[5] or ""), reverse=True)


def compute_player_seasons(
    totals: Mapping[tuple[int, int], Sequence[int]],
    registrations: Iterable[Sequence],
    season_rosters: Iterable[Sequence],
) -> dict[int, list[dict]]:
    """每位球員各賽季一列，依 cup_id 由小到大；沒有登錄也沒有報名名單的球員不出現。

    totals: (player_id, cup_id) -> [出賽場數, *STAT_FIELDS 合計]，缺的視為全 0。
    registrations: (player_id, cup_id, team_id, jersey_number, position, week_start_date, registration_id)，
        不必排序；最新一週登錄決定球隊與背號，position 為各週登錄的多數決（不計空白），同數取最近一次。
    season_rosters: (player_id, cup_id, team_id, jersey_number, position)。
        只補顯示（ADR 0004）：沒有登錄的賽季補一列零統計；有登錄且同隊才補 season_roster_position
        與缺的背號，position 仍是主要位置。
    """
    groups: dict[tuple[int, int], list[Sequence]] = {}
    for row in registrations:
        groups.setdefault((row[0], row[1]), []).append(row)

    empty = [0] * (len(STAT_FIELDS) + 1)
    seasons: dict[int, list[dict]] = {}
    for (player_id, cup_id) in sorted(groups):
        rows = _newest_first(groups[(player_id, cup_id)])
        _, _, team_id, jersey, _, _, _ = rows[0]
        games, *stats = totals.get((player_id, cup_id), empty)
        entry = _season_entry(cup_id, team_id, jersey, games, stats)
        counts: dict[str, int] = {}
        for row in rows:
            if row[4] is not None:
                counts[row[4]] = counts.get(row[4], 0) + 1
        # 最近登錄優先的順序插入，max 同數時取第一個即最近一次
        entry["position"] = max(counts, key=counts.get) if counts else None
        seasons.setdefault(player_id, []).append(entry)

    for player_id, cup_id, team_id, jersey, position in season_rosters:
        entries = seasons.setdefault(player_id, [])
        entry = next((e for e in entries if e["cup_id"] == cup_id), None)
        if entry is None:
            entries.append({**_season_entry(cup_id, team_id, jersey, 0, [0] * len(STAT_FIELDS)),
                            "season_roster_position": position})
            entries.sort(key=lambda e: e["cup_id"])
            continue
        if entry["team_id"] != team_id:
            continue
        entry["season_roster_position"] = position
        if entry["jersey_number"] is None:
            entry["jersey_number"] = jersey
    return seasons
