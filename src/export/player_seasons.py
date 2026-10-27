"""球員賽季列的純函式：主要位置、最新球隊／背號、賽季報名名單補位。

不碰 SQL，匯出器只負責讀取三份輸入再呼叫 compute_player_seasons。
"""
from collections.abc import Iterable, Mapping
from typing import NamedTuple

STAT_FIELDS = [
    "sets_played", "attack_total", "attack_points", "block_points", "serve_total", "serve_points",
    "receive_total", "receive_excellent", "dig_total", "dig_excellent", "set_total", "set_excellent",
    "total_points",
]


class SeasonTotals(NamedTuple):
    """球員一個賽季的出賽場數與技術統計合計，stats 依 STAT_FIELDS 順序。"""
    games: int
    stats: tuple[int, ...]


class Registration(NamedTuple):
    """一筆週登錄。"""
    player_id: int
    cup_id: int
    team_id: int
    jersey_number: int | None
    position: str | None
    week_start_date: str | None
    registration_id: int


class SeasonRoster(NamedTuple):
    """賽季報名名單上的一名球員。"""
    player_id: int
    cup_id: int
    team_id: int
    jersey_number: int | None
    position: str | None


ZERO_TOTALS = SeasonTotals(0, (0,) * len(STAT_FIELDS))


def _season_entry(cup_id: int, team_id: int, jersey: int | None, totals: SeasonTotals) -> dict:
    return {
        "cup_id": cup_id,
        "team_id": team_id,
        "jersey_number": jersey,
        "position": None,
        "season_roster_position": None,
        "games": totals.games,
        **dict(zip(STAT_FIELDS, totals.stats)),
    }


def _newest_first(rows: list[Registration]) -> list[Registration]:
    """week_start_date 非 NULL 優先、新到舊，再 registration_id 大到小。"""
    ordered = sorted(rows, key=lambda r: r.registration_id, reverse=True)
    return sorted(ordered, key=lambda r: (r.week_start_date is not None, r.week_start_date or ""), reverse=True)


def compute_player_seasons(
    totals: Mapping[tuple[int, int], SeasonTotals],
    registrations: Iterable[Registration],
    season_rosters: Iterable[SeasonRoster],
) -> dict[int, list[dict]]:
    """每位球員各賽季一列，依 cup_id 由小到大；沒有登錄也沒有報名名單的球員不出現。

    totals: (player_id, cup_id) -> SeasonTotals，缺的視為全 0。
    registrations: 週登錄（Registration），不必排序；最新一週登錄決定球隊與背號，
        position 為各週登錄的多數決（不計空白），同數取最近一次。
    season_rosters: 賽季報名名單（SeasonRoster）。
        只補顯示（ADR 0004）：沒有登錄的賽季補一列零統計；有登錄且同隊才補 season_roster_position
        與缺的背號，position 仍是主要位置。
    """
    groups: dict[tuple[int, int], list[Registration]] = {}
    for reg in registrations:
        groups.setdefault((reg.player_id, reg.cup_id), []).append(reg)

    seasons: dict[int, list[dict]] = {}
    for (player_id, cup_id) in sorted(groups):
        rows = _newest_first(groups[(player_id, cup_id)])
        newest = rows[0]
        entry = _season_entry(cup_id, newest.team_id, newest.jersey_number, totals.get((player_id, cup_id), ZERO_TOTALS))
        counts: dict[str, int] = {}
        for reg in rows:
            if reg.position is not None:
                counts[reg.position] = counts.get(reg.position, 0) + 1
        # 最近登錄優先的順序插入，max 同數時取第一個即最近一次
        entry["position"] = max(counts, key=counts.get) if counts else None
        seasons.setdefault(player_id, []).append(entry)

    for roster in season_rosters:
        entries = seasons.setdefault(roster.player_id, [])
        entry = next((e for e in entries if e["cup_id"] == roster.cup_id), None)
        if entry is None:
            entries.append({**_season_entry(roster.cup_id, roster.team_id, roster.jersey_number, ZERO_TOTALS),
                            "season_roster_position": roster.position})
            entries.sort(key=lambda e: e["cup_id"])
            continue
        if entry["team_id"] != roster.team_id:
            continue
        entry["season_roster_position"] = roster.position
        if entry["jersey_number"] is None:
            entry["jersey_number"] = roster.jersey_number
    return seasons
