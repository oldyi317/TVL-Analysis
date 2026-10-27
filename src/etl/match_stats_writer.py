"""
官網單場數據寫入器（spec #75）
對本季已有比分、尚無統計的場次抓官網賽事頁，解析兩隊單場數據後寫入
player_match_stats 與 source='match_page' 的週登錄；每場一個 transaction、insert-only。
"""

import argparse
import sqlite3
import time
from collections.abc import Callable

import requests

from src.etl.official_box_score import fetch_page, parse_box_score
from src.etl.stats_crawler import (
    PlayerResolver, find_registration_id, insert_stat_rows, upsert_roster_registration,
)
from src.utils.constants import OFFICIAL_BOX_SCORE_FROM_CUP
from src.utils.db_config import get_connection
from src.utils.logger import get_logger
from src.utils.match_status import HAS_RESULT_WHERE
from src.utils.teams import UnknownTeamError, resolve_team

logger = get_logger(__name__)

_TEAM_HAS_STATS = """EXISTS (
    SELECT 1 FROM player_match_stats s
    JOIN roster_registrations r ON r.registration_id = s.registration_id
    WHERE r.team_id = m.{side}_team_id AND r.gender = m.gender
      AND s.match_date = m.match_date AND s.is_golden_set = m.is_golden_set
)"""


class SkipMatch(Exception):
    """整場不寫的原因；該場 rollback 並記 log，下次每日更新重試。"""


def candidate_matches(conn: sqlite3.Connection) -> list[dict]:
    cursor = conn.execute(
        f"""SELECT m.match_id, m.game_id, m.gender, m.cup_id, m.match_date, m.round_name,
                   m.is_golden_set, m.home_team, m.away_team, m.home_team_id, m.away_team_id,
                   m.home_total, m.away_total
            FROM matches m
            WHERE m.cup_id >= ? AND {HAS_RESULT_WHERE}
              AND m.home_team_id IS NOT NULL AND m.away_team_id IS NOT NULL
              AND NOT {_TEAM_HAS_STATS.format(side="home")}
              AND NOT {_TEAM_HAS_STATS.format(side="away")}
            ORDER BY m.match_date, m.gender, m.is_golden_set, m.game_id""",
        (OFFICIAL_BOX_SCORE_FROM_CUP,),
    )
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _warn_unresolved_teams(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        f"""SELECT gender, game_id, home_team, away_team FROM matches
            WHERE cup_id >= ? AND {HAS_RESULT_WHERE}
              AND (home_team_id IS NULL OR away_team_id IS NULL)""",
        (OFFICIAL_BOX_SCORE_FROM_CUP,),
    ).fetchall()
    for gender, game_id, home, away in rows:
        logger.warning("[%s/%d] %s vs %s 有比分但隊伍未解析，不成為候選（請補 sql/team_seed.sql）", gender, game_id, home, away)


def _match_label(match: dict) -> str:
    return (
        f"[{match['gender']}/{match['game_id']}] {match['match_date']} "
        f"{match['home_team']} vs {match['away_team']}"
        f"{'（黃金局）' if match['is_golden_set'] else ''}"
    )


def _sides(conn: sqlite3.Connection, match: dict, teams: list[dict]) -> dict[int, dict]:
    sides = {}
    for team in teams:
        try:
            team_id, _ = resolve_team(conn, team["name"], match["gender"])
        except UnknownTeamError as exc:
            raise SkipMatch(str(exc)) from exc
        if team_id in sides:
            raise SkipMatch(f"頁面兩隊 {[t['name'] for t in teams]} 解析為同一球隊 team_id={team_id}")
        sides[team_id] = team
    expected = {match["home_team_id"], match["away_team_id"]}
    if set(sides) != expected:
        raise SkipMatch(f"頁面隊伍 {[t['name'] for t in teams]} 對不回賽事主客隊 {sorted(expected)}")
    return sides


def _week_start(conn: sqlite3.Connection, match: dict) -> str:
    row = conn.execute(
        "SELECT MIN(match_date) FROM matches WHERE cup_id = ? AND gender = ? AND round_name = ?",
        (match["cup_id"], match["gender"], match["round_name"]),
    ).fetchone()
    return row[0] or match["match_date"]


def _resolve_players(
    resolver: PlayerResolver, team_id: int, gender: str, team: dict,
) -> tuple[list[tuple[int, dict]], list[str]]:
    resolved, unresolved, seen = [], [], set()
    for row in team["unresolved_rows"]:
        unresolved.append(f"{team['name']} #{row['jersey_number']}（姓名空白、數值非零）")
    for player in team["players"]:
        pid = resolver.resolve(team_id, gender, player["name"], player["jersey_number"])
        if pid is None:
            unresolved.append(f"{team['name']} #{player['jersey_number']} {player['name']}（對不上賽季報名名單）")
        elif pid in seen:
            unresolved.append(f"{team['name']} #{player['jersey_number']} {player['name']}（與同隊另一列對到同一球員）")
        else:
            seen.add(pid)
            resolved.append((pid, player))
    return resolved, unresolved


def _write_side(conn: sqlite3.Connection, match: dict, team_id: int, resolved: list, week_start: str) -> int:
    gender = match["gender"]
    opponent = match["away_team"] if team_id == match["home_team_id"] else match["home_team"]
    rows = []
    for pid, p in resolved:
        upsert_roster_registration(
            conn, pid,
            {"team_id": team_id, "team_gender": gender,
             "jersey_number": p["jersey_number"], "position": p["position"]},
            match["round_name"], week_start, source="match_page", cup_id=match["cup_id"],
        )
        if p["all_zero"] and not p["total_points"]:
            continue
        rows.append((
            find_registration_id(conn, pid, team_id, gender, match["cup_id"], match["round_name"]),
            match["match_date"], opponent, None,
            p["attack_total"], p["attack_points"], p["block_points"],
            p["serve_total"], p["serve_points"],
            p["receive_total"], p["receive_excellent"],
            p["dig_total"], p["dig_excellent"],
            p["set_total"], p["set_excellent"], p["total_points"],
            match["is_golden_set"],
        ))
    insert_stat_rows(conn, rows)
    return len(rows)


def write_match(
    conn: sqlite3.Connection, match: dict, result: dict, resolver: PlayerResolver, allow_partial: bool = False,
) -> tuple[int, int]:
    """單場寫入（呼叫端負責 transaction）。回傳 (統計列數, 登錄數)。"""
    label = _match_label(match)
    if result["gate_errors"]:
        raise SkipMatch("閘門不通過：" + "；".join(result["gate_errors"]))
    if not match["round_name"]:
        raise SkipMatch("matches.round_name 為空，無法決定 week_label")
    for warning in result["warnings"]:
        logger.warning("%s %s", label, warning)
    sides = _sides(conn, match, result["teams"])
    for team_id, team in sides.items():
        expected = match["home_total"] if team_id == match["home_team_id"] else match["away_total"]
        if team["total_points"] != expected:
            logger.warning("%s %s 頁面總計 %s ≠ matches 總分 %s", label, team["name"], team["total_points"], expected)
    resolved_sides, unresolved = {}, []
    for team_id, team in sides.items():
        resolved_sides[team_id], missing = _resolve_players(resolver, team_id, match["gender"], team)
        unresolved.extend(missing)
    if unresolved and not allow_partial:
        raise SkipMatch("未解析列：" + "；".join(unresolved) + "（可用 --force-partial 寫入其餘球員）")
    if unresolved:
        logger.warning("%s --force-partial：略過未解析列 %s", label, "；".join(unresolved))
    week_start = _week_start(conn, match)
    stats = sum(_write_side(conn, match, team_id, resolved, week_start) for team_id, resolved in resolved_sides.items())
    return stats, sum(len(r) for r in resolved_sides.values())


def write_match_stats(
    conn: sqlite3.Connection,
    fetch: Callable[[str, int], str] = fetch_page,
    force_partial: tuple[str, int] | None = None,
    delay: float = 0.5,
) -> dict:
    """對所有候選場次抓頁、解析、寫入；每場一個 transaction，失敗只記 log。"""
    if conn.in_transaction:
        raise RuntimeError("呼叫端有未提交的交易，寫入器的 rollback 會連帶丟掉它")
    _warn_unresolved_teams(conn)
    summary = {"written": [], "skipped": []}
    resolvers: dict[int, PlayerResolver] = {}
    for index, match in enumerate(candidate_matches(conn)):
        key = (match["gender"], match["game_id"])
        label = _match_label(match)
        if index:
            time.sleep(delay)
        conn.execute("BEGIN")
        try:
            resolver = resolvers.get(match["cup_id"]) or PlayerResolver(conn, match["cup_id"])
            resolvers[match["cup_id"]] = resolver
            if resolver.legacy:
                raise SkipMatch(f"cup_id={match['cup_id']} 無賽季報名名單，官網來源不動態新增球員")
            result = parse_box_score(fetch(match["gender"], match["game_id"]))
            stats, registrations = write_match(conn, match, result, resolver, force_partial == key)
        except Exception as exc:  # 任一場失敗只記 log，其餘候選照常處理
            conn.rollback()
            expected = isinstance(exc, (requests.RequestException, ValueError, SkipMatch, sqlite3.Error))
            reason = str(exc) if expected else f"{type(exc).__name__}: {exc}"
            logger.error("%s 整場不寫：%s", label, reason)
            summary["skipped"].append((key, reason))
            continue
        conn.commit()
        logger.info("%s 寫入統計 %d 筆、週登錄 %d 筆", label, stats, registrations)
        summary["written"].append(key)
    for resolver in resolvers.values():
        resolver.log_reports()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="官網單場數據寫入 player_match_stats（本季候選場次）")
    parser.add_argument("--force-partial", type=int, metavar="GAME_ID", help="該場有未解析列時仍寫入其餘球員")
    parser.add_argument("--gender", choices=("M", "F"), help="--force-partial 的組別；同 game_id 兩組皆為候選時必填")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args(argv)
    conn = get_connection()
    try:
        force = None
        if args.force_partial is not None:
            matched = [
                (m["gender"], m["game_id"]) for m in candidate_matches(conn)
                if m["game_id"] == args.force_partial and args.gender in (None, m["gender"])
            ]
            if len(matched) != 1:
                parser.error(f"--force-partial {args.force_partial} 對到 {len(matched)} 場候選（請確認 game_id 或加 --gender）")
            force = matched[0]
        summary = write_match_stats(conn, fetch_page, force_partial=force, delay=args.delay)
    finally:
        conn.close()
    logger.info("官網單場數據：寫入 %d 場、略過 %d 場", len(summary["written"]), len(summary["skipped"]))
    return 1 if summary["skipped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
