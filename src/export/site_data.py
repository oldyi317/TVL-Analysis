import json
import logging
import math
import re
import sqlite3
from datetime import date
from pathlib import Path
from statistics import median

from src.export.player_advanced import compute_advanced, distributions
from src.utils.db_config import DB_PATH, PROJECT_ROOT, get_connection
from src.utils.teams import UnknownTeamError, resolve_team

DEFAULT_OUT_DIR = PROJECT_ROOT / "web" / "src" / "lib" / "data"
GROUP_BY_GENDER = {"M": "men", "F": "women"}
GROUP_ORDER = list(GROUP_BY_GENDER.values())
STANDINGS_NOTES = {
    (20, "M"): "2024-25 男子組官網只收錄第 6 週起的例行賽，戰績不完整",
    (20, "F"): "2024-25 女子組官網只收錄 8 場賽事，戰績不完整",
}
SET_RECORD_POINTS = {(3, 0): 3, (3, 1): 3, (3, 2): 2, (2, 3): 1}
SET_RECORD_KEYS = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]
WEEKDAYS = "一二三四五六日"
ROUND_NAME = re.compile(r"^(\S+?)\s*Week\s*(\d+)$")
STAT_FIELDS = [
    "sets_played", "attack_total", "attack_points", "block_points", "serve_total", "serve_points",
    "receive_total", "receive_excellent", "dig_total", "dig_excellent", "set_total", "set_excellent",
    "total_points",
]

logger = logging.getLogger(__name__)


def _seasons_payload(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT cup_id, name, start_date, end_date FROM seasons ORDER BY cup_id"
    ).fetchall()
    if not rows:
        raise ValueError("seasons 表是空的，無法判定當季")
    seasons = [
        {"cup_id": cup_id, "name": name, "start_date": start, "end_date": end}
        for cup_id, name, start, end in rows
    ]
    return {"current_cup_id": seasons[-1]["cup_id"], "seasons": seasons}


def _cup_ids_by_team(conn: sqlite3.Connection) -> dict[tuple[int, str], list[int]]:
    rows = conn.execute(
        """
        SELECT team_id, gender, cup_id FROM (
            SELECT home_team_id AS team_id, gender, cup_id FROM matches
            UNION
            SELECT away_team_id AS team_id, gender, cup_id FROM matches
        )
        WHERE team_id IS NOT NULL
        ORDER BY cup_id
        """
    ).fetchall()
    result: dict[tuple[int, str], list[int]] = {}
    for team_id, gender, cup_id in rows:
        result.setdefault((team_id, gender), []).append(cup_id)
    return result


def _teams_payload(conn: sqlite3.Connection) -> dict:
    cup_ids = _cup_ids_by_team(conn)
    rows = conn.execute(
        "SELECT team_id, gender, team_name, short_name, color, is_active FROM teams ORDER BY team_id"
    ).fetchall()
    teams = [
        {
            "team_id": team_id,
            "group": GROUP_BY_GENDER[gender],
            "name": team_name,
            "short_name": short_name or team_name,
            "color": color,
            "is_active": bool(is_active),
            "cup_ids": cup_ids.get((team_id, gender), []),
        }
        for team_id, gender, team_name, short_name, color, is_active in rows
    ]
    teams.sort(key=lambda t: GROUP_ORDER.index(t["group"]))
    return {"teams": teams}


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _new_tally() -> dict:
    return {
        "wins": 0, "losses": 0, "points": 0, "sets_won": 0, "sets_lost": 0,
        "points_for": 0, "points_against": 0, "results": [],
        "set_record": dict.fromkeys(SET_RECORD_KEYS, 0),
    }


def _add_result(tally: dict, own_sets: int, opp_sets: int, own_pts: int, opp_pts: int) -> None:
    won = own_sets > opp_sets
    tally["wins" if won else "losses"] += 1
    tally["points"] += SET_RECORD_POINTS.get((own_sets, opp_sets), 0)
    tally["sets_won"] += own_sets
    tally["sets_lost"] += opp_sets
    tally["points_for"] += own_pts
    tally["points_against"] += opp_pts
    tally["results"].append("W" if won else "L")
    key = f"{own_sets}-{opp_sets}"
    if key in tally["set_record"]:
        tally["set_record"][key] += 1


def _sum_scores(values: list) -> int:
    return sum(v for v in values if v is not None)


def _group_tallies(conn: sqlite3.Connection, cup_id: int, gender: str) -> dict[int, dict]:
    team_ids = conn.execute(
        """
        SELECT home_team_id FROM matches WHERE cup_id = ? AND gender = ? AND is_golden_set = 0
        UNION
        SELECT away_team_id FROM matches WHERE cup_id = ? AND gender = ? AND is_golden_set = 0
        """,
        (cup_id, gender, cup_id, gender),
    ).fetchall()
    tallies = {team_id: _new_tally() for (team_id,) in team_ids if team_id is not None}
    games = conn.execute(
        """
        SELECT home_team_id, away_team_id, home_sets_won, away_sets_won,
               home_set1, home_set2, home_set3, home_set4, home_set5,
               away_set1, away_set2, away_set3, away_set4, away_set5
        FROM matches
        WHERE cup_id = ? AND gender = ? AND is_golden_set = 0 AND round_name LIKE '例行賽%'
          AND home_sets_won IS NOT NULL AND away_sets_won IS NOT NULL
        ORDER BY match_date, game_id
        """,
        (cup_id, gender),
    ).fetchall()
    for home, away, home_sets, away_sets, *scores in games:
        if home is None or away is None:
            continue
        home_pts, away_pts = _sum_scores(scores[:5]), _sum_scores(scores[5:])
        _add_result(tallies[home], home_sets, away_sets, home_pts, away_pts)
        _add_result(tallies[away], away_sets, home_sets, away_pts, home_pts)
    return tallies


def _inf_if_none(value: float | None) -> float:
    return math.inf if value is None else value


def _sort_key(team_id: int, tally: dict):
    return (
        -tally["wins"],
        not tally["results"],
        -tally["points"],
        -_inf_if_none(_ratio(tally["sets_won"], tally["sets_lost"])),
        -_inf_if_none(_ratio(tally["points_for"], tally["points_against"])),
        team_id,
    )


def _standings_row(rank: int, team_id: int, tally: dict) -> dict:
    played = tally["wins"] + tally["losses"]
    return {
        "rank": rank,
        "team_id": team_id,
        "wins": tally["wins"],
        "losses": tally["losses"],
        "win_pct": _rounded(_ratio(tally["wins"], played)),
        "set_record": tally["set_record"],
        "points": tally["points"],
        "sets_won": tally["sets_won"],
        "sets_lost": tally["sets_lost"],
        "set_ratio": _rounded(_ratio(tally["sets_won"], tally["sets_lost"])),
        "points_for": tally["points_for"],
        "points_against": tally["points_against"],
        "point_ratio": _rounded(_ratio(tally["points_for"], tally["points_against"])),
        "last5": tally["results"][-5:],
    }


def _standings_payload(conn: sqlite3.Connection) -> dict:
    cup_ids = [r[0] for r in conn.execute("SELECT cup_id FROM seasons ORDER BY cup_id")]
    standings = []
    for cup_id in cup_ids:
        for gender, group in GROUP_BY_GENDER.items():
            tallies = _group_tallies(conn, cup_id, gender)
            ordered = sorted(tallies, key=lambda t: _sort_key(t, tallies[t]))
            standings.append({
                "cup_id": cup_id,
                "group": group,
                "note": STANDINGS_NOTES.get((cup_id, gender)),
                "rows": [_standings_row(i, t, tallies[t]) for i, t in enumerate(ordered, 1)],
            })
    return {"standings": standings}


def parse_round_name(round_name: str | None) -> tuple[str | None, int | None]:
    if not round_name:
        return None, None
    m = ROUND_NAME.match(round_name.strip())
    if not m:
        return round_name.strip(), None
    return m.group(1), int(m.group(2))


def _weekday(iso_date: str) -> str:
    return WEEKDAYS[date.fromisoformat(iso_date).weekday()]


def _set_pairs(scores: list) -> list[list[int]]:
    home, away = scores[:5], scores[5:]
    return [[h, a] for h, a in zip(home, away) if h is not None and a is not None]


def _match_entry(row: tuple) -> dict:
    (game_id, match_date, start_time, venue, round_name,
     home_id, away_id, home_sets_won, away_sets_won, *scores) = row
    stage, week = parse_round_name(round_name)
    return {
        "game_id": game_id,
        "date": match_date,
        "weekday": _weekday(match_date),
        "start_time": start_time,
        "venue": venue,
        "stage": stage,
        "week": week,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "sets": _set_pairs(scores),
        "home_sets_won": home_sets_won,
        "away_sets_won": away_sets_won,
        "golden_set": None,
    }


def _attach_golden_sets(entries: list[dict], golden_rows: list[tuple]) -> None:
    by_key: dict[tuple, list[dict]] = {}
    for e in entries:
        by_key.setdefault((e["date"], frozenset((e["home_team_id"], e["away_team_id"]))), []).append(e)
    for game_id, match_date, home_id, away_id, home_pts, away_pts in golden_rows:
        candidates = by_key.get((match_date, frozenset((home_id, away_id))), [])
        if len(candidates) != 1:
            raise ValueError(
                f"黃金決勝局 game_id={game_id} 同日同隊的母場次有 {len(candidates)} 場，無法對應"
            )
        parent = candidates[0]
        same_side = parent["home_team_id"] == home_id
        parent["golden_set"] = [home_pts, away_pts] if same_side else [away_pts, home_pts]


def _group_matches(conn: sqlite3.Connection, cup_id: int, gender: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT game_id, match_date, start_time, venue, round_name,
               home_team_id, away_team_id, home_sets_won, away_sets_won,
               home_set1, home_set2, home_set3, home_set4, home_set5,
               away_set1, away_set2, away_set3, away_set4, away_set5
        FROM matches
        WHERE cup_id = ? AND gender = ? AND is_golden_set = 0
        ORDER BY match_date, game_id
        """,
        (cup_id, gender),
    ).fetchall()
    entries = [_match_entry(r) for r in rows]
    golden_rows = conn.execute(
        """
        SELECT game_id, match_date, home_team_id, away_team_id, home_set1, away_set1
        FROM matches
        WHERE cup_id = ? AND gender = ? AND is_golden_set = 1
        """,
        (cup_id, gender),
    ).fetchall()
    _attach_golden_sets(entries, golden_rows)
    return entries


def _matches_payload(conn: sqlite3.Connection) -> dict:
    cup_ids = [r[0] for r in conn.execute("SELECT cup_id FROM seasons ORDER BY cup_id")]
    return {"matches": [
        {"cup_id": cup_id, "group": group, "matches": _group_matches(conn, cup_id, gender)}
        for cup_id in cup_ids
        for gender, group in GROUP_BY_GENDER.items()
    ]}


def _player_seasons(conn: sqlite3.Connection) -> dict[int, list[dict]]:
    sums = ", ".join(f"COALESCE(SUM(s.{f}), 0)" for f in STAT_FIELDS)
    totals = {
        (player_id, cup_id): row
        for player_id, cup_id, *row in conn.execute(
            f"""
            SELECT r.player_id, r.cup_id, COUNT(*), {sums}
            FROM player_match_stats s
            JOIN roster_registrations r ON r.registration_id = s.registration_id
            GROUP BY r.player_id, r.cup_id
            """
        )
    }
    latest = conn.execute(
        """
        SELECT player_id, cup_id, team_id, jersey_number, position
        FROM roster_registrations
        ORDER BY player_id, cup_id, week_start_date IS NULL, week_start_date DESC, registration_id DESC
        """
    ).fetchall()
    seasons: dict[int, list[dict]] = {}
    seen: set[tuple[int, int]] = set()
    for player_id, cup_id, team_id, jersey, position in latest:
        if (player_id, cup_id) in seen:
            continue
        seen.add((player_id, cup_id))
        games, *stats = totals.get((player_id, cup_id), [0] + [0] * len(STAT_FIELDS))
        seasons.setdefault(player_id, []).append({
            "cup_id": cup_id,
            "team_id": team_id,
            "jersey_number": jersey,
            "position": position,
            "games": games,
            **dict(zip(STAT_FIELDS, stats)),
        })
    return seasons


def _players_payload(conn: sqlite3.Connection) -> dict:
    seasons = _player_seasons(conn)
    rows = conn.execute(
        "SELECT player_id, gender, name, dob, height_cm, weight_kg FROM players ORDER BY player_id"
    ).fetchall()
    players = [
        {
            "player_id": player_id,
            "group": GROUP_BY_GENDER[gender],
            "name": name,
            "dob": dob,
            "height_cm": height,
            "weight_kg": weight,
            "seasons": seasons.get(player_id, []),
        }
        for player_id, gender, name, dob, height, weight in rows
    ]
    players.sort(key=lambda p: GROUP_ORDER.index(p["group"]))
    return {"players": players}


def _advanced_scopes(conn: sqlite3.Connection) -> list[tuple[int, str, list[dict]]]:
    """有統計列的球員賽季合計，依 (賽季, 組別) 分池，池內依 player_id 排序。"""
    genders = dict(conn.execute("SELECT player_id, gender FROM players"))
    scopes: dict[tuple[int, str], list[dict]] = {}
    for player_id, seasons in _player_seasons(conn).items():
        for season in seasons:
            if season["games"] == 0:
                continue
            scopes.setdefault((season["cup_id"], genders[player_id]), []).append({"player_id": player_id, **season})
    ordered = sorted(scopes, key=lambda scope: (scope[0], GROUP_ORDER.index(GROUP_BY_GENDER[scope[1]])))
    return [(cup_id, gender, sorted(scopes[(cup_id, gender)], key=lambda r: r["player_id"])) for cup_id, gender in ordered]


def _advanced_entry(cup_id: int, group: str, result: dict) -> dict:
    return {
        "player_id": result["player_id"],
        "cup_id": cup_id,
        "group": group,
        "position": result["position"],
        "games": result["games"],
        "sets_played": result["sets_played"],
        "insufficient_sets": result["insufficient_sets"],
        "ratios": {key: round(value, 2) for key, value in result["ratios"].items()},
        "pr": result["pr"],
    }


def _distribution_entry(cup_id: int, group: str, position: str, metrics: dict[str, list[float]]) -> dict:
    return {
        "cup_id": cup_id,
        "group": group,
        "position": position,
        "player_count": len(metrics["attack_success_rate"]),
        "metrics": {
            key: {"median": round(median(values), 2), "values": [round(v, 2) for v in values]}
            for key, values in metrics.items()
        },
    }


def _player_advanced_payload(conn: sqlite3.Connection) -> dict:
    players: list[dict] = []
    dists: list[dict] = []
    for cup_id, gender, rows in _advanced_scopes(conn):
        group = GROUP_BY_GENDER[gender]
        results = compute_advanced(rows)
        players.extend(_advanced_entry(cup_id, group, r) for r in results)
        dists.extend(
            _distribution_entry(cup_id, group, position, metrics)
            for position, metrics in sorted(distributions(results).items())
        )
    return {"players": players, "distributions": dists}


def _rosters_payload(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        """
        SELECT cup_id, gender, team_id, week_label, week_start_date,
               player_id, jersey_number, position, source
        FROM roster_registrations
        ORDER BY cup_id, gender, team_id, week_start_date IS NULL, week_start_date, week_label,
                 jersey_number IS NULL, jersey_number, player_id
        """
    ).fetchall()
    rosters: dict[tuple, dict] = {}
    weeks: dict[tuple, dict] = {}
    for cup_id, gender, team_id, label, start, player_id, jersey, position, source in rows:
        team_key = (cup_id, gender, team_id)
        if team_key not in rosters:
            rosters[team_key] = {
                "cup_id": cup_id, "group": GROUP_BY_GENDER[gender], "team_id": team_id, "weeks": [],
            }
        week_key = (*team_key, label, start)
        if week_key not in weeks:
            weeks[week_key] = {"week_label": label, "week_start_date": start, "players": []}
            rosters[team_key]["weeks"].append(weeks[week_key])
        weeks[week_key]["players"].append(
            {"player_id": player_id, "jersey_number": jersey, "position": position, "source": source}
        )
    ordered = sorted(rosters.values(), key=lambda r: (r["cup_id"], GROUP_ORDER.index(r["group"]), r["team_id"]))
    return {"rosters": ordered}


def _resolve_opponents(conn: sqlite3.Connection) -> dict[str, tuple[int, str] | None]:
    names = [r[0] for r in conn.execute("SELECT DISTINCT opponent FROM player_match_stats")]
    resolved: dict[str, tuple[int, str] | None] = {}
    for name in names:
        try:
            resolved[name] = resolve_team(conn, name)
        except UnknownTeamError:
            resolved[name] = None
    return resolved


def _parent_matches(conn: sqlite3.Connection) -> dict[tuple, list[tuple]]:
    matches: dict[tuple, list[tuple]] = {}
    for cup_id, game_id, gender, match_date, home_id, away_id in conn.execute(
        """
        SELECT cup_id, game_id, gender, match_date, home_team_id, away_team_id
        FROM matches WHERE is_golden_set = 0 AND home_team_id IS NOT NULL AND away_team_id IS NOT NULL
        """
    ):
        key = (match_date, gender, frozenset((home_id, away_id)))
        matches.setdefault(key, []).append((cup_id, game_id, match_date, home_id))
    return matches


def _match_for_stat(matches: dict, opponents: dict, match_date: str, team_id: int, gender: str,
                    opponent: str) -> tuple[tuple | None, str | None]:
    opp = opponents.get(opponent)
    if opp is None:
        return None, "對手隊名無法解析"
    opp_team_id, opp_gender = opp
    if opp_gender != gender:
        return None, "對手組別與登錄組別不符"
    found = matches.get((match_date, gender, frozenset((team_id, opp_team_id))), [])
    if len(found) != 1:
        return None, f"對不回賽事（符合 {len(found)} 場）"
    return found[0], None


def _stat_entry(player_id: int, team_id: int, jersey, position, *values) -> dict:
    return {
        "player_id": player_id, "team_id": team_id, "jersey_number": jersey, "position": position,
        **dict(zip(STAT_FIELDS, values)),
    }


def _side_rows(game: dict, golden: int, side: str) -> list:
    if not golden:
        return game[side]
    if game["golden_set"] is None:
        game["golden_set"] = {"home": [], "away": []}
    return game["golden_set"][side]


def _box_score_games(conn: sqlite3.Connection) -> dict[tuple[int, str], dict[int, dict]]:
    matches = _parent_matches(conn)
    opponents = _resolve_opponents(conn)
    sums = ", ".join(f"COALESCE(s.{f}, 0)" for f in STAT_FIELDS)
    stat_rows = conn.execute(
        f"""
        SELECT s.stat_id, s.match_date, r.team_id, r.gender, s.opponent, s.is_golden_set,
               r.player_id, r.jersey_number, r.position, {sums}
        FROM player_match_stats s
        JOIN roster_registrations r ON r.registration_id = s.registration_id
        ORDER BY r.jersey_number IS NULL, r.jersey_number, r.player_id, s.stat_id
        """
    ).fetchall()
    games: dict[tuple[int, str], dict[int, dict]] = {}
    for stat_id, match_date, team_id, gender, opponent, golden, player_id, *rest in stat_rows:
        found, reason = _match_for_stat(matches, opponents, match_date, team_id, gender, opponent)
        if found is None:
            logger.warning(
                "統計列排除 stat_id=%s match_date=%s team_id=%s opponent=%s：%s",
                stat_id, match_date, team_id, opponent, reason,
            )
            continue
        cup_id, game_id, match_date, home_id = found
        game = games.setdefault((cup_id, gender), {}).setdefault(
            game_id, {"game_id": game_id, "date": match_date, "home": [], "away": [], "golden_set": None}
        )
        side = "home" if team_id == home_id else "away"
        _side_rows(game, golden, side).append(_stat_entry(player_id, team_id, *rest))
    return games


def _box_scores_payload(conn: sqlite3.Connection) -> dict:
    cup_ids = [r[0] for r in conn.execute("SELECT cup_id FROM seasons ORDER BY cup_id")]
    games = _box_score_games(conn)
    box_scores = []
    for cup_id in cup_ids:
        for gender, group in GROUP_BY_GENDER.items():
            ordered = sorted(games.get((cup_id, gender), {}).values(), key=lambda g: (g["date"], g["game_id"]))
            box_scores.append({
                "cup_id": cup_id,
                "group": group,
                "games": [{k: v for k, v in g.items() if k != "date"} for g in ordered],
            })
    return {"box_scores": box_scores}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_site_data(db_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path=db_path)
    try:
        _write_json(out_dir / "seasons.json", _seasons_payload(conn))
        _write_json(out_dir / "teams.json", _teams_payload(conn))
        _write_json(out_dir / "standings.json", _standings_payload(conn))
        _write_json(out_dir / "matches.json", _matches_payload(conn))
        _write_json(out_dir / "players.json", _players_payload(conn))
        _write_json(out_dir / "player_advanced.json", _player_advanced_payload(conn))
        _write_json(out_dir / "rosters.json", _rosters_payload(conn))
        _write_json(out_dir / "box_scores.json", _box_scores_payload(conn))
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    export_site_data(DB_PATH, DEFAULT_OUT_DIR)
    print(f"已匯出到 {DEFAULT_OUT_DIR}")
