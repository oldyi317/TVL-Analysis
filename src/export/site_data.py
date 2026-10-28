import json
import logging
import re
import sqlite3
from datetime import date, datetime
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

from src.export.player_advanced import compute_advanced, distributions
from src.export.player_seasons import STAT_FIELDS, Registration, SeasonRoster, SeasonTotals, compute_player_seasons
from src.export.standings import compute_standings
from src.utils.db_config import DB_PATH, PROJECT_ROOT, get_connection
from src.utils.match_status import HAS_RESULT_WHERE
from src.utils.teams import UnknownTeamError, resolve_team

DEFAULT_OUT_DIR = PROJECT_ROOT / "web" / "src" / "lib" / "data"
TAIPEI = ZoneInfo("Asia/Taipei")
GROUP_BY_GENDER = {"M": "men", "F": "women"}
GROUP_ORDER = list(GROUP_BY_GENDER.values())
STANDINGS_NOTES: dict[tuple[int, str], str] = {}
WEEKDAYS = "一二三四五六日"
ROUND_NAME = re.compile(r"^(\S+?)\s*Week\s*(\d+)$")

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


def _season_names_by_team(conn: sqlite3.Connection) -> dict[tuple[int, str], dict[int, tuple[str, str]]]:
    rows = conn.execute(
        "SELECT team_id, gender, cup_id, team_name, short_name FROM team_season_names"
    ).fetchall()
    result: dict[tuple[int, str], dict[int, tuple[str, str]]] = {}
    for team_id, gender, cup_id, team_name, short_name in rows:
        result.setdefault((team_id, gender), {})[cup_id] = (team_name, short_name)
    return result


def _names_by_cup(
    cup_ids: list[int], season_names: dict[int, tuple[str, str]], name: str, short_name: str
) -> dict[str, dict]:
    result = {}
    for cup_id in cup_ids:
        season_name, season_short = season_names.get(cup_id, (name, short_name))
        result[str(cup_id)] = {"name": season_name, "short_name": season_short}
    return result


def _teams_payload(conn: sqlite3.Connection) -> dict:
    cup_ids = _cup_ids_by_team(conn)
    season_names = _season_names_by_team(conn)
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
            "names_by_cup": _names_by_cup(
                cup_ids.get((team_id, gender), []),
                season_names.get((team_id, gender), {}),
                team_name,
                short_name or team_name,
            ),
        }
        for team_id, gender, team_name, short_name, color, is_active in rows
    ]
    teams.sort(key=lambda t: GROUP_ORDER.index(t["group"]))
    return {"teams": teams}


def _group_standings(conn: sqlite3.Connection, cup_id: int, gender: str) -> list[dict]:
    team_ids = conn.execute(
        """
        SELECT home_team_id FROM matches WHERE cup_id = ? AND gender = ? AND is_golden_set = 0
        UNION
        SELECT away_team_id FROM matches WHERE cup_id = ? AND gender = ? AND is_golden_set = 0
        """,
        (cup_id, gender, cup_id, gender),
    ).fetchall()
    results = conn.execute(
        f"""
        SELECT home_team_id, away_team_id, home_sets_won, away_sets_won,
               home_set1, home_set2, home_set3, home_set4, home_set5,
               away_set1, away_set2, away_set3, away_set4, away_set5
        FROM matches
        WHERE cup_id = ? AND gender = ? AND is_golden_set = 0 AND round_name LIKE '例行賽%'
          AND {HAS_RESULT_WHERE}
        ORDER BY match_date, game_id
        """,
        (cup_id, gender),
    ).fetchall()
    return compute_standings([t for (t,) in team_ids if t is not None], results)


def _standings_payload(conn: sqlite3.Connection) -> dict:
    cup_ids = [r[0] for r in conn.execute("SELECT cup_id FROM seasons ORDER BY cup_id")]
    standings = []
    for cup_id in cup_ids:
        for gender, group in GROUP_BY_GENDER.items():
            standings.append({
                "cup_id": cup_id,
                "group": group,
                "note": STANDINGS_NOTES.get((cup_id, gender)),
                "rows": _group_standings(conn, cup_id, gender),
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
     home_id, away_id, home_sets_won, away_sets_won, home_win_prob, *scores) = row
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
        "home_win_prob": home_win_prob,
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


def _warn_unresolved_teams(conn: sqlite3.Connection, cup_id: int, gender: str) -> None:
    """隊名尚未進 team_seed 的場次 team_id 為 NULL，匯出略過並警告，不插補也不讓 schema 驗證失敗。"""
    rows = conn.execute(
        """
        SELECT game_id, home_team, away_team FROM matches
        WHERE cup_id = ? AND gender = ? AND (home_team_id IS NULL OR away_team_id IS NULL)
        ORDER BY game_id
        """,
        (cup_id, gender),
    ).fetchall()
    for game_id, home, away in rows:
        logger.warning(
            "隊伍未解析，略過匯出：cup_id=%s gender=%s game_id=%s %s vs %s（請補 sql/team_seed.sql）",
            cup_id, gender, game_id, home, away,
        )


def _group_matches(conn: sqlite3.Connection, cup_id: int, gender: str) -> list[dict]:
    _warn_unresolved_teams(conn, cup_id, gender)
    rows = conn.execute(
        """
        SELECT game_id, match_date, start_time, venue, round_name,
               home_team_id, away_team_id, home_sets_won, away_sets_won,
               (SELECT p.home_win_prob FROM match_predictions p WHERE p.match_id = matches.match_id
                ORDER BY length(p.model_version) DESC, p.model_version DESC, p.computed_at DESC LIMIT 1),
               home_set1, home_set2, home_set3, home_set4, home_set5,
               away_set1, away_set2, away_set3, away_set4, away_set5
        FROM matches
        WHERE cup_id = ? AND gender = ? AND is_golden_set = 0
          AND home_team_id IS NOT NULL AND away_team_id IS NOT NULL
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
          AND home_team_id IS NOT NULL AND away_team_id IS NOT NULL
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
    sums = ", ".join(("CASE WHEN COUNT(s.sets_played) = COUNT(*) THEN SUM(s.sets_played) ELSE NULL END"
                      if f == "sets_played" else f"COALESCE(SUM(s.{f}), 0)") for f in STAT_FIELDS)
    totals = {
        (player_id, cup_id): SeasonTotals(games, tuple(stats))
        for player_id, cup_id, games, *stats in conn.execute(
            f"""
            SELECT r.player_id, r.cup_id, COUNT(*), {sums}
            FROM player_match_stats s
            JOIN roster_registrations r ON r.registration_id = s.registration_id
            GROUP BY r.player_id, r.cup_id
            """
        )
    }
    registrations = [
        Registration(*row)
        for row in conn.execute(
            """
            SELECT player_id, cup_id, team_id, jersey_number, position, week_start_date, registration_id
            FROM roster_registrations
            """
        )
    ]
    season_rosters = [
        SeasonRoster(*row)
        for row in conn.execute("SELECT player_id, cup_id, team_id, jersey_number, position FROM season_rosters")
    ]
    return compute_player_seasons(totals, registrations, season_rosters)


def _players_payload(conn: sqlite3.Connection) -> dict:
    seasons = _player_seasons(conn)
    rows = conn.execute(
        "SELECT player_id, gender, name, height_cm, weight_kg FROM players ORDER BY player_id"
    ).fetchall()
    players = [
        {
            "player_id": player_id,
            "group": GROUP_BY_GENDER[gender],
            "name": name,
            "height_cm": height,
            "weight_kg": weight,
            "seasons": seasons.get(player_id, []),
        }
        for player_id, gender, name, height, weight in rows
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
        "ratios": {key: round(value, 2) if value is not None else None for key, value in result["ratios"].items()},
        "pr": result["pr"],
    }


def _distribution_entry(cup_id: int, group: str, position: str, metrics: dict[str, list[float]]) -> dict:
    return {
        "cup_id": cup_id,
        "group": group,
        "position": position,
        "player_count": len(metrics["attack_success_rate"]),
        "metrics": {
            key: {"median": round(median(values), 2) if values else None, "values": [round(v, 2) for v in values]}
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

    def team_roster(cup_id, gender, team_id):
        team_key = (cup_id, gender, team_id)
        if team_key not in rosters:
            rosters[team_key] = {
                "cup_id": cup_id, "group": GROUP_BY_GENDER[gender], "team_id": team_id,
                "weeks": [], "season_roster": [],
            }
        return rosters[team_key]

    for cup_id, gender, team_id, label, start, player_id, jersey, position, source in rows:
        roster = team_roster(cup_id, gender, team_id)
        week_key = (cup_id, gender, team_id, label, start)
        if week_key not in weeks:
            weeks[week_key] = {"week_label": label, "week_start_date": start, "players": []}
            roster["weeks"].append(weeks[week_key])
        weeks[week_key]["players"].append(
            {"player_id": player_id, "jersey_number": jersey, "position": position, "source": source}
        )
    season_rows = conn.execute(
        """
        SELECT cup_id, gender, team_id, player_id, jersey_number, position
        FROM season_rosters
        ORDER BY cup_id, gender, team_id, jersey_number IS NULL, jersey_number, player_id
        """
    ).fetchall()
    for cup_id, gender, team_id, player_id, jersey, position in season_rows:
        team_roster(cup_id, gender, team_id)["season_roster"].append(
            {"player_id": player_id, "jersey_number": jersey, "position": position}
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
    sums = ", ".join((f"s.{f}" if f == "sets_played" else f"COALESCE(s.{f}, 0)") for f in STAT_FIELDS)
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


def _site_payload() -> dict:
    return {"generated_at": datetime.now(TAIPEI).replace(microsecond=0).isoformat()}


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
        _write_json(out_dir / "site.json", _site_payload())
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    export_site_data(DB_PATH, DEFAULT_OUT_DIR)
    print(f"已匯出到 {DEFAULT_OUT_DIR}")
