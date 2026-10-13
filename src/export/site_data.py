import json
import sqlite3
from pathlib import Path

from src.utils.db_config import DB_PATH, PROJECT_ROOT, get_connection

DEFAULT_OUT_DIR = PROJECT_ROOT / "web" / "src" / "lib" / "data"
GROUP_BY_GENDER = {"M": "men", "F": "women"}
GROUP_ORDER = list(GROUP_BY_GENDER.values())


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


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_site_data(db_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path=db_path)
    try:
        _write_json(out_dir / "seasons.json", _seasons_payload(conn))
        _write_json(out_dir / "teams.json", _teams_payload(conn))
    finally:
        conn.close()


if __name__ == "__main__":
    export_site_data(DB_PATH, DEFAULT_OUT_DIR)
    print(f"已匯出到 {DEFAULT_OUT_DIR}")
