"""依賽季報名名單比對球員，寫入週登錄與單場數據。cup_id 一律必填。
只讀取既有球員，不新增 players。交易由呼叫者掌控，本模組不 BEGIN、commit 或 rollback。
供各寫入者共用，不匯入呼叫端模組。
"""

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_name(name: str) -> str:
    """正規化姓名：去除全形/半形空白、轉小寫、去除不間斷空白。"""
    return re.sub(r"[\s\u3000\xa0]+", "", name).lower()


class NoSeasonRoster(Exception):
    """該季沒有賽季報名名單，無法比對球員。"""


class SeasonRosterMatcher:
    def __init__(self, conn: sqlite3.Connection, cup_id: int):
        self.conn = conn
        self.cup_id = cup_id
        if conn.execute(
            "SELECT 1 FROM season_rosters WHERE cup_id = ? LIMIT 1", (cup_id,)
        ).fetchone() is None:
            raise NoSeasonRoster(f"cup_id={cup_id} 無賽季報名名單")
        self.matched_by_jersey: dict[str, None] = {}
        self.skipped: dict[str, None] = {}
        self._rosters: dict[tuple[int, str], list[tuple[int, str, int | None]]] = {}

    def _roster(self, team_id: int, gender: str) -> list[tuple[int, str, int | None]]:
        key = (team_id, gender)
        if key not in self._rosters:
            self._rosters[key] = self.conn.execute(
                """SELECT p.player_id, p.name, sr.jersey_number
                   FROM season_rosters sr JOIN players p ON p.player_id = sr.player_id
                   WHERE sr.team_id = ? AND sr.gender = ? AND sr.cup_id = ?""",
                (team_id, gender, self.cup_id),
            ).fetchall()
        return self._rosters[key]

    def _team_label(self, team_id: int, gender: str) -> str:
        row = self.conn.execute(
            "SELECT team_name FROM teams WHERE team_id = ? AND gender = ?", (team_id, gender)
        ).fetchone()
        return row[0] if row else f"team_id={team_id}"

    def match(self, team_id: int, gender: str, name: str, jersey_number: int | None) -> int | None:
        roster = self._roster(team_id, gender)
        norm = normalize_name(name)
        for pid, local_name, _ in roster:
            if normalize_name(local_name) == norm:
                return pid
        team = self._team_label(team_id, gender)
        if jersey_number is not None:
            for pid, local_name, jersey in roster:
                if jersey == jersey_number:
                    self.matched_by_jersey[
                        f"  外部 {name}　本地 {local_name}　{team}　背號 {jersey_number}"
                    ] = None
                    return pid
        logger.warning("[略過] %s 背號 %s 外部球員 %s 對不上賽季報名名單", team, jersey_number, name)
        self.skipped[f"  {team}　背號 {jersey_number}　外部姓名 {name}"] = None
        return None

    def log_reports(self) -> None:
        if self.matched_by_jersey:
            logger.warning("姓名不同、以背號對上（%d 筆）：\n%s", len(self.matched_by_jersey), "\n".join(self.matched_by_jersey))
        else:
            logger.info("姓名不同、以背號對上：無")
        if self.skipped:
            logger.warning("略過（對不上賽季報名名單，%d 筆）：\n%s", len(self.skipped), "\n".join(self.skipped))
        else:
            logger.info("略過：無")


@dataclass(frozen=True, kw_only=True)
class Registration:
    player_id: int
    team_id: int
    gender: str
    cup_id: int
    week_label: str
    week_start_date: str


def _registration_id(conn: sqlite3.Connection, reg: Registration) -> int | None:
    row = conn.execute(
        """SELECT registration_id FROM roster_registrations
           WHERE player_id = ? AND team_id = ? AND gender = ? AND cup_id = ? AND week_label = ?""",
        (reg.player_id, reg.team_id, reg.gender, reg.cup_id, reg.week_label),
    ).fetchone()
    return row[0] if row else None


def _write_registration(
    conn: sqlite3.Connection, reg: Registration, jersey_number: int | None,
    position: str | None, source: str,
) -> int:
    conn.execute(
        """
        INSERT INTO roster_registrations
            (player_id, team_id, gender, cup_id, week_label, week_start_date, jersey_number, position, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (player_id, team_id, gender, cup_id, week_label) DO UPDATE SET
            jersey_number = excluded.jersey_number,
            position = excluded.position,
            week_start_date = excluded.week_start_date,
            source = excluded.source
        """,
        (reg.player_id, reg.team_id, reg.gender, reg.cup_id, reg.week_label,
         reg.week_start_date, jersey_number, position, source),
    )
    rid = _registration_id(conn, reg)
    assert rid is not None
    return rid


def write_page_registration(
    conn: sqlite3.Connection, reg: Registration, *, jersey_number: int | None, position: str | None,
) -> int:
    return _write_registration(conn, reg, jersey_number, position, "match_page")


def fill_missing_registration(conn: sqlite3.Connection, reg: Registration) -> int:
    rid = _registration_id(conn, reg)
    if rid is not None:
        return rid
    row = conn.execute(
        "SELECT jersey_number, position FROM season_rosters WHERE player_id = ? AND gender = ? AND cup_id = ?",
        (reg.player_id, reg.gender, reg.cup_id),
    ).fetchone()
    jersey, position = row if row else (None, None)
    return _write_registration(conn, reg, jersey, position, "backfill")


@dataclass(frozen=True, kw_only=True)
class StatLine:
    match_date: str | None
    opponent: str | None
    sets_played: int | None = None
    attack_total: int | None
    attack_points: int | None
    block_points: int | None
    serve_total: int | None
    serve_points: int | None
    receive_total: int | None
    receive_excellent: int | None
    dig_total: int | None
    dig_excellent: int | None
    set_total: int | None
    set_excellent: int | None
    total_points: int | None
    is_golden_set: int


def write_stat_lines(conn: sqlite3.Connection, rows: Sequence[tuple[int, StatLine]]) -> None:
    if not rows:
        return
    conn.executemany(
        """INSERT INTO player_match_stats
           (registration_id, match_date, opponent, sets_played,
            attack_total, attack_points, block_points, serve_total, serve_points,
            receive_total, receive_excellent, dig_total, dig_excellent,
            set_total, set_excellent, total_points, is_golden_set)
           VALUES (:registration_id, :match_date, :opponent, :sets_played,
                   :attack_total, :attack_points, :block_points, :serve_total, :serve_points,
                   :receive_total, :receive_excellent, :dig_total, :dig_excellent,
                   :set_total, :set_excellent, :total_points, :is_golden_set)""",
        (dict(vars(line), registration_id=rid) for rid, line in rows),
    )
