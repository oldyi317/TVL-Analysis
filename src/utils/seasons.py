import sqlite3
from datetime import date
from typing import NamedTuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Season(NamedTuple):
    cup_id: int
    name: str
    start_date: date
    end_date: date


def get_season(conn: sqlite3.Connection, cup_id: int) -> Season:
    row = conn.execute(
        "SELECT cup_id, name, start_date, end_date FROM seasons WHERE cup_id = ?",
        (cup_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(
            f"找不到 cup_id={cup_id} 的賽季，請在 sql/schema.sql 的 seasons 種子"
            "與 DB 的 seasons 表新增該賽季。"
        )
    return Season(row[0], row[1], date.fromisoformat(row[2]), date.fromisoformat(row[3]))


def resolve_date_in_season(month: int, day: int, season: Season) -> str | None:
    for year in sorted({season.start_date.year, season.end_date.year}):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if season.start_date <= candidate <= season.end_date:
            return candidate.isoformat()
    logger.warning(
        "cup_id=%d 無法將 %d/%d 對應到賽季區間 %s～%s，已跳過",
        season.cup_id, month, day, season.start_date, season.end_date,
    )
    return None
