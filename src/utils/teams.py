import sqlite3

from src.utils.db_config import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)

TEAM_SEED_PATH = PROJECT_ROOT / "sql" / "team_seed.sql"


_NAME_SOURCE_COND = "source <> 'ext_stats'"


class UnknownTeamError(ValueError):
    pass


def apply_team_seed(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(teams)")}
    if "short_name" not in cols:
        raise RuntimeError("teams 尚未遷移，請先執行 python -m src.etl.migrate_add_team_identity")
    conn.executescript(TEAM_SEED_PATH.read_text(encoding="utf-8"))


def _lookup(conn: sqlite3.Connection, alias: str, source_cond: str) -> tuple[int, str] | None:
    if not alias:
        return None
    row = conn.execute(
        f"SELECT team_id, gender FROM team_aliases WHERE alias = ? AND {source_cond}", (alias,)
    ).fetchone()
    return (row[0], row[1]) if row else None


def resolve_team(conn: sqlite3.Connection, name: str, gender: str | None = None) -> tuple[int, str]:
    key = name.strip() if isinstance(name, str) else ""
    result = _lookup(conn, key, _NAME_SOURCE_COND)
    if result is None:
        raise UnknownTeamError(f"未知隊名：{name!r}（請補 sql/team_seed.sql 的 team_aliases）")
    if gender is not None and result[1] != gender:
        raise UnknownTeamError(f"隊名 {name!r} 性別為 {result[1]}，與預期的 {gender} 不符")
    return result


def resolve_team_id(conn: sqlite3.Connection, name: str, gender: str, context: str = "") -> int | None:
    try:
        return resolve_team(conn, name, gender)[0]
    except UnknownTeamError as e:
        logger.warning("[%s] team_id 寫 NULL：%s", context, e)
        return None


def resolve_ext_team(conn: sqlite3.Connection, ext_team_id: int) -> tuple[int, str]:
    result = _lookup(conn, str(ext_team_id), "source = 'ext_stats'")
    if result is None:
        raise UnknownTeamError(f"未知外部統計系統 TeamID：{ext_team_id!r}（請補 sql/team_seed.sql 的 team_aliases）")
    return result
