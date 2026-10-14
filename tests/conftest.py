import json
import sqlite3
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from src.export.site_data import export_site_data
from src.utils.teams import apply_team_seed

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SQL = ROOT / "sql" / "schema.sql"
STAT_COLUMNS = [
    "sets_played", "attack_total", "attack_points", "block_points", "serve_total", "serve_points",
    "receive_total", "receive_excellent", "dig_total", "dig_excellent", "set_total", "set_excellent",
    "total_points",
]


@pytest.fixture
def tmp_db_path(tmp_path) -> Path:
    """回傳隔離的 SQLite 檔案路徑，測試用，不觸碰正式 DB。"""
    return tmp_path / "test.db"


def connect_seeded(db_path):
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
    apply_team_seed(conn)
    return conn


def seed_player(conn, player_id, name, gender="M", dob=None, height=None, weight=None):
    conn.execute(
        "INSERT INTO players (player_id, name, gender, dob, height_cm, weight_kg) VALUES (?, ?, ?, ?, ?, ?)",
        (player_id, name, gender, dob, height, weight),
    )


def seed_registration(conn, reg_id, player_id, team_id, cup_id=21, week="例行賽 Week 1", week_start="2025-11-01",
                      jersey=None, position=None, gender="M", source="match_page"):
    conn.execute(
        """
        INSERT INTO roster_registrations (registration_id, player_id, team_id, gender, cup_id, week_label,
                                          week_start_date, jersey_number, position, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (reg_id, player_id, team_id, gender, cup_id, week, week_start, jersey, position, source),
    )


def seed_stat(conn, reg_id, values, match_date="2025-11-01", opponent="雲林美津濃", golden=0):
    conn.execute(
        f"""
        INSERT INTO player_match_stats (registration_id, match_date, opponent, is_golden_set, {", ".join(STAT_COLUMNS)})
        VALUES (?, ?, ?, ?, {", ".join("?" * len(STAT_COLUMNS))})
        """,
        (reg_id, match_date, opponent, golden, *values),
    )


@pytest.fixture
def export_json(tmp_db_path, tmp_path):
    """建 seed DB → 跑匯出器 → 讀指定 JSON 並用 web/schema 驗證，回傳 (payload, out_dir)。"""
    def run(build, name):
        conn = connect_seeded(tmp_db_path)
        build(conn)
        conn.commit()
        conn.close()
        out_dir = tmp_path / "out"
        export_site_data(tmp_db_path, out_dir)
        payload = json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))
        schema = json.loads((ROOT / "web" / "schema" / f"{name}.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(payload)
        return payload, out_dir
    return run
