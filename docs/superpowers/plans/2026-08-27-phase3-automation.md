# Phase 3：賽季限定鍵 + 每日自動化 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 根除 week_label 跨賽季碰撞（roster_registrations 加 cup_id 進 UNIQUE），並建立 self-hosted runner 每日自動爬蟲 → `.db` 有變更才 commit/push → Streamlit Cloud 重佈。

**Architecture:** 兩段式——先完成 schema 遷移與全部讀寫路徑的 cup_id 化（Task 1–6），再疊自動化基礎設施（Task 7–10）。SQLite 改 UNIQUE 需重建表，採「rename 舊表 → schema.sql 建新表 → 複製補 cup_id → drop 舊表」模式，registration_id 原值保留使 player_match_stats 外鍵不受影響。

**Tech Stack:** Python 3 + SQLite + pytest；bash script；GitHub Actions（self-hosted runner，WSL systemd service）。

**Spec:** `docs/superpowers/specs/2026-08-27-phase3-automation-design.md`

## Global Constraints

- 工作分支：`phase3-automation`（自 main 分出；Task 1 Step 0 建立）。
- Schema DDL 唯一來源是 `sql/schema.sql`；遷移腳本不得 inline 重寫 `CREATE TABLE roster_registrations`（用 executescript 讀 schema.sql）。
- **Task 1 起至 Task 6 完成前，不得對正式 DB（`data/db/tvl_database.db`）跑任何爬蟲**——程式碼已帶 cup_id 欄位而正式 DB 尚未遷移，跑了會直接 OperationalError。現為淡季，正常情況無人會跑。
- commit message、UI 文案、註解一律繁體中文；行尾 LF；commit 前 `git diff --stat -w` 確認無行尾雜訊。
- 測試用 `venv/bin/python -m pytest`（repo 根目錄執行）。
- DB 連線一律 `src.utils.db_config.get_connection()`（測試 fixture 用 `sqlite3.connect` + `PRAGMA foreign_keys = ON` 現況照舊）。
- 只標記與警告異常，不插補、不竄改原始數據。
- 現有常數：`EXT_CUP_ID = 21`（`src/utils/constants.py:8`）。本計畫所有「當季」語意皆以它為準。

## 查詢面盤點總表（Task 5 的完整改動清單，逐項勾稽）

以下為 grep `roster_registrations` + `week_label` 對 `src/app/`、`src/models/` 的**全部**命中點（行號為改動前）：

| # | 位置 | 現況 | 改法 |
|---|------|------|------|
| Q1 | `src/app/helpers.py:224-239` `get_current_roster` | `week_start_date = MAX(...)` 子查詢，無季限定 | 外層 `AND r.cup_id = ?`、子查詢 `AND cup_id = r.cup_id`（關聯條件，不加參數） |
| Q2 | `src/app/helpers.py:242-288` `get_league_aggregated_stats` | 聚合全部 player_match_stats，無季限定 | WHERE 加 `AND r.cup_id = ?`；latest 子查詢不動（最新登錄天然是當季） |
| Q3 | `src/app/tabs/player_deep.py:40-45`（`_league_position_agg` 內查詢） | `WHERE r.gender = ?`，無季限定 | 加 `AND r.cup_id = ?` |
| Q4 | `src/app/tabs/player_deep.py:74-81`（`render` 球員逐場查詢） | `WHERE r.player_id = ?`，無季限定 | 加 `AND r.cup_id = ?` |
| Q5 | `src/app/tabs/box_score.py:80-89`（比賽選單） | `WHERE r.team_id = ? AND r.gender = ?` | 加 `AND r.cup_id = ?` |
| Q6 | `src/app/tabs/box_score.py:155-175`（Team A box） | 同上 | 加 `AND r.cup_id = ?` |
| Q7 | `src/app/tabs/box_score.py:185-200`（Team B box） | 同上 | 加 `AND r.cup_id = ?` |
| Q8 | `src/app/tabs/match_trend.py:21-28`（球員逐場） | `WHERE r.player_id = ?` | 加 `AND r.cup_id = ?` |

不需改動（已盤點確認）：`src/app/main.py:91`（只查 teams 表）；`src/models/` 無任何命中；`src/app/helpers.py:120,153`（外部系統 HTTP 請求，已帶 `EXT_CUP_ID`）。

---

### Task 1: schema.sql 加 cup_id + 全部測試 fixture 同步

**Files:**
- Modify: `sql/schema.sql:22-35`（roster_registrations DDL）
- Modify: `src/etl/stats_crawler.py:482-502`（`upsert_roster_registration`）
- Test: `tests/test_schema_v2.py`（UNIQUE 測試改寫 + 新增共存測試）
- Modify（fixture 齊步，INSERT 語句加 cup_id 欄）: `tests/test_helpers_phase2_queries.py:67,71,76,112,117,168`、`tests/test_tab_queries_phase2.py:20,25,71,76,117,122`、`tests/test_db_loader_idempotent.py:75`、`tests/test_stats_crawler_dedup.py:28`、`tests/test_stats_crawler_registration.py:28`

**Interfaces:**
- Produces: `roster_registrations` 新 DDL（cup_id INTEGER NOT NULL，UNIQUE 五欄）；`upsert_roster_registration(conn, player_id, row, week_label, week_start_date, source="match_page", cup_id=CUP_ID)`——後續 Task 2/3/4 依賴此簽名。

- [ ] **Step 0: 建立分支**

```bash
git checkout -b phase3-automation
```

- [ ] **Step 1: 寫失敗測試——不同 cup_id 的同名週次可並存**

在 `tests/test_schema_v2.py` 現有 `test_roster_registrations_unique_constraint` 之後新增：

```python
def test_roster_registrations_same_week_label_different_cup_coexists(conn):
    conn.execute("INSERT INTO teams (team_id, team_name, gender) VALUES (1, '測試隊', 'F')")
    conn.execute("INSERT INTO players (name, gender) VALUES ('測試球員', 'F')")
    pid = conn.execute("SELECT player_id FROM players").fetchone()[0]
    for cup in (21, 22):
        conn.execute(
            """INSERT INTO roster_registrations
               (player_id, team_id, gender, cup_id, week_label, jersey_number, position, source)
               VALUES (?, 1, 'F', ?, '例行賽 Week 1', 5, 'OH', 'match_page')""",
            (pid, cup),
        )
    count = conn.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()[0]
    assert count == 2, "不同賽季的同名週次應為兩筆獨立登錄"
```

同時把既有 `test_roster_registrations_unique_constraint`（`tests/test_schema_v2.py:36-52`）兩段 INSERT 的欄位清單改為 `(player_id, team_id, gender, cup_id, week_label, jersey_number, position, source)`、VALUES 補 `21`（衝突鍵仍是同 cup 同週，語意不變）。

- [ ] **Step 2: 跑紅**

Run: `venv/bin/python -m pytest tests/test_schema_v2.py -v`
Expected: 新測試 FAIL（`table roster_registrations has no column named cup_id`）。

- [ ] **Step 3: 改 schema.sql**

`sql/schema.sql:22-35` 改為（gender 之後插入 cup_id、UNIQUE 加 cup_id）：

```sql
CREATE TABLE IF NOT EXISTS roster_registrations (
    registration_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id        INTEGER NOT NULL,
    team_id          INTEGER NOT NULL,
    gender           TEXT    NOT NULL CHECK (gender IN ('M', 'F')),
    cup_id           INTEGER NOT NULL,
    week_label       TEXT    NOT NULL,
    week_start_date  DATE,
    jersey_number    INTEGER,
    position         TEXT,
    source           TEXT    NOT NULL CHECK (source IN ('match_page', 'backfill')),
    FOREIGN KEY (player_id) REFERENCES players (player_id),
    FOREIGN KEY (team_id, gender) REFERENCES teams (team_id, gender),
    UNIQUE (player_id, team_id, gender, cup_id, week_label)
);
```

- [ ] **Step 4: 改 upsert_roster_registration**

`src/etl/stats_crawler.py:482-502` 改為：

```python
def upsert_roster_registration(
    conn: sqlite3.Connection, player_id: int, row: dict,
    week_label: str, week_start_date: str, source: str = "match_page",
    cup_id: int = CUP_ID,
) -> None:
    """upsert 一筆 roster_registrations。source 預設 'match_page'（真實出賽名單）；
    統計寫入路徑查無登錄時會傳入 source='backfill'，補一筆背號/位置皆 NULL 的登錄。
    cup_id 為賽季限定鍵：不同賽季的同名週次是不同登錄，不互相覆寫。"""
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
        (player_id, row["team_id"], row["team_gender"], cup_id, week_label,
         week_start_date, row["jersey_number"], row["position"], source),
    )
```

- [ ] **Step 5: 全部測試 fixture 的 INSERT 補 cup_id**

下列直接 `INSERT INTO roster_registrations` 的測試語句，欄位清單在 `gender` 後加 `cup_id`、VALUES 對應位置補 `21`：

- `tests/test_helpers_phase2_queries.py:67,71,76,112,117,168`
- `tests/test_tab_queries_phase2.py:20,25,71,76,117,122`
- `tests/test_db_loader_idempotent.py:75`
- `tests/test_stats_crawler_dedup.py:28`
- `tests/test_stats_crawler_registration.py:28`

（`tests/test_roster_crawler.py` 走 `upsert_roster_registration` 函式，不需改 INSERT。）

- [ ] **Step 6: 跑綠——全套測試**

Run: `venv/bin/python -m pytest -q`
Expected: 全數通過。若 `test_migrate_to_phase2` 因 `_backfill_registration` 缺 cup_id 而 FAIL，屬預期——立即在本 task 一併修（見 Step 7）；若它通過（該路徑被 mock 掉）則 Step 7 照做（防禦真實執行路徑）。

- [ ] **Step 7: migrate_to_phase2 的 INSERT 同步 cup_id**

`src/etl/migrate_to_phase2.py:46-70` `_backfill_registration` 改為：

```python
def _backfill_registration(conn: sqlite3.Connection, old_player_row: tuple, week_label: str, week_start_date: str) -> int:
    """
    用舊 players_old 的快照（team_id/gender/jersey_number/position）建一筆
    source='backfill' 的登錄記錄。只在 crawl_all_rosters() 抓不到真實出賽
    名單時才會呼叫（例如爬蟲涵蓋範圍外的週次、或該球員該週未被系統記錄）。
    """
    player_id, team_id, gender, jersey_number, position = old_player_row
    cur = conn.execute(
        """
        INSERT INTO roster_registrations
            (player_id, team_id, gender, cup_id, week_label, week_start_date, jersey_number, position, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'backfill')
        ON CONFLICT (player_id, team_id, gender, cup_id, week_label) DO NOTHING
        """,
        (player_id, team_id, gender, CUP_ID, week_label, week_start_date, jersey_number, position),
    )
    if cur.lastrowid and cur.rowcount:
        return cur.lastrowid
    # ON CONFLICT DO NOTHING 命中時要回頭查已存在的那筆
    row = conn.execute(
        """SELECT registration_id FROM roster_registrations
           WHERE player_id = ? AND team_id = ? AND gender = ? AND cup_id = ? AND week_label = ?""",
        (player_id, team_id, gender, CUP_ID, week_label),
    ).fetchone()
    return row[0]
```

（`CUP_ID` 該檔已 import：`migrate_to_phase2.py:11`。）

- [ ] **Step 8: 跑綠 + commit**

Run: `venv/bin/python -m pytest -q`
Expected: 全數通過。

```bash
git add sql/schema.sql src/etl/stats_crawler.py src/etl/migrate_to_phase2.py tests/
git commit -m "feat: roster_registrations 加 cup_id 賽季限定鍵（schema + upsert + 測試 fixture）"
```

---

### Task 2: 統計寫入路徑的 cup_id 解析

**Files:**
- Modify: `src/etl/stats_crawler.py:505-534`（`resolve_registration_for_stats`）、`:537-581`（`crawl_all_rosters` 的 upsert 呼叫）
- Test: `tests/test_stats_crawler_registration.py`

**Interfaces:**
- Consumes: Task 1 的 `upsert_roster_registration(..., cup_id=CUP_ID)`。
- Produces: `resolve_registration_for_stats(conn, player_id, team_id, gender, match_date, cup_id=CUP_ID)`——`main()` 既有呼叫點（`stats_crawler.py:275`）走預設值，不需改。

- [ ] **Step 1: 寫失敗測試——他季同鍵登錄不得誤抓**

在 `tests/test_stats_crawler_registration.py` 新增：

```python
def test_resolve_registration_scopes_by_cup_id(tmp_db_path):
    """他季（cup_id=20）已有同 player/team/week_label 登錄時，
    當季解析不得誤抓他季那筆，應另建當季 backfill。"""
    conn = _make_conn(tmp_db_path)
    try:
        pid = conn.execute("SELECT player_id FROM players").fetchone()[0]
        conn.execute(
            """INSERT INTO roster_registrations
               (player_id, team_id, gender, cup_id, week_label, jersey_number, position, source)
               VALUES (?, 5, 'F', 20, '例行賽 Week 1', 9, 'OH', 'match_page')""",
            (pid,),
        )
        conn.commit()

        rid = resolve_registration_for_stats(conn, pid, 5, "F", "2025-11-01", cup_id=21)

        row = conn.execute(
            "SELECT cup_id, source FROM roster_registrations WHERE registration_id = ?",
            (rid,),
        ).fetchone()
        assert row == (21, "backfill"), "應建立當季新登錄，而非誤用他季登錄"
        count = conn.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()[0]
        assert count == 2
    finally:
        conn.close()
```

- [ ] **Step 2: 跑紅**

Run: `venv/bin/python -m pytest tests/test_stats_crawler_registration.py -v`
Expected: 新測試 FAIL（`resolve_registration_for_stats() got an unexpected keyword argument 'cup_id'`）。

- [ ] **Step 3: 改 resolve_registration_for_stats**

`src/etl/stats_crawler.py:505-534` 改為（簽名加 `cup_id: int = CUP_ID`，兩處 SELECT 與 upsert 呼叫帶 cup_id）：

```python
def resolve_registration_for_stats(
    conn: sqlite3.Connection, player_id: int, team_id: int, gender: str, match_date: str,
    cup_id: int = CUP_ID,
) -> int:
    """
    統計寫入路徑（Player.ashx 逐場數值統計，無背號/位置資訊）解析 registration_id。
    解析順序：match_date -> resolve_week_label() 取 week_label -> 以含 cup_id 的
    五元鍵查 roster_registrations -> 查無則以 source='backfill' 補一筆
    （背號/位置皆 NULL，只標記不插補，不得用其他資料推測填入）。
    """
    week_label, week_start_date = resolve_week_label(conn, match_date, "")

    row = conn.execute(
        """SELECT registration_id FROM roster_registrations
           WHERE player_id = ? AND team_id = ? AND gender = ? AND cup_id = ? AND week_label = ?""",
        (player_id, team_id, gender, cup_id, week_label),
    ).fetchone()
    if row:
        return row[0]

    upsert_roster_registration(
        conn, player_id,
        {"team_id": team_id, "team_gender": gender, "jersey_number": None, "position": None},
        week_label, week_start_date, source="backfill", cup_id=cup_id,
    )
    row = conn.execute(
        """SELECT registration_id FROM roster_registrations
           WHERE player_id = ? AND team_id = ? AND gender = ? AND cup_id = ? AND week_label = ?""",
        (player_id, team_id, gender, cup_id, week_label),
    ).fetchone()
    return row[0]
```

- [ ] **Step 4: crawl_all_rosters 傳遞 cup_id**

`src/etl/stats_crawler.py:575` 的呼叫改為：

```python
            upsert_roster_registration(conn, player_id, row, week_label, week_start_date, cup_id=cup_id)
```

- [ ] **Step 5: 跑綠 + commit**

Run: `venv/bin/python -m pytest tests/test_stats_crawler_registration.py tests/test_roster_crawler.py -v` 然後 `venv/bin/python -m pytest -q`
Expected: 全數通過。

```bash
git add src/etl/stats_crawler.py tests/test_stats_crawler_registration.py
git commit -m "feat: 統計寫入與名單爬蟲路徑以 cup_id 五元鍵解析登錄"
```

---

### Task 3: build_name_to_pid 改 (name, gender) 鍵

**Files:**
- Modify: `src/etl/stats_crawler.py:62-65`（`build_name_to_pid`）、`:229-249`（`main()` 查找/寫回）、`:563-572`（`crawl_all_rosters` 查找/寫回）
- Test: `tests/test_roster_crawler.py`

**Interfaces:**
- Produces: `build_name_to_pid(conn) -> dict[tuple[str, str], int]`，鍵為 `(normalize_name(name), gender)`。呼叫端一律以 `(norm, gender)` 查找。

- [ ] **Step 1: 寫失敗測試**

在 `tests/test_roster_crawler.py` 新增：

```python
def test_build_name_to_pid_distinguishes_gender(tmp_db_path):
    """同名不同性別是兩位球員，查找表不得互相覆蓋。"""
    from pathlib import Path as _P
    from src.etl.stats_crawler import build_name_to_pid

    schema_sql = (_P(__file__).resolve().parents[1] / "sql" / "schema.sql").read_text(encoding="utf-8")
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript(schema_sql)
    conn.execute("INSERT INTO players (name, gender) VALUES ('陳大文', 'M')")
    conn.execute("INSERT INTO players (name, gender) VALUES ('陳大文', 'F')")
    conn.commit()

    name_map = build_name_to_pid(conn)

    pid_m = conn.execute("SELECT player_id FROM players WHERE gender = 'M'").fetchone()[0]
    pid_f = conn.execute("SELECT player_id FROM players WHERE gender = 'F'").fetchone()[0]
    assert name_map[("陳大文", "M")] == pid_m
    assert name_map[("陳大文", "F")] == pid_f
    conn.close()
```

- [ ] **Step 2: 跑紅**

Run: `venv/bin/python -m pytest tests/test_roster_crawler.py::test_build_name_to_pid_distinguishes_gender -v`
Expected: FAIL（KeyError：現版鍵是純姓名字串）。

- [ ] **Step 3: 改 build_name_to_pid 與兩個呼叫端**

`src/etl/stats_crawler.py:62-65` 改為：

```python
def build_name_to_pid(conn: sqlite3.Connection) -> dict[tuple[str, str], int]:
    """建立 {(正規化姓名, gender): player_id} 的查找表。
    同名不同性別是不同球員；同名同性別仍會碰撞（已知限制，罕見）。"""
    rows = conn.execute("SELECT player_id, name, gender FROM players").fetchall()
    return {(normalize_name(name), gender): pid for pid, name, gender in rows}
```

`main()`（`stats_crawler.py:234` 與 `:249`）改為：

```python
            player_id = name_map.get((norm_name, gender))
```

```python
                name_map[(norm_name, gender)] = player_id
```

`crawl_all_rosters`（`stats_crawler.py:564-572`）改為：

```python
            norm = normalize_name(row["name"])
            player_id = name_map.get((norm, row["team_gender"]))
            if player_id is None:
                cursor = conn.execute(
                    "INSERT INTO players (name, gender) VALUES (?, ?)",
                    (row["name"], row["team_gender"]),
                )
                player_id = cursor.lastrowid
                name_map[(norm, row["team_gender"])] = player_id
                stats["new_players"] += 1
```

- [ ] **Step 4: 跑綠 + commit**

Run: `venv/bin/python -m pytest -q`
Expected: 全數通過。

```bash
git add src/etl/stats_crawler.py tests/test_roster_crawler.py
git commit -m "fix: name→pid 查找表改以 (姓名, 性別) 為鍵，防同名跨性別誤指"
```

---

### Task 4: 遷移腳本 migrate_add_cup_id.py

**Files:**
- Create: `src/etl/migrate_add_cup_id.py`
- Test: `tests/test_migrate_add_cup_id.py`

**Interfaces:**
- Consumes: `backup_database()`（`src/etl/backup_db.py:16`）、`get_connection()`、`sql/schema.sql`（Task 1 後含 cup_id 的版本）。
- Produces: `run_migration(conn, cup_id=CUP_ID) -> dict`；`python -m src.etl.migrate_add_cup_id` 入口（Task 6 執行用）。

- [ ] **Step 1: 寫失敗測試**

新檔 `tests/test_migrate_add_cup_id.py`：

```python
import sqlite3
from pathlib import Path

import pytest

from src.etl.migrate_add_cup_id import run_migration

# 遷移前的 v2 schema（無 cup_id）——固定快照，不隨 schema.sql 演進
SCHEMA_V2_OLD = """
CREATE TABLE teams (
    team_id INTEGER NOT NULL, team_name TEXT NOT NULL,
    gender TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    PRIMARY KEY (team_id, gender)
);
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
    gender TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    dob DATE, height_cm REAL, weight_kg REAL
);
CREATE TABLE roster_registrations (
    registration_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL, team_id INTEGER NOT NULL,
    gender TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    week_label TEXT NOT NULL, week_start_date DATE,
    jersey_number INTEGER, position TEXT,
    source TEXT NOT NULL CHECK (source IN ('match_page', 'backfill')),
    FOREIGN KEY (player_id) REFERENCES players (player_id),
    FOREIGN KEY (team_id, gender) REFERENCES teams (team_id, gender),
    UNIQUE (player_id, team_id, gender, week_label)
);
CREATE TABLE player_match_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    registration_id INTEGER NOT NULL, match_date DATE, opponent TEXT,
    sets_played INTEGER, attack_total INTEGER, attack_points INTEGER,
    block_points INTEGER, serve_total INTEGER, serve_points INTEGER,
    receive_total INTEGER, receive_excellent INTEGER, dig_total INTEGER,
    dig_excellent INTEGER, set_total INTEGER, set_excellent INTEGER,
    total_points INTEGER, is_golden_set INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (registration_id) REFERENCES roster_registrations (registration_id)
);
CREATE TABLE matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL,
    gender TEXT NOT NULL, match_date DATE NOT NULL, round_name TEXT,
    home_team TEXT NOT NULL, away_team TEXT NOT NULL,
    UNIQUE (game_id, gender)
);
CREATE INDEX idx_roster_player      ON roster_registrations(player_id);
CREATE INDEX idx_roster_team_gender ON roster_registrations(team_id, gender);
CREATE INDEX idx_roster_week        ON roster_registrations(week_label);
"""


def _seed_old_db(tmp_db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_V2_OLD)
    conn.execute("INSERT INTO teams VALUES (5, '新北中纖', 'F')")
    conn.execute("INSERT INTO players (name, gender) VALUES ('張瓈文', 'F')")
    conn.execute(
        "INSERT INTO roster_registrations "
        "(registration_id, player_id, team_id, gender, week_label, week_start_date, jersey_number, position, source) "
        "VALUES (77, 1, 5, 'F', '例行賽 Week 1', '2025-11-01', 2, 'OP', 'match_page')"
    )
    conn.execute(
        "INSERT INTO player_match_stats (registration_id, match_date, opponent, total_points) "
        "VALUES (77, '2025-11-01', '義力營造', 20)"
    )
    conn.commit()
    return conn


def test_migration_adds_cup_id_and_preserves_registration_id(tmp_db_path):
    conn = _seed_old_db(tmp_db_path)
    try:
        result = run_migration(conn, cup_id=21)

        row = conn.execute(
            "SELECT registration_id, cup_id, week_label, jersey_number FROM roster_registrations"
        ).fetchone()
        assert row == (77, 21, "例行賽 Week 1", 2), "registration_id 必須原值保留、cup_id 全補 21"
        assert result["registrations_migrated"] == 1

        fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        assert fk_errors == [], "遷移後不得有外鍵孤兒"

        stat = conn.execute("SELECT registration_id FROM player_match_stats").fetchone()
        assert stat == (77,)
    finally:
        conn.close()


def test_migration_recreates_indexes(tmp_db_path):
    conn = _seed_old_db(tmp_db_path)
    try:
        run_migration(conn, cup_id=21)
        idx = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='roster_registrations'"
                " AND name LIKE 'idx_%'"
            )
        }
        assert {"idx_roster_player", "idx_roster_team_gender", "idx_roster_week"} <= idx
    finally:
        conn.close()


def test_migration_refuses_rerun(tmp_db_path):
    conn = _seed_old_db(tmp_db_path)
    try:
        run_migration(conn, cup_id=21)
        with pytest.raises(RuntimeError, match="已有 cup_id"):
            run_migration(conn, cup_id=21)
    finally:
        conn.close()
```

- [ ] **Step 2: 跑紅**

Run: `venv/bin/python -m pytest tests/test_migrate_add_cup_id.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'src.etl.migrate_add_cup_id'`）。

- [ ] **Step 3: 寫遷移腳本**

新檔 `src/etl/migrate_add_cup_id.py`：

```python
"""
Phase 3 一次性遷移：roster_registrations 加 cup_id 賽季限定鍵。
SQLite 無法 ALTER 既有 UNIQUE，須重建表；registration_id 原值保留，
player_match_stats 的外鍵不受影響。執行前自動備份（backup_db.py）。
"""

import sqlite3

from src.etl.backup_db import backup_database
from src.utils.constants import EXT_CUP_ID as CUP_ID
from src.utils.db_config import PROJECT_ROOT, get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


def _assert_not_migrated(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(roster_registrations)")}
    if "cup_id" in cols:
        raise RuntimeError("roster_registrations 已有 cup_id，請勿重複執行。")


def run_migration(conn: sqlite3.Connection, cup_id: int = CUP_ID) -> dict:
    _assert_not_migrated(conn)
    expected = conn.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()[0]

    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        # legacy 模式：RENAME 不改寫 player_match_stats 的 FK 參照名稱，
        # 舊表 drop、新表補位後，FK 仍指向 roster_registrations 本名。
        conn.execute("PRAGMA legacy_alter_table = ON")
        conn.execute("ALTER TABLE roster_registrations RENAME TO roster_registrations_old")
        conn.execute("PRAGMA legacy_alter_table = OFF")

        # 舊索引名稱仍佔用（附掛在改名後的舊表），先清掉，
        # schema.sql 的 CREATE INDEX IF NOT EXISTS 才會建到新表上。
        conn.execute("DROP INDEX IF EXISTS idx_roster_player")
        conn.execute("DROP INDEX IF EXISTS idx_roster_team_gender")
        conn.execute("DROP INDEX IF EXISTS idx_roster_week")

        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

        conn.execute(
            """
            INSERT INTO roster_registrations
                (registration_id, player_id, team_id, gender, cup_id,
                 week_label, week_start_date, jersey_number, position, source)
            SELECT registration_id, player_id, team_id, gender, ?,
                   week_label, week_start_date, jersey_number, position, source
            FROM roster_registrations_old
            """,
            (cup_id,),
        )

        actual = conn.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()[0]
        if actual != expected:
            raise RuntimeError(f"筆數不符：預期 {expected}，實際 {actual}，未清除舊表，可安全重查。")

        orphans = conn.execute("""
            SELECT COUNT(*) FROM player_match_stats s
            WHERE NOT EXISTS (
                SELECT 1 FROM roster_registrations r WHERE r.registration_id = s.registration_id
            )
        """).fetchone()[0]
        if orphans > 0:
            raise RuntimeError(f"遷移驗證失敗：{orphans} 筆孤兒 player_match_stats，未清除舊表。")

        conn.execute("DROP TABLE roster_registrations_old")
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")

    logger.info("cup_id 遷移完成：%d 筆登錄補上 cup_id=%d", actual, cup_id)
    return {"registrations_migrated": actual, "cup_id": cup_id, "orphans_found": 0}


def main():
    backup_database()
    conn = get_connection()
    try:
        result = run_migration(conn)
        print("\n===== cup_id 遷移完成 =====")
        for k, v in result.items():
            print(f"{k}: {v}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑綠 + commit**

Run: `venv/bin/python -m pytest tests/test_migrate_add_cup_id.py -v` 然後 `venv/bin/python -m pytest -q`
Expected: 全數通過。

```bash
git add src/etl/migrate_add_cup_id.py tests/test_migrate_add_cup_id.py
git commit -m "feat: cup_id 遷移腳本——重建 roster_registrations 並保留 registration_id"
```

---

### Task 5: 查詢面 cup_id 限定（Q1–Q8）+ DEFAULT_YEAR 修正

**Files:**
- Modify: `src/app/helpers.py`（Q1、Q2、`:135` 硬編 2026）
- Modify: `src/app/tabs/player_deep.py`（Q3、Q4）、`src/app/tabs/box_score.py`（Q5、Q6、Q7）、`src/app/tabs/match_trend.py`（Q8）
- Test: `tests/test_helpers_phase2_queries.py`、`tests/test_tab_queries_phase2.py`

**Interfaces:**
- Consumes: `EXT_CUP_ID`（helpers.py 已 import；tabs 檔需新增 `from src.utils.constants import EXT_CUP_ID`）。
- Produces: 對外函式簽名皆不變（cup_id 從常數帶入，不進參數）。

- [ ] **Step 1: 寫失敗測試——他季資料不得混入聚合**

`tests/test_helpers_phase2_queries.py` 的既有模式是：把 helper 的 SQL **複製成模組常數**（`GET_CURRENT_ROSTER_SQL`、`GET_LEAGUE_AGGREGATED_STATS_SQL`，檔頭 7-56 行）直接對 seeded connection 執行，無 streamlit monkeypatch。因此本步分兩件事：

(a) 先把該檔兩個 SQL 常數改成 **cup_id 版**（與 Step 3 要改的 helpers.py 新 SQL 逐字一致——`GET_CURRENT_ROSTER_SQL` 外層加 `AND r.cup_id = ?`、子查詢加 `AND cup_id = r.cup_id`；`GET_LEAGUE_AGGREGATED_STATS_SQL` 的 `WHERE latest.gender = ?` 改 `WHERE latest.gender = ? AND r.cup_id = ?`），既有測試的 `conn.execute(...)` 呼叫參數尾端補 `21`。

(b) 新增跨季測試：

```python
def test_aggregated_stats_excludes_other_seasons(tmp_path):
    """他季（cup_id=20）的登錄與統計不得混入當季聚合。"""
    tmp_db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(tmp_db_path))
    pid_a, pid_b = _seed(conn)
    # 幫球員A掛一筆當季統計（達到 HAVING 門檻）
    rid_now = conn.execute(
        "SELECT registration_id FROM roster_registrations WHERE player_id = ? AND week_label = '例行賽 Week 2'",
        (pid_a,),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO player_match_stats (registration_id, match_date, opponent, sets_played, total_points) "
        "VALUES (?, '2025-11-08', '義力營造', 5, 10)", (rid_now,),
    )
    # 他季（cup_id=20）同名週次登錄 + 一筆 99 局的統計
    conn.execute(
        "INSERT INTO roster_registrations (player_id, team_id, gender, cup_id, week_label, week_start_date, jersey_number, position, source) "
        "VALUES (?, 5, 'F', 20, '例行賽 Week 2', '2024-11-08', 2, 'OP', 'match_page')", (pid_a,),
    )
    rid_old = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO player_match_stats (registration_id, match_date, opponent, sets_played, total_points) "
        "VALUES (?, '2024-11-08', '義力營造', 99, 99)", (rid_old,),
    )
    conn.commit()

    rows = conn.execute(GET_LEAGUE_AGGREGATED_STATS_SQL, ("F", 21)).fetchall()

    # 只該有球員A一列，且 total_sets 為當季的 5，而非混入他季後的 104
    assert len(rows) == 1
    assert rows[0][4] == 5, f"他季統計混入了當季聚合：total_sets={rows[0][4]}"
    conn.close()
```

（注意：`_seed` 的 INSERT 在 Task 1 已補 `cup_id=21`；此測試的 `(\"F\", 21)` 參數順序須與 (a) 改後的 SQL 佔位符一致。）

在 `tests/test_tab_queries_phase2.py` 新增同構案例：該檔同樣以 SQL 副本模式測 tab 查詢——複製其比賽選單測試（`test_match_selector_query_returns_distinct_matches_per_team_and_date`）的 seed，將選單 SQL 副本加上 `AND r.cup_id = ?`，種一筆 cup_id=20 的他季登錄＋統計，斷言以 `(team_id, gender, 21)` 查詢只回當季場次。

- [ ] **Step 2: 跑紅**

Run: `venv/bin/python -m pytest tests/test_helpers_phase2_queries.py tests/test_tab_queries_phase2.py -v`
Expected: 新測試 FAIL（他季資料被混入）。

- [ ] **Step 3: 逐項改 Q1–Q8**

Q1 `helpers.py get_current_roster`——SQL 改為：

```sql
        SELECT r.player_id, r.jersey_number, p.name, r.position
        FROM roster_registrations r
        JOIN players p ON r.player_id = p.player_id
        WHERE r.team_id = ? AND r.gender = ? AND r.cup_id = ?
          AND r.week_start_date = (
              SELECT MAX(week_start_date) FROM roster_registrations
              WHERE team_id = r.team_id AND gender = r.gender AND cup_id = r.cup_id
          )
        ORDER BY r.jersey_number IS NULL, r.jersey_number
```

參數改 `(team_id, gender_code, EXT_CUP_ID)`。

Q2 `helpers.py get_league_aggregated_stats`——`WHERE latest.gender = ?` 改為 `WHERE latest.gender = ? AND r.cup_id = ?`，參數改 `(gender_code, EXT_CUP_ID)`。latest 子查詢不動。

Q3–Q8（六個 tab 查詢）——各查詢的 WHERE 子句加 `AND r.cup_id = ?`，對應 `load_data` 參數 tuple 尾端加 `EXT_CUP_ID`；`player_deep.py`、`box_score.py`、`match_trend.py` 檔頭 import 加 `EXT_CUP_ID`（若該檔尚未 import）。逐項對照本計畫開頭的「查詢面盤點總表」勾稽，八項缺一不可。

`helpers.py:135`——`year = SEASON_YEAR_MAP.get(month, 2026)` 改為 `year = SEASON_YEAR_MAP.get(month, DEFAULT_YEAR)`，並在 `helpers.py:16-18` 的 constants import 清單加 `DEFAULT_YEAR`。

- [ ] **Step 4: 跑綠 + commit**

Run: `venv/bin/python -m pytest -q`
Expected: 全數通過。

```bash
git add src/app/ tests/test_helpers_phase2_queries.py tests/test_tab_queries_phase2.py
git commit -m "feat: 儀表板八處查詢加 cup_id 當季限定，helpers 年份 fallback 改用 DEFAULT_YEAR"
```

---

### Task 6: 正式 DB 遷移執行

**Files:**
- Modify: `data/db/tvl_database.db`（遷移產物，刻意 commit）

**Interfaces:**
- Consumes: Task 4 的 `python -m src.etl.migrate_add_cup_id`。

- [ ] **Step 1: 遷移前快照**

```bash
venv/bin/python - <<'EOF'
from src.utils.db_config import get_connection
conn = get_connection()
for t in ("roster_registrations", "player_match_stats", "players"):
    print(t, conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
conn.close()
EOF
```

記下三個數字（預期 roster_registrations 約 2,034、player_match_stats 3,807）。

- [ ] **Step 2: 執行遷移（腳本內含自動備份）**

Run: `venv/bin/python -m src.etl.migrate_add_cup_id`
Expected: 印出 `cup_id 遷移完成`、`registrations_migrated` 等於 Step 1 的登錄筆數、`orphans_found: 0`。備份檔 `data/db/tvl_database.db.bak-<timestamp>` 生成（留本機，勿 commit——`.gitignore` 若未涵蓋 `*.bak-*` 則不 add 即可）。

- [ ] **Step 3: 遷移後驗證**

```bash
venv/bin/python - <<'EOF'
from src.utils.db_config import get_connection
conn = get_connection()
print("筆數:", conn.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()[0])
print("cup_id 分布:", conn.execute("SELECT cup_id, COUNT(*) FROM roster_registrations GROUP BY cup_id").fetchall())
print("FK check:", conn.execute("PRAGMA foreign_key_check").fetchall())
print("stats 筆數:", conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0])
conn.close()
EOF
```

Expected: 筆數與 Step 1 一致；cup_id 分布只有 `(21, <全部筆數>)`；FK check 空；stats 筆數不變。

- [ ] **Step 4: 冒煙測 app**

Run: `venv/bin/python -m pytest -q`（全綠）。可另 `streamlit run src/app/main.py` 手動點過五個 tab 確認無紅頁。

- [ ] **Step 5: commit 遷移後 DB**

```bash
git add data/db/tvl_database.db
git diff --stat -w --cached
git commit -m "feat: 正式 DB 完成 cup_id 遷移（登錄筆數守恆、孤兒 0）"
```

---

### Task 7: scripts/daily_update.sh

**Files:**
- Create: `scripts/daily_update.sh`

**Interfaces:**
- Consumes: 三支爬蟲 module 入口（`src.etl.match_crawler`、`src.etl.stats_crawler --rosters`、`--incremental`）。
- Produces: `bash scripts/daily_update.sh`；環境變數 `DRY_RUN=1`（跑爬蟲但不 commit/push）、`PYTHON`（覆寫直譯器，預設 `$HOME/venvs/tvl/bin/python`）。Task 8 的 workflow 呼叫它。

- [ ] **Step 1: 寫 script**

新檔 `scripts/daily_update.sh`：

```bash
#!/usr/bin/env bash
# TVL 每日增量更新：爬蟲 → .db 有變更才 commit/push（觸發 Streamlit Cloud 重佈）
# 本機驗證：DRY_RUN=1 bash scripts/daily_update.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$HOME/venvs/tvl/bin/python}"

"$PYTHON" -m src.etl.match_crawler
"$PYTHON" -m src.etl.stats_crawler --rosters
"$PYTHON" -m src.etl.stats_crawler --incremental

if [ -z "$(git status --porcelain data/db/tvl_database.db)" ]; then
    echo "資料庫無變更，今日無新資料。"
    exit 0
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN：資料庫有變更，略過 commit/push。"
    git status --porcelain data/db/tvl_database.db
    exit 0
fi

git add data/db/tvl_database.db
git -c user.name="github-actions[bot]" \
    -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
    commit -m "chore: 每日自動更新 $(date +%F)"
git push
```

```bash
chmod +x scripts/daily_update.sh
```

- [ ] **Step 2: 本機 dry-run 驗證**

Run: `DRY_RUN=1 bash scripts/daily_update.sh`
Expected: 三支爬蟲依序執行完（淡季應為增量 no-op），結尾印「資料庫無變更」或 DRY_RUN 略過訊息；**不產生任何 commit**（`git log -1` 確認）。若爬蟲對外部系統連線失敗，記錄輸出、排查後重跑（script 應以非零碼中止——`set -e` 保證）。

注意：此步會真的動到本機 DB 的工作副本；跑完 `git status` 確認 `.db` 無 diff（淡季預期），若有 diff 需檢視內容是否合理（可能是外部系統資料修正），不合理則 `git checkout -- data/db/tvl_database.db` 還原並排查。

- [ ] **Step 3: commit**

```bash
git add scripts/daily_update.sh
git commit -m "feat: 每日增量更新 script——爬蟲三連跑，.db 有變更才 commit/push"
```

---

### Task 8: workflow + runner 安裝精靈

**Files:**
- Create: `.github/workflows/daily-crawl.yml`
- Create: `scripts/setup_runner.sh`

**Interfaces:**
- Consumes: Task 7 的 `scripts/daily_update.sh`。
- Produces: 排程 workflow（merge 進 main 後才生效）；runner 安裝精靈（Task 10 由使用者在 WSL 執行）。

- [ ] **Step 1: 寫 workflow**

新檔 `.github/workflows/daily-crawl.yml`：

```yaml
# 每日增量爬蟲：台灣 09:00（01:00 UTC）排程。
# 跑在 self-hosted runner（使用者 WSL；官網有 Cloudflare，雲端 runner 進不去）。
# 機器未開機時 job 會排隊等 runner 上線（至多 24 小時），開機即補跑。
name: daily-crawl
on:
  schedule:
    - cron: "0 1 * * *"
  workflow_dispatch:

concurrency:
  group: daily-crawl
  cancel-in-progress: false

jobs:
  update:
    runs-on: [self-hosted, tvl]
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - name: 每日增量更新
        run: bash scripts/daily_update.sh
```

（`actions/checkout` 預設 `persist-credentials: true`，script 內的 `git push` 用它的 GITHUB_TOKEN 憑證；GITHUB_TOKEN 的 push 不會再觸發其他 GitHub workflow，而 Streamlit Cloud 是自己輪詢 repo，不受此限，照常重佈。）

- [ ] **Step 2: 寫 runner 安裝精靈**

新檔 `scripts/setup_runner.sh`：

```bash
#!/usr/bin/env bash
# self-hosted runner 一次性安裝精靈（WSL 內執行）
# 前置需求：gh 已登入、sudo 權限、WSL 已啟用 systemd（ps -p 1 顯示 systemd）
set -euo pipefail

REPO="oldyi317/TVL-Analysis"
RUNNER_DIR="$HOME/actions-runner"

step() { echo; echo "── $1 ──"; read -rp "按 Enter 繼續（Ctrl+C 中止）..."; }

echo "TVL self-hosted runner 安裝精靈"

step "步驟 1/5：檢查 gh 登入與 systemd"
gh auth status
[ "$(ps -p 1 -o comm=)" = "systemd" ] || { echo "WSL 未啟用 systemd，請先在 /etc/wsl.conf 開啟後 wsl --shutdown 重進"; exit 1; }

step "步驟 2/5：下載 actions-runner 到 $RUNNER_DIR"
mkdir -p "$RUNNER_DIR" && cd "$RUNNER_DIR"
VER=$(gh api repos/actions/runner/releases/latest --jq '.tag_name' | tr -d v)
curl -fL -o runner.tar.gz \
  "https://github.com/actions/runner/releases/download/v${VER}/actions-runner-linux-x64-${VER}.tar.gz"
tar xzf runner.tar.gz && rm runner.tar.gz

step "步驟 3/5：向 repo 註冊 runner（label: tvl）"
TOKEN=$(gh api -X POST "repos/${REPO}/actions/runners/registration-token" --jq '.token')
./config.sh --url "https://github.com/${REPO}" --token "$TOKEN" \
  --name tvl-wsl --labels tvl --unattended

step "步驟 4/5：安裝並啟動 systemd service（需要 sudo）"
sudo ./svc.sh install "$USER"
sudo ./svc.sh start
sudo ./svc.sh status || true

step "步驟 5/5：Windows 端設定（手動）"
cat <<'EOF'
在 Windows 設定「登入時自動啟動 WSL」，讓 runner 開機即上線：
1. 開始功能表搜尋「工作排程器」→ 建立基本工作
2. 名稱：Start WSL for TVL runner；觸發程序：當我登入時
3. 動作：啟動程式；程式：wsl.exe；引數：-d <發行版名稱> --exec /bin/true
   （發行版名稱在 PowerShell 跑 `wsl -l -q` 查）
4. 完成。runner 的 systemd service 隨 WSL 啟動，常駐進程使 WSL 不被閒置回收。
EOF
echo "安裝完成。到 https://github.com/${REPO}/settings/actions/runners 確認 runner 顯示 Idle。"
```

```bash
chmod +x scripts/setup_runner.sh
```

- [ ] **Step 3: 語法驗證 + commit**

Run: `bash -n scripts/setup_runner.sh && bash -n scripts/daily_update.sh`（語法檢查）；workflow 的 yaml 以 `venv/bin/python -c "import yaml, sys; yaml.safe_load(open('.github/workflows/daily-crawl.yml'))"` 驗證可解析。
Expected: 無輸出（皆通過）。

```bash
git add .github/workflows/daily-crawl.yml scripts/setup_runner.sh
git commit -m "feat: 每日排程 workflow 與 self-hosted runner 安裝精靈"
```

---

### Task 9: 維運文件

**Files:**
- Create: `docs/ops/season-switch.md`
- Modify: `CLAUDE.md`（地雷條目改寫 + 指標）

- [ ] **Step 1: 寫換季 checklist**

新檔 `docs/ops/season-switch.md`：

```markdown
# 換季手動 Checklist

每年新賽季開打前逐項執行。未完成前每日排程照跑但抓不到新季資料（EXT_CUP_ID
仍指舊季），屬預期行為，不需先停 workflow。

## 常數更新（src/utils/constants.py 為主）

- [ ] `EXT_CUP_ID`（constants.py）：到 `http://114.35.229.141/Match.aspx` 逐一
      試 CupID 找到新季編號。
- [ ] `SEASON_YEAR_MAP` 與 `DEFAULT_YEAR`（constants.py）：改成新賽季的
      「11、12 月屬 X 年，其餘屬 X+1 年」。
- [ ] `match_crawler.py` 的 `--range-start`/`--range-end` argparse 預設值：
      新季 game_id 區間要實際到官網賽程頁確認起始編號。
- [ ] 隊伍對照表四份（有新隊伍/改名才需要動）：`EXT_TEAM_MAP`、
      `OPP_SHORT_TO_TEAM`、`TEAM_NAME_SHORT`（皆在 constants.py）、
      `TEAM_ALIAS`（match_crawler.py）。

## 季前作業

- [ ] 跑 `python -m src.etl.crawler` 抓新季官網名單 → `data/raw/`。
- [ ] 跑 `python -m src.etl.db_loader` 更新 players 身分層與 teams。
- [ ] 手動 `workflow_dispatch` 觸發一次 daily-crawl 驗證整條管道，
      確認新季第一批資料正確落庫（cup_id 應為新季編號）。

## 模型（可延後）

- [ ] 新季累積足夠場次後重訓 match_predictor（見 Phase 4）。
```

- [ ] **Step 2: 更新 CLAUDE.md**

`CLAUDE.md` 的地雷條目（現行第 37-38 行）：

```markdown
- **week_label 直接用 `matches.round_name`，跨賽季同名會碰撞**：Phase 3 開季前
  必須加賽季限定鍵（cup_id/season），下季資料進來前不得先跑爬蟲。
```

改為：

```markdown
- **換季前必跑 `docs/ops/season-switch.md` checklist**：cup_id 已進
  roster_registrations 的 UNIQUE（Phase 3），跨季同名週次不再互相覆寫；但
  `EXT_CUP_ID` 等常數是手動更新的，未更新前爬蟲抓不到新季資料。
```

並在「常用指令」區塊尾端加一行：

```markdown
bash scripts/daily_update.sh                     # 每日增量更新（GitHub Actions 排程同款；DRY_RUN=1 可演練）
```

- [ ] **Step 3: commit**

```bash
git add docs/ops/season-switch.md CLAUDE.md
git commit -m "docs: 換季手動 checklist；CLAUDE.md 地雷條目更新為 cup_id 已修狀態"
```

---

### Task 10: 收尾與端到端驗證

此 task 有兩個使用者參與點（merge 決策、在 WSL 跑安裝精靈），依序執行：

- [ ] **Step 1: 全套驗證**

Run: `venv/bin/python -m pytest -q`（全綠）；`git diff main --stat -w` 確認無行尾雜訊。

- [ ] **Step 2: 分支收尾**

照 superpowers:finishing-a-development-branch 流程與使用者確認整合方式（預期：merge 回 main 並 push——push 即上線 schema 遷移後的 `.db` 並使 workflow 生效；排程 cron 明早才會首跑，先 push 無風險）。

- [ ] **Step 3: 使用者執行 runner 安裝精靈**

在 WSL 終端執行 `bash scripts/setup_runner.sh`，逐步完成（含 Windows 工作排程器那步）。完成判準：repo Settings → Actions → Runners 顯示 `tvl-wsl` 為 **Idle**。

- [ ] **Step 4: 端到端實測（no-op 情境）**

```bash
gh workflow run daily-crawl --ref main
sleep 20 && gh run list --workflow=daily-crawl --limit 1
gh run watch --exit-status "$(gh run list --workflow=daily-crawl --limit 1 --json databaseId --jq '.[0].databaseId')"
```

Expected: run 為 success，log 尾端印「資料庫無變更，今日無新資料。」（淡季 no-op）；repo 無新增 commit。失敗則讀 `gh run view --log` 排查（常見：runner venv 路徑、python 依賴）。

- [ ] **Step 5: 收尾記錄**

確認明日排程（台灣 09:00）後續幾天在 GitHub Actions 頁面留下綠色 no-op 紀錄即為穩定。Phase 3 完成條件：連續兩個排程日 success。
```

---

## 自查（Self-Review）

**1. Spec 覆蓋：**
- spec §1 schema/遷移 → Task 1、4、6 ✓
- spec §2 寫入/查詢/同名/DEFAULT_YEAR → Task 2、3、5 ✓
- spec §3 自動化（script/workflow/runner/授權） → Task 7、8、10 ✓
- spec §4 測試 → 各 task 內嵌 + Task 7 dry-run + Task 10 e2e ✓
- spec §5 維運文件 → Task 9 ✓
- spec 硬約束（遷移先於 workflow 啟用）→ workflow 檔在 Task 8 才建立、且 merge 前不在 main 上，天然滿足 ✓

**2. 佔位符掃描：** Task 5 Step 1 的測試碼引用「該檔既有 helper 命名」屬對既有測試模式的對齊指示（實作者需先讀該檔），非 TBD；其餘任務皆為完整可執行內容。

**3. 型別/命名一致：** `upsert_roster_registration(..., source, cup_id)` 簽名在 Task 1 定義、Task 2 使用一致；`run_migration(conn, cup_id)` 兩個遷移腳本同名但分屬不同模組（沿用 phase2 前例）；`runs-on: [self-hosted, tvl]` 與精靈 `--labels tvl` 一致。
