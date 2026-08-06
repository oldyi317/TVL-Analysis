# PCAI 搬遷計畫二：Dashboard 改造 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修正 TVL-Analysis Streamlit dashboard 在計畫一遺留的正式運行問題（`load_data` 的 qmark 佔位符對 PostgreSQL 不安全、`st.stop()` 誤用、快取無 TTL、`prediction.py` 讀錯 pkl 鍵名），並將 AI 戰報從 Gemini 改接 PCAI MLIS 的 OpenAI 相容 endpoint，新增可在 UI 設定 endpoint/model/API key 的「系統設定」分頁，最後在 sidebar 加入賽季選擇器，解決跨賽季後聯盟 PR 頁同一人出現兩筆的問題。

**Architecture:** `src/app/helpers.py` 的 `load_data()` 改用 SQLAlchemy `text()` + 具名參數（`dict`），取代原本只在 SQLite 上碰巧能動的 `?` qmark + tuple 寫法，`main.py` 與 5 個 tab 檔案的呼叫端同步更新；新增 `src/app/llm_client.py` 封裝 MLIS 的 OpenAI 相容呼叫（含簡單重試），設定讀取順序為 `src/app/settings_store.py`（新增，讀寫 Postgres/SQLite 皆可的 `app_settings` key-value 表）→ 環境變數 → 皆無則顯示引導訊息；新增第 7 個分頁 `src/app/tabs/settings_tab.py` 提供設定 UI；sidebar 最上層加賽季下拉，選定的 season 經 `ctx` 傳給所有 tab，聯盟聚合與週報查詢皆加上 season 過濾。UI 早退行為的驗證使用 `streamlit.testing.v1.AppTest`（streamlit 1.61 內建，已於本機驗證可行）。

**Tech Stack:** Python 3.11、Streamlit 1.61、SQLAlchemy 2.x Core、openai 2.53.0（OpenAI 相容 client）、pytest、streamlit.testing.v1.AppTest。

## Global Constraints

- Python >= 3.10（程式碼使用 `X | None` 型別語法，禁止改回 `Optional[X]`）
- 介面與資料皆為繁體中文；程式碼、指令、SQL、commit 訊息本體以外一律英文
- 環境變數皆需有預設值（`MLIS_BASE_URL`/`MLIS_API_KEY`/`MLIS_MODEL` 例外——這三者沒有合理預設值，未設定時視為「未設定」）
- 所有 DB 讀寫 SQL 一律用 SQLAlchemy `text()` + 具名參數（`:name`），不得使用 `?` qmark（psycopg 不支援，這是 PostgreSQL 上線的必要條件）
- 本計畫**不修改**任何 ETL 冪等 upsert 邏輯本體（`db_loader.py`/`stats_crawler.py`/`match_crawler.py` 的既有 upsert 語句），除新增 `app_settings` 表與既有查詢加上 season 過濾之外
- 本計畫**不涉及**容器化、Helm chart、Airflow DAG（計畫三範圍，對應 spec §8-10）
- commit 前依專案慣例（`/mnt/d/CLAUDE.md`）需先詢問使用者；本計畫每個 Task 的「Commit」步驟僅列出建議指令，實際執行者仍需在跑 `git commit` 前向使用者確認
- 每個 Task 的程式碼修改完成後立即跑對應 `pytest`，不得攢到最後一次跑
- MLIS 真實 endpoint 無法在本機驗證（叢集尚未部署或本機無法連線）；`llm_client` 的測試一律使用 `httpx.MockTransport` 或注入假 client，不連真實網路；真實 endpoint 的驗證留待 PCAI 上進行
- `streamlit.testing.v1.AppTest` 用於驗證早退行為與賽季選擇器，僅覆蓋本次改動涉及的分支，不追求窮舉每個 `st.stop()`/`return` 分支（沿用計畫一「不追求全面覆蓋，只覆蓋本次改動的部分」的測試策略）
- openai client 建立時一律帶 `max_retries=0`（避免與本專案自訂的重試邏輯疊加造成請求次數倍增——已實測驗證：openai SDK 預設 `max_retries=2`，若不關閉，一次呼叫失敗會變成 3×3=9 次實際 HTTP 請求）

---

## 專案現況（實測基準，2026-08-06）

- `openai` 套件版本：於乾淨虛擬環境 `pip install openai` 實測取得 **`openai==2.53.0`**（連帶 `httpx==0.28.1`、`pydantic==2.13.4`）。
- `openai.OpenAI(base_url=..., api_key=..., http_client=httpx.Client(transport=httpx.MockTransport(handler)))` 已實測可行，可完全不連真實網路測試 `chat.completions.create()` 的成功與失敗路徑（`InternalServerError` 等例外會被正常拋出）。
- `streamlit.testing.v1.AppTest`（streamlit 1.61 內建）已實測三種形式皆可用：
  - `AppTest.from_file(path)`：對本專案的 `main.py` 完整跑一次（含 sidebar 三層連動 + 6 個既有 tab）約需 20–50 秒（冷啟動較慢，暖機後較快），無例外，sidebar 的 `selectbox` 可讀取選項與程式互動（`.select(...).run()`）。
  - `AppTest.from_function(fn)`：把一個「自帶所有 import」的巢狀函式當成整份 script 執行，適合對單一 tab 模組的 `render(ctx)` 做隔離測試，已實測可正確捕捉 `st.stop()` 中止腳本、`st.info`/`st.text` 等元素輸出。
  - `st.form` + `st.form_submit_button` 的按鈕 key 慣例已實測確認為 `FormSubmitter:<form_key>-<button_label>`。
- 目前 `src/app/tabs/*.py` 6 個檔案中共有 **9 處** `st.stop()`（`box_score.py:71,92`、`league_pr.py:38,114,128`、`match_trend.py:28`、`player_deep.py:80`、`weekly_report_tab.py:322,351`）——spec §4d 原文寫「共 8 處」，實際逐檔核對後為 9 處（同一份清單裡列出的座標加總即為 9，應是 spec 撰寫時的算術筆誤，不影響本計畫範圍，以下列出的座標為準）。
- `load_data()` 呼叫端（含 `helpers.py` 內部）加總共有 **11 處**使用 `?` qmark：`main.py`×2、`box_score.py`×4、`match_trend.py`×1、`player_deep.py`×2（`_load_league_agg` 定義處 + 呼叫端邏輯共用同一函式）、`weekly_report_tab.py`×1（`_attach_set_scores` 動態 `IN (...)` 子句）、`helpers.py` 內部的 `get_league_aggregated_stats`×1。Task 1 的步驟本身已完整轉換這 11 處，此處僅修正先前敘述的算術錯誤。
- `src/etl/weekly_report.py` 已在計畫一完成時就使用 `text()` + 具名參數（`gather_weekly_data`/`get_match_weeks`），不在本計畫 Task 1 的轉換範圍內，但 Task 9 會替它們加上 `season` 參數。

---

### Task 1: `helpers.load_data` 具名參數化 + 快取 TTL（對應 spec §4d 遺留項目）

`app/helpers.py` 的 `load_data(query, params=())` 目前用 `?` qmark + tuple，`pandas.read_sql_query` 對純字串查詢會直接把 `?` 丟給底層 DBAPI 驅動；SQLite 原生 paramstyle 剛好是 `qmark` 所以本地能動，但 PostgreSQL 的 `psycopg` 驅動原生 paramstyle 是 `pyformat`，不認得 `?`，`DATABASE_URL` 指向 PostgreSQL 時會全部報錯（計畫一「已知風險 #1」）。本任務把 `load_data` 改為 `text()` + 具名參數 `dict`，並把 `main.py`、4 個 tab 檔案、`helpers.py` 內部的 `get_league_aggregated_stats` 全部呼叫端同步轉換；同時把 `load_data` 與 `get_league_aggregated_stats` 的 `@st.cache_data` 加上 `ttl=3600`（spec §4d，配合每日 ETL，避免資料更新後儀表板仍顯示舊快取）。

**Files:**
- Modify: `src/app/helpers.py:76-80`（`load_data`）、`src/app/helpers.py:205-239`（`get_league_aggregated_stats` 的 SQL 與呼叫）
- Modify: `src/app/main.py:86-101`
- Modify: `src/app/tabs/box_score.py:65-68,80-89,155-172,180-197`
- Modify: `src/app/tabs/match_trend.py:21-24`
- Modify: `src/app/tabs/player_deep.py:28-44,73-76,109,165-167`
- Modify: `src/app/tabs/weekly_report_tab.py:93-104`（`_attach_set_scores`）
- Modify: `tests/test_helpers_db.py`

**Interfaces:**
- Produces: `helpers.load_data(query: str, params: dict | None = None) -> pd.DataFrame`（`@st.cache_data(ttl=3600)`，內部用 `sqlalchemy.text()` 包裝查詢字串）。取代原本 `params: tuple = ()` 的 qmark 介面，Task 2/7/8/9 之後所有呼叫端一律使用此新簽名。
- Produces: `helpers.get_league_aggregated_stats(gender_code: str) -> pd.DataFrame`（本 Task 僅將內部查詢改為具名參數並加上 `ttl=3600`，簽名維持不變；Task 9 會再加上 `season` 參數）。
- Consumes：`tests/conftest.py` 的 `sqlite_engine` fixture（既有）。

- [ ] **Step 1: 修改 `tests/test_helpers_db.py` 改用具名參數（此時 `load_data` 尚未改，測試應失敗）**

```python
# tests/test_helpers_db.py
import pandas as pd
from sqlalchemy import create_engine, text


def test_load_data_reads_via_db_config_engine(sqlite_engine, monkeypatch):
    with sqlite_engine.begin() as conn:
        conn.execute(text("INSERT INTO teams (team_id, team_name, gender) VALUES (1, 'X', 'M')"))

    import src.app.helpers as helpers

    helpers.load_data.clear()  # 清除 st.cache_data 快取，避免跨測試互相汙染
    df = helpers.load_data(
        "SELECT * FROM teams WHERE gender = :gender_code",
        {"gender_code": "M"},
    )
    assert len(df) == 1
    assert df.iloc[0]["team_name"] == "X"


def test_load_data_supports_multiple_named_params(sqlite_engine):
    with sqlite_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO teams (team_id, team_name, gender) VALUES "
            "(1, 'A', 'M'), (2, 'B', 'M'), (3, 'C', 'F')"
        ))

    import src.app.helpers as helpers

    helpers.load_data.clear()
    df = helpers.load_data(
        "SELECT * FROM teams WHERE gender = :gender_code AND team_id = :team_id",
        {"gender_code": "M", "team_id": 2},
    )
    assert len(df) == 1
    assert df.iloc[0]["team_name"] == "B"


def test_load_data_query_compiles_under_postgresql_dialect():
    """具名參數查詢須能被 PostgreSQL dialect compiler 編譯，確保上線後 psycopg 相容。"""
    pg_engine = create_engine("postgresql+psycopg://user:pass@localhost:5432/tvl")
    compiled = text("SELECT * FROM teams WHERE gender = :gender_code").compile(dialect=pg_engine.dialect)
    assert "gender_code" in str(compiled)


def test_fetch_match_index_uses_season_year_for_month():
    from src.app.helpers import fetch_match_index

    import src.app.helpers as helpers

    assert helpers.season_year_for_month(11) == 2025
    assert helpers.season_year_for_month(3) == 2026
    assert callable(fetch_match_index)
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_helpers_db.py -v`
Expected: FAIL — `sqlite3.ProgrammingError`（`?` 佔位符無法對應 dict 參數）或 `TypeError`

- [ ] **Step 3: 改寫 `helpers.py` 的 `load_data` 與 `get_league_aggregated_stats`**

`src/app/helpers.py` 第 66-80 行（原本 `from src.utils.constants import (...)` / `from src.utils.db_config import get_engine` / `load_data`）改為：

```python
from src.utils.constants import (
    EXT_BASE, EXT_CUP_ID, EXT_HEADERS, OPP_SHORT_TO_TEAM, season_year_for_month,
)
from src.utils.db_config import get_engine
from sqlalchemy import text

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "match_predictor.pkl"


# ── DB 查詢 ──────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_data(query: str, params: dict | None = None) -> pd.DataFrame:
    """透過 db_config 的 engine 執行具名參數化查詢並回傳 DataFrame（ttl=3600 配合每日 ETL）。"""
    engine = get_engine()
    return pd.read_sql_query(text(query), engine, params=params or {})
```

（`from sqlalchemy import text` 暫時放在此處，Task 2 會統一把所有 import 移到檔案頂部。）

`get_league_aggregated_stats`（第 205-239 行）改為：

```python
@st.cache_data(ttl=3600)
def get_league_aggregated_stats(gender_code: str) -> pd.DataFrame:
    """
    撈取該組別所有球員的聚合統計數據，JOIN players + teams 取得姓名/球隊/位置。
    僅保留總局數 >= 5 的球員，排除極端值。
    """
    raw = load_data(
        """
        SELECT p.player_id,
               p.name,
               p.position,
               t.team_name,
               SUM(s.sets_played)       AS total_sets,
               SUM(s.attack_points)     AS atk_pts,
               SUM(s.attack_total)      AS atk_tot,
               SUM(s.block_points)      AS blk_pts,
               SUM(s.serve_points)      AS srv_pts,
               SUM(s.serve_total)       AS srv_tot,
               SUM(s.receive_excellent) AS rcv_exc,
               SUM(s.receive_total)     AS rcv_tot,
               SUM(s.dig_excellent)     AS dig_exc,
               SUM(s.dig_total)         AS dig_tot,
               SUM(s.set_excellent)     AS set_exc,
               SUM(s.set_total)         AS set_tot,
               SUM(s.total_points)      AS total_points,
               COUNT(*)                 AS n_games
        FROM player_match_stats s
        JOIN players p ON s.player_id = p.player_id
        JOIN teams   t ON p.team_id = t.team_id AND p.gender = t.gender
        WHERE p.gender = :gender_code
        GROUP BY p.player_id
        HAVING SUM(s.sets_played) >= 5
        """,
        {"gender_code": gender_code},
    )
    # 計算進階比率指標（向量化，與原邏輯相同）
    raw["asr"] = vec_pct(raw["atk_pts"], raw["atk_tot"])
    raw["gp_pct"] = vec_pct(raw["rcv_exc"], raw["rcv_tot"])
    raw["ace_pct"] = vec_pct(raw["srv_pts"], raw["srv_tot"])
    raw["dig_pct"] = vec_pct(raw["dig_exc"], raw["dig_tot"])
    raw["ppg"] = np.where(raw["n_games"] > 0, raw["total_points"] / raw["n_games"], 0.0)
    raw["set_pct"] = vec_pct(raw["set_exc"], raw["set_tot"])
    raw["blk_per_set"] = np.where(raw["total_sets"] > 0, raw["blk_pts"] / raw["total_sets"], 0.0)
    raw["def_load"] = raw["rcv_tot"] + raw["dig_tot"]
    raw["def_pct"] = vec_pct(raw["rcv_exc"] + raw["dig_exc"], raw["rcv_tot"] + raw["dig_tot"])

    pr_cols = ["asr", "gp_pct", "ace_pct", "dig_pct", "set_pct", "blk_per_set", "def_pct"]
    for col in pr_cols:
        raw[f"{col}_pr"] = (
            raw.groupby("position")[col]
            .rank(pct=True)
            .mul(100)
            .round(1)
        )

    return raw
```

- [ ] **Step 4: 更新呼叫端 —— `main.py`**

第 86-101 行改為：

```python
teams_df = load_data(
    "SELECT team_id, team_name FROM teams WHERE gender = :gender_code ORDER BY team_id",
    {"gender_code": gender_code},
)
if teams_df.empty:
    st.warning("該組別目前沒有球隊資料。")
    st.stop()

team_name = st.sidebar.selectbox("選擇球隊", teams_df["team_name"].tolist())
team_id = int(teams_df.loc[teams_df["team_name"] == team_name, "team_id"].iloc[0])

players_df = load_data(
    "SELECT player_id, jersey_number, name, position FROM players "
    "WHERE team_id = :team_id AND gender = :gender_code ORDER BY jersey_number",
    {"team_id": team_id, "gender_code": gender_code},
)
if players_df.empty:
    st.warning("該球隊目前沒有球員資料。")
    st.stop()
```

- [ ] **Step 5: 更新呼叫端 —— `box_score.py`**

第 65-68 行：

```python
    bs_teams = load_data(
        "SELECT team_id, team_name FROM teams WHERE gender = :gender_code ORDER BY team_id",
        {"gender_code": bs_gender_code},
    )
```

第 80-89 行：

```python
    matches_df = load_data(
        """
        SELECT DISTINCT s.match_date, s.opponent
        FROM player_match_stats s
        JOIN players p ON s.player_id = p.player_id
        WHERE p.team_id = :team_id AND p.gender = :gender_code
        ORDER BY s.match_date
        """,
        {"team_id": bs_team_id, "gender_code": bs_gender_code},
    )
```

第 155-172 行：

```python
    team_a_df = load_data(
        """
        SELECT p.name, p.position, s.sets_played,
               s.attack_points, s.attack_total,
               s.block_points,
               s.serve_points, s.serve_total,
               s.receive_excellent, s.receive_total,
               s.dig_excellent, s.dig_total,
               s.set_excellent, s.set_total,
               s.total_points
        FROM player_match_stats s
        JOIN players p ON s.player_id = p.player_id
        WHERE p.team_id = :team_id AND p.gender = :gender_code
          AND s.match_date = :match_date AND s.opponent = :opponent
        ORDER BY s.total_points DESC
        """,
        {
            "team_id": bs_team_id, "gender_code": bs_gender_code,
            "match_date": sel_date, "opponent": sel_opponent,
        },
    )
```

第 180-197 行：

```python
        team_b_df = load_data(
            """
            SELECT p.name, p.position, s.sets_played,
                   s.attack_points, s.attack_total,
                   s.block_points,
                   s.serve_points, s.serve_total,
                   s.receive_excellent, s.receive_total,
                   s.dig_excellent, s.dig_total,
                   s.set_excellent, s.set_total,
                   s.total_points
            FROM player_match_stats s
            JOIN players p ON s.player_id = p.player_id
            WHERE p.team_id = :team_id AND p.gender = :gender_code
              AND s.match_date = :match_date
            ORDER BY s.total_points DESC
            """,
            {"team_id": opp_team_id, "gender_code": opp_gender, "match_date": sel_date},
        )
```

- [ ] **Step 6: 更新呼叫端 —— `match_trend.py`**

第 21-24 行：

```python
    match_df = load_data(
        "SELECT * FROM player_match_stats WHERE player_id = :player_id ORDER BY match_date",
        {"player_id": player_id},
    )
```

- [ ] **Step 7: 更新呼叫端 —— `player_deep.py`**

第 28-44 行（`_load_league_agg`）：

```python
def _load_league_agg(gender_code: str, pos_filter: str = "", params: dict | None = None):
    """撈取聯盟聚合數據（全組別或特定位置）。"""
    return load_data(
        f"""
        SELECT SUM(s.attack_points) AS atk_pts, SUM(s.attack_total) AS atk_tot,
               SUM(s.receive_excellent) AS rcv_exc, SUM(s.receive_total) AS rcv_tot,
               SUM(s.serve_points) AS srv_pts, SUM(s.serve_total) AS srv_tot,
               SUM(s.dig_excellent) AS dig_exc, SUM(s.dig_total) AS dig_tot,
               SUM(s.set_excellent) AS set_exc, SUM(s.set_total) AS set_tot,
               SUM(s.block_points) AS blk_pts, SUM(s.sets_played) AS tot_sets,
               SUM(s.total_points) AS tot_pts, COUNT(*) AS n_games
        FROM player_match_stats s
        JOIN players p ON s.player_id = p.player_id
        WHERE p.gender = :gender_code {pos_filter}
        """,
        params if params is not None else {"gender_code": gender_code},
    ).iloc[0]
```

第 109 行（全聯盟平均呼叫端）：

```python
    la = _parse_agg(_load_league_agg(gender_code, params={"gender_code": gender_code}))
```

第 165-167 行（同位置平均呼叫端）：

```python
    pos_filter = "AND p.position = :position" if player_position else ""
    pos_params = (
        {"gender_code": gender_code, "position": player_position}
        if player_position else {"gender_code": gender_code}
    )
    lg = _parse_agg(_load_league_agg(gender_code, pos_filter, pos_params))
```

第 73-76 行（`stats_df`）：

```python
    stats_df = load_data(
        "SELECT * FROM player_match_stats WHERE player_id = :player_id ORDER BY match_date",
        {"player_id": player_id},
    )
```

- [ ] **Step 8: 更新呼叫端 —— `weekly_report_tab.py`（動態 `IN (...)` 子句）**

第 93-104 行（`_attach_set_scores` 內的 `matches_db` 查詢）改為：

```python
    date_params = {f"d{i}": d for i, d in enumerate(dates)}
    placeholders = ",".join(f":{k}" for k in date_params)
    matches_db = load_data(
        f"""SELECT match_date, home_team, away_team,
                   home_set1, home_set2, home_set3, home_set4, home_set5,
                   away_set1, away_set2, away_set3, away_set4, away_set5,
                   home_total, away_total,
                   home_sets_won, away_sets_won, is_golden_set
            FROM matches
            WHERE match_date IN ({placeholders})""",
        date_params,
    )
```

（原本的 `placeholders = ",".join(["?"] * len(dates))` 與 `tuple(dates)` 整段移除。）

- [ ] **Step 9: 執行完整測試套件確認通過**

Run: `pytest tests/ -v`
Expected: 全部 PASS（既有 48 個測試 + 本 Task 新增的測試）

- [ ] **Step 10: Commit**

```bash
git add src/app/helpers.py src/app/main.py src/app/tabs/box_score.py \
        src/app/tabs/match_trend.py src/app/tabs/player_deep.py \
        src/app/tabs/weekly_report_tab.py tests/test_helpers_db.py
git commit -m "fix: load_data 改用 SQLAlchemy text() 具名參數，修正 PostgreSQL 不相容問題"
```

---

### Task 2: Streamlit 正確性修正（`st.stop()` → `return`、快取、`load_dotenv` 選用、import 排序）

對應 spec §4d。6 個 tab 檔案內共 9 處 `st.stop()`（見「專案現況」列表）——`st.stop()` 會中止**整份 script**，在 `with tab:` 的 eager 執行模型下，其中一個 tab 提前呼叫 `st.stop()` 會讓後面所有分頁一併空白；改用 `return` 只會結束當前 tab 的 `render()`。同時：`_purge_mpl_font_cache` 包進 `@st.cache_resource`（`_init_matplotlib_fonts` 已有此裝飾器，不需再改）；`weekly_report_tab.py` 的 `load_dotenv` 改為僅在 `.env` 檔案存在時才呼叫；`helpers.py` 中段的 import 移到檔案頂部。

**Files:**
- Modify: `src/app/tabs/box_score.py:69-71,90-92`
- Modify: `src/app/tabs/league_pr.py:36-38,112-114,126-128`
- Modify: `src/app/tabs/match_trend.py:26-28`
- Modify: `src/app/tabs/player_deep.py:78-80`
- Modify: `src/app/tabs/weekly_report_tab.py:1-17,320-322,349-351`
- Modify: `src/app/main.py:18-28`
- Modify: `src/app/helpers.py:1-14,66-71`（import 排序，延續 Task 1 的內容）
- Create: `tests/test_streamlit_ui.py`
- Create: `tests/test_weekly_report_tab_dotenv.py`

**Interfaces:**
- Produces: `weekly_report_tab._load_env_if_present(path: Path) -> None`（新的內部函式，供測試驗證「檔案不存在即跳過」行為）。
- Consumes：`tests/conftest.py` 的 `sqlite_engine` fixture（建立空 schema 的暫存 SQLite，觸發本 Task 各 tab 的空資料早退分支）。

- [ ] **Step 1: 寫失敗測試 —— `st.stop()` → `return` 的 AppTest 驗證**

```python
# tests/test_streamlit_ui.py
"""
Streamlit UI 正確性測試：驗證 6 個 tab 的 st.stop() → return 修正。
st.stop() 會中止整份 script，害後面所有分頁空白；改用 return 只結束當前 tab。
使用 AppTest 模擬渲染，只驗證各檔案「第一個」空資料早退分支，
不追求窮舉每一處 st.stop() 的分支（沿用計畫一「不追求全面覆蓋」的測試策略）。
"""

from streamlit.testing.v1 import AppTest


def _assert_returns_without_stopping_script(harness) -> None:
    at = AppTest.from_function(harness)
    at.run(timeout=30)
    assert not at.exception, f"渲染時發生例外：{at.exception}"
    markers = [t.value for t in at.text if t.value == "MARKER_AFTER_RENDER"]
    assert markers, "render() 應以 return 結束當前 tab，其後的程式碼仍應正常執行"


def test_box_score_empty_teams_returns_without_stopping_script(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        import streamlit as st
        from src.app.tabs import box_score

        box_score.render({})
        st.text("MARKER_AFTER_RENDER")

    _assert_returns_without_stopping_script(_harness)


def test_league_pr_empty_league_returns_without_stopping_script(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        import streamlit as st
        from src.app.tabs import league_pr

        ctx = {
            "player_id": 1, "player_name": "測試球員", "player_position": "OH",
            "gender_code": "M", "gender": "男子組", "team_name": "測試隊",
        }
        league_pr.render(ctx)
        st.text("MARKER_AFTER_RENDER")

    _assert_returns_without_stopping_script(_harness)


def test_match_trend_empty_data_returns_without_stopping_script(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        import streamlit as st
        from src.app.tabs import match_trend

        ctx = {
            "player_id": 1, "player_name": "測試球員", "player_position": "OH",
            "gender_code": "M", "gender": "男子組", "team_name": "測試隊",
        }
        match_trend.render(ctx)
        st.text("MARKER_AFTER_RENDER")

    _assert_returns_without_stopping_script(_harness)


def test_player_deep_empty_data_returns_without_stopping_script(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        import streamlit as st
        from src.app.tabs import player_deep

        ctx = {
            "player_id": 1, "player_name": "測試球員", "player_position": "OH",
            "gender_code": "M", "gender": "男子組",
        }
        player_deep.render(ctx)
        st.text("MARKER_AFTER_RENDER")

    _assert_returns_without_stopping_script(_harness)


def test_weekly_report_no_weeks_returns_without_stopping_script(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        import streamlit as st
        from src.app.tabs import weekly_report_tab

        weekly_report_tab.render({})
        st.text("MARKER_AFTER_RENDER")

    _assert_returns_without_stopping_script(_harness)
```

```python
# tests/test_weekly_report_tab_dotenv.py
"""驗證 load_dotenv 改為選用：.env 不存在時應直接跳過，不拋例外。"""

import os


def test_load_env_if_present_skips_missing_file(tmp_path, monkeypatch):
    from src.app.tabs.weekly_report_tab import _load_env_if_present

    monkeypatch.delenv("TVL_TEST_ENV_KEY", raising=False)
    missing_path = tmp_path / "does_not_exist.env"

    _load_env_if_present(missing_path)  # 不應拋出例外

    assert "TVL_TEST_ENV_KEY" not in os.environ


def test_load_env_if_present_loads_existing_file(tmp_path, monkeypatch):
    from src.app.tabs.weekly_report_tab import _load_env_if_present

    monkeypatch.delenv("TVL_TEST_ENV_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("TVL_TEST_ENV_KEY=hello\n", encoding="utf-8")

    _load_env_if_present(env_file)

    assert os.environ.get("TVL_TEST_ENV_KEY") == "hello"
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_streamlit_ui.py tests/test_weekly_report_tab_dotenv.py -v`
Expected: `test_streamlit_ui.py` 全部 FAIL（目前 `st.stop()` 會中止腳本，`MARKER_AFTER_RENDER` 不會被寫入）；`test_weekly_report_tab_dotenv.py` 因 `_load_env_if_present` 尚不存在而 FAIL（`ImportError`）

- [ ] **Step 3: 修正 `box_score.py`**

第 69-71 行：

```python
    if bs_teams.empty:
        st.warning("該組別無球隊資料。")
        return
```

第 90-92 行：

```python
    if matches_df.empty:
        st.info("該球隊尚無比賽紀錄。")
        return
```

- [ ] **Step 4: 修正 `league_pr.py`**

第 36-38 行：

```python
    if league_all.empty:
        st.info("目前無足夠的聯盟數據。")
        return
```

第 112-114 行：

```python
    if not positions:
        st.warning("球員位置資料不足，無法繪製散佈圖。")
        return
```

第 126-128 行：

```python
    if len(pos_df) < 2:
        st.info(f"位置 {selected_pos} 僅有 {len(pos_df)} 位球員，資料不足。")
        return
```

- [ ] **Step 5: 修正 `match_trend.py`**

第 26-28 行：

```python
    if match_df.empty:
        st.info("該球員目前沒有比賽數據紀錄。")
        return
```

- [ ] **Step 6: 修正 `player_deep.py`**

第 78-80 行：

```python
    if stats_df.empty:
        st.info("該球員目前沒有比賽數據紀錄。")
        return
```

- [ ] **Step 7: 修正 `weekly_report_tab.py`（`st.stop()` → `return` + `load_dotenv` 選用）**

第 1-17 行（檔案開頭）改為：

```python
"""
Tab 6：每周戰報（MLIS AI 生成）
提供周次選擇、性別篩選，以視覺化卡片呈現該周比賽摘要，並透過 MLIS 產生專業戰報。
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.app.helpers import load_data
from src.etl.weekly_report import gather_weekly_data, get_match_weeks


def _load_env_if_present(path: Path) -> None:
    """僅在 .env 檔案存在時載入，避免正式環境（無 .env）產生不必要的行為。"""
    if path.exists():
        load_dotenv(path)


_load_env_if_present(Path(__file__).resolve().parents[3] / ".env")
```

（`import os` 移除——`os` 僅被舊有的 `_get_gemini_key` 使用，該函式會在 Task 7 移除；此處先移除 import 避免 lint 警告未使用。標題與 docstring 的「Gemini」字樣也一併改為「MLIS」，Task 7 會實際串接。）

第 320-322 行：

```python
    if not weeks:
        st.info("資料庫中尚無比賽紀錄。")
        return
```

第 349-351 行：

```python
    if not all_matches:
        st.info("該周次無符合條件的比賽。")
        return
```

- [ ] **Step 8: main.py —— `_purge_mpl_font_cache` 包進 `@st.cache_resource`**

第 18-28 行改為：

```python
# ── 在 matplotlib 匯入前清除舊字型快取，避免抓到缺少 CJK 字型的舊快取 ──
@st.cache_resource
def _purge_mpl_font_cache():
    import os, glob
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "matplotlib")
    if os.path.isdir(cache_dir):
        for f in glob.glob(os.path.join(cache_dir, "fontlist-*")):
            try:
                os.remove(f)
            except OSError:
                pass
_purge_mpl_font_cache()
```

（`_init_matplotlib_fonts`，第 39-40 行，已經有 `@st.cache_resource`，不需修改。）

- [ ] **Step 9: `helpers.py` —— import 移到檔案頂部**

檔案開頭（第 1-14 行）改為：

```python
"""
TVL 儀表板共用函式
提供 DB 查詢、外部系統資料擷取、進階指標計算等共用功能。
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sqlalchemy import text

from src.utils.constants import (
    EXT_BASE, EXT_CUP_ID, EXT_HEADERS, OPP_SHORT_TO_TEAM, season_year_for_month,
)
from src.utils.db_config import get_engine

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "match_predictor.pkl"
```

並移除 Task 1 遺留在 `compact_margin` 函式後方（原第 66-71 行）的 `from src.utils.constants import (...)`、`from src.utils.db_config import get_engine`、`from sqlalchemy import text`、`MODEL_PATH = ...` 這幾行（已搬到檔案頂部，此處整段刪除，`load_data` 函式緊接在 `compact_margin` 之後的 `# ── DB 查詢 ──` 註解區塊維持不變）。

- [ ] **Step 10: 執行測試確認通過**

Run: `pytest tests/test_streamlit_ui.py tests/test_weekly_report_tab_dotenv.py -v`
Expected: 全部 PASS

- [ ] **Step 11: 執行完整測試套件確認未破壞既有功能**

Run: `pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 12: Commit**

```bash
git add src/app/tabs/box_score.py src/app/tabs/league_pr.py src/app/tabs/match_trend.py \
        src/app/tabs/player_deep.py src/app/tabs/weekly_report_tab.py src/app/main.py \
        src/app/helpers.py tests/test_streamlit_ui.py tests/test_weekly_report_tab_dotenv.py
git commit -m "fix: st.stop() 改 return 避免中止整份腳本，快取包 cache_resource，load_dotenv 選用化"
```

---

### Task 3: `prediction.py` 修正 `feature_cols` 鍵名 bug（對應 spec §4d 遺留項目）

`src/app/tabs/prediction.py:125` 讀取 `artifact.get("feature_names", [])`，但模型 pkl 實際的鍵是 `feature_cols`（計畫一「已知風險 #3」）。導致 `n_features` 恆為 0，恆定落入 V1（5 特徵）分支——目前恰好與 pkl 實際的 5 特徵吻合，功能上無症狀，但若未來訓練出 11 特徵版模型（V2）並更新 pkl，這個 bug 會讓 UI 誤判為 V1。本任務抽出一個可獨立測試的 `_select_slider_config` 函式，修正鍵名並補迴歸測試。

**Files:**
- Modify: `src/app/tabs/prediction.py:113-136`
- Create: `tests/test_prediction_slider_config.py`

**Interfaces:**
- Produces: `prediction._select_slider_config(artifact: dict) -> tuple[list[tuple], str, int]`（回傳 `(slider_cfg, version_label, n_features)`），供 `render()` 呼叫，也供測試直接驗證鍵名讀取邏輯，不需啟動 Streamlit runtime。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_prediction_slider_config.py
"""
迴歸測試：修正前的 bug 讀取 artifact["feature_names"]（不存在的鍵），導致 n_features
恆為 0、恆定判定為 V1。確保現在正確讀取 artifact["feature_cols"]。
"""

from src.app.tabs.prediction import V1_SLIDER_CFG, V2_SLIDER_CFG, _select_slider_config


def test_select_slider_config_reads_feature_cols_key():
    artifact = {"feature_cols": ["ASR", "GP_pct", "DIG_pct", "BLK_per_set", "ACE_pct"]}
    cfg, label, n = _select_slider_config(artifact)
    assert n == 5
    assert cfg == V1_SLIDER_CFG
    assert "V1" in label


def test_select_slider_config_detects_v2_11_features():
    artifact = {"feature_cols": [f"f{i}" for i in range(11)]}
    cfg, label, n = _select_slider_config(artifact)
    assert n == 11
    assert cfg == V2_SLIDER_CFG
    assert "V2" in label


def test_select_slider_config_ignores_wrong_key_name():
    # 即使 artifact 內誤留有 "feature_names" 鍵，也必須以 "feature_cols" 為準
    artifact = {
        "feature_names": [],
        "feature_cols": [f"f{i}" for i in range(11)],
    }
    cfg, label, n = _select_slider_config(artifact)
    assert n == 11
    assert cfg == V2_SLIDER_CFG


def test_select_slider_config_defaults_to_v1_when_feature_cols_missing():
    cfg, label, n = _select_slider_config({})
    assert n == 0
    assert cfg == V1_SLIDER_CFG
    assert "V1" in label
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_prediction_slider_config.py -v`
Expected: FAIL — `ImportError: cannot import name '_select_slider_config'`

- [ ] **Step 3: 修改 `prediction.py`**

第 113-136 行改為：

```python
def _select_slider_config(artifact: dict) -> tuple[list[tuple], str, int]:
    """依模型 artifact 的 feature_cols 數量決定使用哪組 Slider 設定。"""
    feature_cols = artifact.get("feature_cols", [])
    n_features = len(feature_cols)
    if n_features == 11:
        return V2_SLIDER_CFG, "V2（滑動窗口 + 連勝）", n_features
    return V1_SLIDER_CFG, "V1（基本五指標）", n_features


def render(ctx, cjk_font_path=None, cjk_font_stack=None):
    """繪製 Tab 5 — 賽果預測頁面。"""

    st.header("賽果預測 (ML Match Prediction)")

    # ------ 檢查模型檔案是否存在 ------
    if not MODEL_PATH.exists():
        st.info("尚未訓練模型，請先執行模型訓練流程以產生模型檔案。")
        return

    # ------ 載入模型與 SHAP 解釋器 ------
    artifact, model, explainer = _load_model_and_explainer()
    slider_cfg, version_label, n_features = _select_slider_config(artifact)

    st.caption(f"模型版本：{version_label}｜特徵數：{n_features}")
```

（`if n_features == 11: slider_cfg = V2_SLIDER_CFG ...` 這段舊邏輯整段刪除，已內移到 `_select_slider_config`；`render()` 後續使用 `slider_cfg` 的地方維持不變。）

- [ ] **Step 4: 執行測試確認通過**

Run: `pytest tests/test_prediction_slider_config.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 跑既有模型相容性測試確認未破壞**

Run: `pytest tests/test_model_compat.py -v`
Expected: 全部 PASS（本 Task 不改動模型載入邏輯，僅改 UI 讀取鍵名）

- [ ] **Step 6: Commit**

```bash
git add src/app/tabs/prediction.py tests/test_prediction_slider_config.py
git commit -m "fix: prediction.py 修正讀取 pkl 鍵名 bug（feature_names → feature_cols）"
```

---

### Task 4: schema 新增 `app_settings` 表（對應 spec §6）

新增 key-value 設定表，供 Task 5 的 `settings_store.py` 存取 MLIS endpoint/model/API key。兩份 schema（SQLite/PostgreSQL）皆需新增，沿用既有的冪等 `CREATE TABLE IF NOT EXISTS` 風格。

`tests/test_schema.py` 目前已有計畫一最終審查加入的 schema drift 防護：`_split_top_level`/`_parse_tables` 這兩個 helper，以及 `test_sqlite_and_postgres_schema_structural_parity`（逐表比對 SQLite/PostgreSQL 兩份 schema 的欄位集合與 `UNIQUE` 約束是否一致）。本任務**不得**整檔覆蓋這份測試檔，只在既有內容上做增量新增：新增 `app_settings` 相關斷言、並把 `"app_settings"` 加進 `test_sqlite_and_postgres_schema_structural_parity` 的 `expected_tables` 集合——`_parse_tables` 本身是掃描檔案裡全部 `CREATE TABLE IF NOT EXISTS` 語句、不是寫死表名清單，所以只要兩份 schema 都新增了 `app_settings`、且 `expected_tables` 也納入它，這個既有測試就會**自動**比對新表的欄位與 `UNIQUE` 約束是否兩份 schema 一致，不需要再手寫一份新的結構比對測試。

**Files:**
- Modify: `sql/schema.sql`
- Modify: `sql/schema_postgres.sql`
- Modify: `tests/test_schema.py`（增量新增，保留 `_split_top_level`/`_parse_tables`/`test_sqlite_and_postgres_schema_structural_parity`）

**Interfaces:**
- Produces: `app_settings` 表（`key TEXT PRIMARY KEY`、`value TEXT NOT NULL`），SQLite 與 PostgreSQL 兩份 schema 結構一致。Task 5 的 `settings_store.get_setting`/`set_setting` 依賴此表。

- [ ] **Step 1: 在既有 `tests/test_schema.py` 上做增量修改（寫失敗測試）**

在既有 `test_sqlite_schema_applies_twice_idempotently`（第 73-82 行）的表清單加上 `"app_settings"`：

```python
def test_sqlite_schema_applies_twice_idempotently():
    engine = create_engine("sqlite:///:memory:")
    _apply(engine, SCHEMA_SQLITE)
    _apply(engine, SCHEMA_SQLITE)  # 重複套用不可報錯
    with engine.begin() as conn:
        tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).scalars().all()
    for expected in ["teams", "players", "player_match_stats", "matches", "app_settings"]:
        assert expected in tables
```

在既有 `test_postgres_schema_has_no_sqlite_specific_syntax`（第 94-99 行）的表清單加上 `"app_settings"`：

```python
def test_postgres_schema_has_no_sqlite_specific_syntax():
    content = SCHEMA_POSTGRES.read_text(encoding="utf-8")
    assert "AUTOINCREMENT" not in content
    assert "GENERATED BY DEFAULT AS IDENTITY" in content
    for table in ["teams", "players", "player_match_stats", "matches", "app_settings"]:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in content
```

在既有 `test_sqlite_and_postgres_schema_structural_parity`（第 102-118 行，`_split_top_level`/`_parse_tables` 這兩個 helper 維持完全不變）的 `expected_tables` 加上 `"app_settings"`：

```python
def test_sqlite_and_postgres_schema_structural_parity():
    """兩份 schema 檔的表欄位集合（含順序）與 UNIQUE 約束須完全一致，避免其中一份漏改而產生 drift。"""
    sqlite_tables = _parse_tables(SCHEMA_SQLITE)
    postgres_tables = _parse_tables(SCHEMA_POSTGRES)
    expected_tables = {"teams", "players", "player_match_stats", "matches", "app_settings"}
    assert set(sqlite_tables) == expected_tables
    assert set(postgres_tables) == expected_tables

    for table in expected_tables:
        assert sqlite_tables[table]["columns"] == postgres_tables[table]["columns"], (
            f"{table} 欄位集合不一致：sqlite={sqlite_tables[table]['columns']} "
            f"postgres={postgres_tables[table]['columns']}"
        )
        assert sqlite_tables[table]["uniques"] == postgres_tables[table]["uniques"], (
            f"{table} UNIQUE 約束不一致"
        )
```

在檔案末尾新增一個測試，直接驗證 `app_settings` 的 upsert 語意（`_parse_tables` 只比對結構，不驗證實際 SQL 行為）：

```python
def test_app_settings_table_upserts_by_key():
    engine = create_engine("sqlite:///:memory:")
    _apply(engine, SCHEMA_SQLITE)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO app_settings (key, value) VALUES ('mlis_model', 'qwen-a') "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value"
        ))
        conn.execute(text(
            "INSERT INTO app_settings (key, value) VALUES ('mlis_model', 'qwen-b') "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value"
        ))
        value = conn.execute(
            text("SELECT value FROM app_settings WHERE key = 'mlis_model'")
        ).scalar_one()
    assert value == "qwen-b"
```

（`test_sqlite_schema_has_no_drop_table`、`test_sqlite_schema_has_season_columns`、`_split_top_level`、`_parse_tables` 這幾個既有函式完全不動。）

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL —
`test_sqlite_schema_applies_twice_idempotently`（`app_settings` 不在表清單）；
`test_postgres_schema_has_no_sqlite_specific_syntax`（缺 `app_settings`）；
`test_sqlite_and_postgres_schema_structural_parity`（兩份 schema 都還沒有 `app_settings` 表，`_parse_tables` 解析後 `sqlite_tables`/`postgres_tables` 都只有 4 個 key，`assert set(sqlite_tables) == expected_tables` 因 `expected_tables` 已有 5 個 key 而不相等，FAIL）；
`test_app_settings_table_upserts_by_key`（表不存在）

- [ ] **Step 3: 修改 `sql/schema.sql`**

在 `matches` 表區塊（第 78 行 `);` 結尾）之後、`-- 效能索引` 註解（第 81 行）之前，插入：

```sql
CREATE TABLE IF NOT EXISTS app_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

- [ ] **Step 4: 修改 `sql/schema_postgres.sql`**

在同樣位置（`matches` 表區塊之後、索引之前）插入相同的 `app_settings` 表定義（PostgreSQL 對 `TEXT PRIMARY KEY` 原生支援，不需要 `GENERATED ... AS IDENTITY`）：

```sql
CREATE TABLE IF NOT EXISTS app_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

- [ ] **Step 5: 執行測試確認通過**

Run: `pytest tests/test_schema.py -v`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add sql/schema.sql sql/schema_postgres.sql tests/test_schema.py
git commit -m "feat: schema 新增 app_settings key-value 表，供系統設定 UI 使用"
```

---

### Task 5: `settings_store.py`（對應 spec §6）

`app_settings` 表的存取模組，upsert 語意。供 Task 6（`llm_client`）與 Task 8（系統設定分頁）使用。

**Files:**
- Create: `src/app/settings_store.py`
- Create: `tests/test_settings_store.py`

**Interfaces:**
- Produces: `settings_store.get_setting(engine: Engine, key: str) -> str | None`、`settings_store.set_setting(engine: Engine, key: str, value: str) -> None`。
- Consumes：`tests/conftest.py` 的 `sqlite_engine` fixture（Task 4 的 `app_settings` 表已包含在 schema 中）。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_settings_store.py
from src.app.settings_store import get_setting, set_setting


def test_get_setting_returns_none_when_missing(sqlite_engine):
    assert get_setting(sqlite_engine, "does_not_exist") is None


def test_set_setting_then_get_setting_roundtrip(sqlite_engine):
    set_setting(sqlite_engine, "mlis_model", "qwen2.5-72b")
    assert get_setting(sqlite_engine, "mlis_model") == "qwen2.5-72b"


def test_set_setting_upserts_existing_key(sqlite_engine):
    set_setting(sqlite_engine, "mlis_model", "qwen-a")
    set_setting(sqlite_engine, "mlis_model", "qwen-b")
    assert get_setting(sqlite_engine, "mlis_model") == "qwen-b"


def test_set_setting_does_not_affect_other_keys(sqlite_engine):
    set_setting(sqlite_engine, "mlis_base_url", "http://a")
    set_setting(sqlite_engine, "mlis_model", "qwen-a")
    assert get_setting(sqlite_engine, "mlis_base_url") == "http://a"
    assert get_setting(sqlite_engine, "mlis_model") == "qwen-a"
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_settings_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.app.settings_store'`

- [ ] **Step 3: 建立 `settings_store.py`**

```python
# src/app/settings_store.py
"""
app_settings key-value 表存取模組。
供「系統設定」分頁與 llm_client 的設定讀取順序（DB → 環境變數）使用。
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine


def get_setting(engine: Engine, key: str) -> str | None:
    """讀取單一設定值，不存在時回傳 None。"""
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT value FROM app_settings WHERE key = :key"),
            {"key": key},
        ).first()
    return row[0] if row else None


def set_setting(engine: Engine, key: str, value: str) -> None:
    """寫入或更新單一設定值（upsert）。"""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO app_settings (key, value)
                VALUES (:key, :value)
                ON CONFLICT (key) DO UPDATE SET value = excluded.value
            """),
            {"key": key, "value": value},
        )
```

- [ ] **Step 4: 執行測試確認通過**

Run: `pytest tests/test_settings_store.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/settings_store.py tests/test_settings_store.py
git commit -m "feat: 新增 settings_store 模組讀寫 app_settings key-value 表"
```

---

### Task 6: `llm_client.py` —— MLIS OpenAI 相容呼叫層（對應 spec §5）

新增獨立、可單元測試（mock HTTP）的 LLM 呼叫層，取代 `weekly_report_tab.py` 內原本的 Gemini 呼叫邏輯（該邏輯本身在 Task 7 才移除，本 Task 只新增 `llm_client.py` 本體）。設定讀取順序：DB `app_settings` → 環境變數 → 皆無則回傳 `None`。簡單重試（最多 2 次、短間隔 1 秒），移除舊有 `time.sleep(30)` 阻塞式重試與三模型 fallback 迴圈的設計。

**Files:**
- Create: `src/app/llm_client.py`
- Create: `tests/test_llm_client.py`
- Modify: `requirements.txt`（新增 `openai==2.53.0`）
- Modify: `requirements-dev.txt`（新增 `httpx==0.28.1`，測試直接 `import httpx` 做 mock transport）

**Interfaces:**
- Produces: `llm_client.LLMConfig`（dataclass：`base_url: str`、`api_key: str`、`model: str`）、`llm_client.resolve_llm_config(engine: Engine) -> LLMConfig | None`、`llm_client.generate_report(config: LLMConfig, system_prompt: str, user_prompt: str, *, client: OpenAI | None = None) -> str`、`llm_client.test_connection(config: LLMConfig) -> tuple[bool, str]`。Task 7（`weekly_report_tab.py`）與 Task 8（`settings_tab.py`）皆依賴這些函式與 `LLMConfig`。
- Consumes：`settings_store.get_setting`（Task 5）。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_llm_client.py
"""
llm_client 測試：全部使用 httpx.MockTransport 模擬 MLIS 的 OpenAI 相容 endpoint，
不連真實網路（真實 endpoint 於 PCAI 上另行驗證）。
"""

import json

import httpx
import pytest
from openai import OpenAI

from src.app.llm_client import LLMConfig, generate_report, resolve_llm_config, test_connection


def _make_mock_client(handler) -> OpenAI:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    return OpenAI(base_url="http://fake-mlis.local/v1", api_key="test-key", http_client=http_client)


def _success_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["model"] == "qwen-test"
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": "qwen-test",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "測試戰報內容"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def test_generate_report_returns_content_on_success():
    config = LLMConfig(base_url="http://fake-mlis.local/v1", api_key="test-key", model="qwen-test")
    client = _make_mock_client(_success_handler)

    result = generate_report(config, "系統提示", "使用者提示", client=client)

    assert result == "測試戰報內容"


def test_generate_report_retries_then_raises_friendly_error(monkeypatch):
    import src.app.llm_client as llm_client

    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)  # 測試不等待真實間隔

    call_count = {"n": 0}

    def _failing_handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(500, json={"error": {"message": "internal error"}})

    config = LLMConfig(base_url="http://fake-mlis.local/v1", api_key="test-key", model="qwen-test")
    client = _make_mock_client(_failing_handler)

    with pytest.raises(RuntimeError, match="MLIS 服務呼叫失敗"):
        generate_report(config, "系統提示", "使用者提示", client=client)

    assert call_count["n"] == 3  # 初次 + 最多 2 次重試


def test_test_connection_reports_success(monkeypatch):
    config = LLMConfig(base_url="http://fake-mlis.local/v1", api_key="test-key", model="qwen-test")

    import src.app.llm_client as llm_client
    monkeypatch.setattr(llm_client, "_build_client", lambda cfg: _make_mock_client(_success_handler))

    ok, message = test_connection(config)

    assert ok is True
    assert message == "連線成功"


def test_test_connection_reports_failure(monkeypatch):
    def _failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "unauthorized"}})

    config = LLMConfig(base_url="http://fake-mlis.local/v1", api_key="bad-key", model="qwen-test")

    import src.app.llm_client as llm_client
    monkeypatch.setattr(llm_client, "_build_client", lambda cfg: _make_mock_client(_failing_handler))

    ok, message = test_connection(config)

    assert ok is False
    assert "連線失敗" in message


def test_resolve_llm_config_prefers_db_over_env(sqlite_engine, monkeypatch):
    from src.app.settings_store import set_setting

    monkeypatch.setenv("MLIS_BASE_URL", "http://env-endpoint/v1")
    monkeypatch.setenv("MLIS_API_KEY", "env-key")
    monkeypatch.setenv("MLIS_MODEL", "env-model")

    set_setting(sqlite_engine, "mlis_base_url", "http://db-endpoint/v1")
    set_setting(sqlite_engine, "mlis_api_key", "db-key")
    set_setting(sqlite_engine, "mlis_model", "db-model")

    config = resolve_llm_config(sqlite_engine)

    assert config == LLMConfig(base_url="http://db-endpoint/v1", api_key="db-key", model="db-model")


def test_resolve_llm_config_falls_back_to_env_when_db_empty(sqlite_engine, monkeypatch):
    monkeypatch.setenv("MLIS_BASE_URL", "http://env-endpoint/v1")
    monkeypatch.setenv("MLIS_API_KEY", "env-key")
    monkeypatch.setenv("MLIS_MODEL", "env-model")

    config = resolve_llm_config(sqlite_engine)

    assert config == LLMConfig(base_url="http://env-endpoint/v1", api_key="env-key", model="env-model")


def test_resolve_llm_config_returns_none_when_nothing_set(sqlite_engine, monkeypatch):
    for key in ("MLIS_BASE_URL", "MLIS_API_KEY", "MLIS_MODEL"):
        monkeypatch.delenv(key, raising=False)

    assert resolve_llm_config(sqlite_engine) is None
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.app.llm_client'`（`openai`/`httpx` 尚未安裝也會導致 `ImportError`）

- [ ] **Step 3: 新增依賴**

`requirements.txt` 新增一行（維持字母排序，插在 `numpy` 與 `pandas` 之間）：

```txt
openai==2.53.0
```

`requirements-dev.txt` 新增：

```txt
-r requirements.txt
pytest==9.1.1
httpx==0.28.1
```

安裝：

```bash
cd /mnt/d/HPE-PCAI/TVL-Analysis
source .venv/bin/activate
pip install openai==2.53.0 httpx==0.28.1
```

- [ ] **Step 4: 建立 `llm_client.py`**

```python
# src/app/llm_client.py
"""
LLM 呼叫層：透過 OpenAI 相容 API 呼叫 PCAI MLIS 部署的模型，取代原本的 Gemini 呼叫。
設定讀取順序：DB app_settings（UI 設定）→ 環境變數 → 皆無則回傳 None，由呼叫端顯示引導訊息。
"""

import os
import time
from dataclasses import dataclass

from openai import OpenAI
from sqlalchemy.engine import Engine

from src.app.settings_store import get_setting
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.0


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str


def resolve_llm_config(engine: Engine) -> LLMConfig | None:
    """讀取順序：DB app_settings → 環境變數；任一欄位缺漏則回傳 None。"""
    base_url = get_setting(engine, "mlis_base_url") or os.environ.get("MLIS_BASE_URL")
    api_key = get_setting(engine, "mlis_api_key") or os.environ.get("MLIS_API_KEY")
    model = get_setting(engine, "mlis_model") or os.environ.get("MLIS_MODEL")
    if not (base_url and api_key and model):
        return None
    return LLMConfig(base_url=base_url, api_key=api_key, model=model)


def _build_client(config: LLMConfig) -> OpenAI:
    """建立 OpenAI 相容 client。max_retries=0：openai SDK 預設會自己重試 2 次，
    若不關閉，會與下方 generate_report 的重試邏輯疊加，讓一次失敗變成多達 9 次實際請求。"""
    return OpenAI(base_url=config.base_url, api_key=config.api_key, max_retries=0)


def generate_report(
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    *,
    client: OpenAI | None = None,
) -> str:
    """呼叫 MLIS OpenAI 相容 endpoint 產生戰報文字，最多重試 MAX_RETRIES 次、間隔 RETRY_DELAY_SECONDS 秒。"""
    active_client = client or _build_client(config)

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = active_client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=8192,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:  # noqa: BLE001 - 統一轉為友善錯誤，由呼叫端顯示
            last_error = e
            logger.warning("MLIS 呼叫失敗（第 %d/%d 次）：%s", attempt + 1, MAX_RETRIES + 1, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"MLIS 服務呼叫失敗，已重試 {MAX_RETRIES} 次：{last_error}"
    ) from last_error


def test_connection(config: LLMConfig) -> tuple[bool, str]:
    """實際打一次 endpoint 驗證設定是否可用，供「系統設定」頁的測試連線按鈕使用。"""
    try:
        client = _build_client(config)
        client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
        )
        return True, "連線成功"
    except Exception as e:  # noqa: BLE001
        return False, f"連線失敗：{e}"
```

- [ ] **Step 5: 執行測試確認通過**

Run: `pytest tests/test_llm_client.py -v`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add src/app/llm_client.py tests/test_llm_client.py requirements.txt requirements-dev.txt
git commit -m "feat: 新增 llm_client 模組，透過 OpenAI 相容 API 呼叫 MLIS（含重試與設定讀取順序）"
```

---

### Task 7: `weekly_report_tab.py` 改接 MLIS（對應 spec §5）

移除 Gemini 三模型 fallback 迴圈與 `time.sleep(30)` 阻塞重試，改用 Task 6 的 `llm_client`。移除 `google-genai` 依賴。

**Files:**
- Modify: `src/app/tabs/weekly_report_tab.py`（`_get_gemini_key`/`GEMINI_MODELS`/`_call_gemini` 整段、`render()` 內「產生 AI 戰報」區塊——行號在 Task 2 加入 `_load_env_if_present` 後會位移，本 Task 一律以下方程式碼片段的內容比對定位，行號僅供參考）
- Modify: `requirements.txt`（移除 `google-genai==2.16.0`）
- Create: `tests/test_weekly_report_tab_mlis.py`

**Interfaces:**
- Consumes：`llm_client.LLMConfig`、`llm_client.resolve_llm_config`、`llm_client.generate_report`（Task 6）；`db_config.get_engine`（既有）。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_weekly_report_tab_mlis.py
"""
weekly_report_tab 改接 MLIS 後的行為：
- 未設定 MLIS 時顯示引導訊息（不再提示 Gemini API Key）
- 已設定 MLIS 時「產生 AI 戰報」按鈕會呼叫 llm_client.generate_report 並顯示結果
"""

import pandas as pd
from sqlalchemy import text
from streamlit.testing.v1 import AppTest

from src.etl.db_loader import insert_players, insert_teams
from src.etl.stats_crawler import upsert_stats


def _seed_one_match(engine) -> None:
    df = pd.DataFrame([{
        "team_id": 1, "team_name": "屏東台電", "gender": "M",
        "jersey_number": 4, "name": "李元", "position": "OH",
        "dob": None, "height_cm": None, "weight_kg": None,
    }])
    insert_teams(engine, df)
    insert_players(engine, df, season="2025-26")
    with engine.begin() as conn:
        pid = conn.execute(text("SELECT player_id FROM players WHERE name = '李元'")).scalar_one()
    row = dict(
        match_date="2026-01-05", opponent="雲林美津濃", sets_played=3,
        attack_total=10, attack_points=5, block_points=1,
        serve_total=5, serve_points=1, receive_total=5, receive_excellent=3,
        dig_total=5, dig_excellent=2, set_total=0, set_excellent=0,
        total_points=7, is_golden_set=0,
    )
    upsert_stats(engine, pid, [row], "2025-26")


def test_weekly_report_shows_guidance_when_mlis_not_configured(sqlite_engine, monkeypatch):
    for key in ("MLIS_BASE_URL", "MLIS_API_KEY", "MLIS_MODEL"):
        monkeypatch.delenv(key, raising=False)
    _seed_one_match(sqlite_engine)

    def _harness():
        import sys
        sys.path.insert(0, ".")
        from src.app.tabs import weekly_report_tab

        weekly_report_tab.render({})

    at = AppTest.from_function(_harness)
    at.run(timeout=30)

    assert not at.exception
    info_texts = [i.value for i in at.info]
    assert any("系統設定" in t for t in info_texts)


def test_weekly_report_generate_button_calls_llm_client_when_configured(sqlite_engine, monkeypatch):
    from src.app.settings_store import set_setting

    set_setting(sqlite_engine, "mlis_base_url", "http://fake-mlis.local/v1")
    set_setting(sqlite_engine, "mlis_api_key", "test-key")
    set_setting(sqlite_engine, "mlis_model", "qwen-test")
    _seed_one_match(sqlite_engine)

    import src.app.tabs.weekly_report_tab as weekly_report_tab_module
    monkeypatch.setattr(
        weekly_report_tab_module, "generate_report",
        lambda config, system_prompt, user_prompt: "模擬產生的戰報內容",
    )

    def _harness():
        import sys
        sys.path.insert(0, ".")
        from src.app.tabs import weekly_report_tab

        weekly_report_tab.render({})

    at = AppTest.from_function(_harness)
    at.run(timeout=30)
    assert not at.exception

    buttons = [b for b in at.button if b.label == "產生 AI 戰報"]
    assert buttons, "已設定 MLIS 時應顯示「產生 AI 戰報」按鈕"
    buttons[0].click().run(timeout=30)

    markdown_texts = [m.value for m in at.markdown]
    assert any("模擬產生的戰報內容" in t for t in markdown_texts)
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_weekly_report_tab_mlis.py -v`
Expected: FAIL —— 目前仍會顯示「請在 `.env` 或 Streamlit Secrets 中設定 `GOOGLE_API_KEY`」而非「系統設定」訊息；`generate_report` 尚未被 `weekly_report_tab` 引用（`AttributeError`）

- [ ] **Step 3: 移除 Gemini 相關程式碼，改接 MLIS**

刪除 `_get_gemini_key`、`GEMINI_MODELS`、`_call_gemini` 整段（Task 2 之後行號約在第 40-90 行附近，實際位置以下列內容比對定位，不依賴精確行號）：

```python
def _get_gemini_key() -> str | None:
    for key_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        try:
            return st.secrets[key_name]
        except (KeyError, FileNotFoundError):
            val = os.getenv(key_name)
            if val:
                return val
    return None


GEMINI_MODELS = ["gemini-3-flash-preview", "gemini-2.0-flash", "gemini-2.0-flash-lite"]


def _call_gemini(api_key: str, user_prompt: str) -> str:
    ...  # 整段刪除
```

在檔案開頭的 import 區塊（Task 2 已調整過）新增：

```python
from src.app.llm_client import generate_report, resolve_llm_config
from src.utils.db_config import get_engine
```

`render()` 內「產生 AI 戰報」區塊（Task 2 之後行號約在第 367-389 行附近，實際位置以下列內容比對定位）改為：

```python
    # ── 產生 AI 戰報（MLIS） ──────────────────────────────────
    engine = get_engine()
    llm_config = resolve_llm_config(engine)

    if not llm_config:
        st.info(
            "如需 AI 戰報功能，請至「系統設定」分頁設定 MLIS Endpoint、Model 與 API Key，"
            "或設定環境變數 `MLIS_BASE_URL` / `MLIS_API_KEY` / `MLIS_MODEL`。"
        )
    elif st.button("產生 AI 戰報", type="primary", key="wr_generate"):
        data_json = json.dumps(weekly_data, ensure_ascii=False, indent=2)
        user_prompt = (
            f"以下是 {weekly_data['period']} 的 TVL 企業排球聯賽比賽數據，"
            f"共 {len(grouped)} 場比賽。\n"
            f"請根據這些數據撰寫本周戰報。\n\n"
            f"```json\n{data_json}\n```"
        )
        with st.spinner("正在透過 MLIS 產生戰報，請稍候..."):
            try:
                report_text = generate_report(llm_config, REPORT_SYSTEM_PROMPT, user_prompt)
                st.session_state["weekly_report_text"] = report_text
                st.session_state["weekly_report_period"] = weekly_data["period"]
            except Exception as e:
                st.error(f"AI 戰報產生失敗：{e}")
```

- [ ] **Step 4: 移除 `google-genai` 依賴**

`requirements.txt` 移除 `google-genai==2.16.0` 一行，最終內容為：

```txt
beautifulsoup4==4.15.0
joblib==1.5.3
matplotlib==3.11.1
numpy==2.4.6
openai==2.53.0
pandas==3.0.5
plotly==6.9.0
psycopg[binary]==3.3.4
python-dotenv==1.2.2
requests==2.34.2
scikit-learn==1.9.0
shap==0.51.0
SQLAlchemy==2.0.51
streamlit==1.61.0
xgboost==3.2.0
```

- [ ] **Step 5: 執行測試確認通過**

Run: `pytest tests/test_weekly_report_tab_mlis.py -v`
Expected: 全部 PASS

- [ ] **Step 6: 執行完整測試套件**

Run: `pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 7: Commit**

```bash
git add src/app/tabs/weekly_report_tab.py requirements.txt tests/test_weekly_report_tab_mlis.py
git commit -m "feat: 每周戰報改接 MLIS，移除 Gemini fallback 迴圈與 google-genai 依賴"
```

---

### Task 8: 新增「系統設定」分頁（對應 spec §6）

新增第 7 個 tab，提供 MLIS endpoint base URL、model 名稱、API key 三個欄位與「測試連線」按鈕，API key 顯示時遮罩。整個 dashboard 已在平台 SSO 之後（計畫三範圍），設定頁不另做角色權限。

**Files:**
- Create: `src/app/tabs/settings_tab.py`
- Modify: `src/app/main.py:75,137-158`
- Create: `tests/test_settings_tab.py`

**Interfaces:**
- Produces: `settings_tab.render(ctx: dict) -> None`（第 7 個 tab，main.py 掛載）。
- Consumes：`settings_store.get_setting`/`set_setting`（Task 5）、`llm_client.LLMConfig`/`test_connection`（Task 6）。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_settings_tab.py
"""AppTest UI 測試：系統設定分頁（endpoint / model / API key + 測試連線）。"""

from streamlit.testing.v1 import AppTest


def test_settings_tab_shows_empty_state_when_nothing_saved(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        from src.app.tabs import settings_tab

        settings_tab.render({})

    at = AppTest.from_function(_harness)
    at.run(timeout=30)
    assert not at.exception
    captions = [c.value for c in at.caption]
    assert any("尚未設定" in c for c in captions)


def test_settings_tab_save_then_reload_shows_saved_values(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        from src.app.tabs import settings_tab

        settings_tab.render({})

    at = AppTest.from_function(_harness)
    at.run(timeout=30)

    at.text_input(key="settings_base_url").set_value("http://mlis.example/v1").run()
    at.text_input(key="settings_model").set_value("qwen2.5-72b").run()
    at.text_input(key="settings_api_key").set_value("super-secret-key-1234").run()
    at.button(key="FormSubmitter:mlis_settings_form-儲存設定").click().run(timeout=30)

    assert not at.exception

    from src.app.settings_store import get_setting
    from src.utils.db_config import get_engine

    engine = get_engine()
    assert get_setting(engine, "mlis_base_url") == "http://mlis.example/v1"
    assert get_setting(engine, "mlis_model") == "qwen2.5-72b"
    assert get_setting(engine, "mlis_api_key") == "super-secret-key-1234"

    # 迴歸測試：儲存後「同一次」render（提交表單觸發的這次 rerun）就應顯示剛儲存的遮罩
    # 後 API Key，而不是表單提交前讀到的舊值（舊值在此案例中是空字串 → 會誤顯示「尚未設定」）。
    captions = [c.value for c in at.caption]
    assert any("目前已儲存的 API Key" in c and "1234" in c for c in captions), (
        f"儲存後同一次 render 應立刻反映剛儲存的值，實際 captions={captions}"
    )


def test_settings_tab_test_connection_shows_success(sqlite_engine, monkeypatch):
    from src.app.settings_store import set_setting

    set_setting(sqlite_engine, "mlis_base_url", "http://mlis.example/v1")
    set_setting(sqlite_engine, "mlis_model", "qwen-test")
    set_setting(sqlite_engine, "mlis_api_key", "test-key")

    import src.app.tabs.settings_tab as settings_tab_module
    monkeypatch.setattr(settings_tab_module, "test_connection", lambda config: (True, "連線成功"))

    def _harness():
        import sys
        sys.path.insert(0, ".")
        from src.app.tabs import settings_tab

        settings_tab.render({})

    at = AppTest.from_function(_harness)
    at.run(timeout=30)
    at.button(key="settings_test_connection").click().run(timeout=30)

    assert not at.exception
    successes = [s.value for s in at.success]
    assert any("連線成功" in s for s in successes)
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_settings_tab.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.app.tabs.settings_tab'`

- [ ] **Step 3: 建立 `settings_tab.py`**

```python
# src/app/tabs/settings_tab.py
"""
Tab 7：系統設定
設定 PCAI MLIS 的 OpenAI 相容 endpoint（base URL、model、API key），供「每周戰報」分頁呼叫。
整個 dashboard 已在平台 SSO 之後，本頁不另做角色權限。
"""

import streamlit as st

from src.app.llm_client import LLMConfig, test_connection
from src.app.settings_store import get_setting, set_setting
from src.utils.db_config import get_engine

SETTING_KEYS = {
    "base_url": "mlis_base_url",
    "model": "mlis_model",
    "api_key": "mlis_api_key",
}


def _mask_secret(value: str) -> str:
    """遮罩顯示：保留最後 4 碼，其餘以 * 取代；長度不足 4 則全部遮罩。"""
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return f"{'*' * (len(value) - 4)}{value[-4:]}"


def render(ctx: dict) -> None:
    st.subheader("系統設定")
    st.caption("設定 PCAI MLIS 的 OpenAI 相容 endpoint，供「每周戰報」分頁呼叫 AI 產生戰報。")

    engine = get_engine()

    current_base_url = get_setting(engine, SETTING_KEYS["base_url"]) or ""
    current_model = get_setting(engine, SETTING_KEYS["model"]) or ""
    current_api_key = get_setting(engine, SETTING_KEYS["api_key"]) or ""

    with st.form("mlis_settings_form"):
        base_url = st.text_input(
            "Endpoint Base URL", value=current_base_url,
            placeholder="http://mlis-qwen.example.svc.cluster.local/v1",
            key="settings_base_url",
        )
        model = st.text_input(
            "Model 名稱", value=current_model,
            placeholder="qwen2.5-72b-instruct",
            key="settings_model",
        )
        api_key = st.text_input(
            "API Key（留空表示不變更既有金鑰）", value="", type="password",
            placeholder=_mask_secret(current_api_key) or "尚未設定",
            key="settings_api_key",
        )
        submitted = st.form_submit_button("儲存設定")

    if submitted:
        set_setting(engine, SETTING_KEYS["base_url"], base_url.strip())
        set_setting(engine, SETTING_KEYS["model"], model.strip())
        if api_key:
            set_setting(engine, SETTING_KEYS["api_key"], api_key.strip())
        st.success("設定已儲存。")
        # 重新讀取：若不重新讀取，儲存當下這次 rerun 仍會用表單提交「前」讀到的舊
        # current_api_key 渲染下方的遮罩提示，導致剛存完金鑰卻仍顯示「尚未設定」。
        current_base_url = get_setting(engine, SETTING_KEYS["base_url"]) or ""
        current_model = get_setting(engine, SETTING_KEYS["model"]) or ""
        current_api_key = get_setting(engine, SETTING_KEYS["api_key"]) or ""

    st.markdown("---")
    if current_api_key:
        st.caption(f"目前已儲存的 API Key：`{_mask_secret(current_api_key)}`")
    else:
        st.caption("尚未設定 API Key。")

    if st.button("測試連線", key="settings_test_connection"):
        test_base_url = (base_url or current_base_url).strip()
        test_model = (model or current_model).strip()
        test_api_key = (api_key or current_api_key).strip()
        if not (test_base_url and test_model and test_api_key):
            st.warning("請先完整填寫 Endpoint Base URL、Model 名稱與 API Key。")
        else:
            config = LLMConfig(base_url=test_base_url, api_key=test_api_key, model=test_model)
            ok, message = test_connection(config)
            if ok:
                st.success(message)
            else:
                st.error(message)
```

- [ ] **Step 4: 掛載到 `main.py`**

第 75 行改為：

```python
from src.app.tabs import (
    player_deep, league_pr, match_trend, box_score, prediction,
    weekly_report_tab, settings_tab,
)
```

第 137-158 行改為：

```python
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "球員個人深度", "聯盟 PR 值與分佈", "逐場趨勢", "單場 Box Score", "賽果預測",
    "每周戰報", "系統設定",
])

with tab1:
    player_deep.render(ctx)

with tab2:
    league_pr.render(ctx)

with tab3:
    match_trend.render(ctx)

with tab4:
    box_score.render(ctx)

with tab5:
    prediction.render(ctx, cjk_font_path=_CJK_FONT_PATH, cjk_font_stack=CJK_FONT_STACK)

with tab6:
    weekly_report_tab.render(ctx)

with tab7:
    settings_tab.render(ctx)
```

- [ ] **Step 5: 執行測試確認通過**

Run: `pytest tests/test_settings_tab.py -v`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add src/app/tabs/settings_tab.py src/app/main.py tests/test_settings_tab.py
git commit -m "feat: 新增系統設定分頁，可設定 MLIS endpoint/model/API key 並測試連線"
```

---

### Task 9: 賽季選擇器（對應 spec §7，解決計畫一已知風險 #6）

sidebar 最上層加賽季下拉（選項 = DB 中 distinct season，預設最新），選定的 season 經 `ctx` 傳給所有 tab；`get_league_aggregated_stats`、`player_deep._load_league_agg`、`box_score` 的逐場查詢、`weekly_report.py` 的週次/週報彙整皆加上 season 過濾。這同時解決「跨季後聯盟 PR 頁同一人出現兩筆」的問題——修正前，`league_pr.py` 的排名表會把不同賽季的同一位球員（不同 `player_id`）都列進同一張百分位排名表。

**Files:**
- Modify: `src/app/main.py:1-6,80-133`
- Modify: `src/app/helpers.py:205-211`（`get_league_aggregated_stats` 簽名加 `season`）
- Modify: `src/app/tabs/league_pr.py:18-34`
- Modify: `src/app/tabs/player_deep.py:28-44,66-72,109,164-167`
- Modify: `src/app/tabs/box_score.py:57-68,80-89,155-172,180-197`
- Modify: `src/app/tabs/weekly_report_tab.py:314-345`
- Modify: `src/app/tabs/prediction.py:18-23,142`（`_get_data_ranges` 呼叫 `get_league_aggregated_stats` 處——簽名改變後若不同步更新，`_get_data_ranges` 的 bare `except Exception: return {}` 會把因簽名不符產生的 `TypeError` 靜默吞掉，讓預測頁的資料驅動 slider 範圍悄悄失效且無任何錯誤訊息）
- Modify: `src/etl/weekly_report.py:1-121`（`get_match_weeks`/`gather_weekly_data` 加 `season` 參數）
- Modify: `tests/test_weekly_report.py`
- Modify: `tests/test_streamlit_ui.py`（Task 2 建立，`box_score`/`league_pr`/`player_deep`/`weekly_report_tab` 的 harness ctx 需補上 `season`）
- Modify: `tests/test_weekly_report_tab_mlis.py`（Task 7 建立，harness 呼叫需補上 season 相關資料）
- Modify: `tests/test_prediction_slider_config.py`（Task 3 建立，新增 `_get_data_ranges` 的迴歸測試）
- Create: `tests/test_season_filtering.py`

**Interfaces:**
- Produces: `helpers.get_league_aggregated_stats(gender_code: str, season: str) -> pd.DataFrame`（在 Task 1 的簽名上新增 `season` 必要參數）。
- Produces: `weekly_report.get_match_weeks(season: str) -> list[tuple[str, str]]`、`weekly_report.gather_weekly_data(date_from: str, date_to: str, season: str, gender_filter: str | None = None) -> dict`（在既有簽名上新增 `season` 必要參數，插入於 `gender_filter` 之前）。
- Produces: `prediction._get_data_ranges(gender_code: str, season: str) -> dict[str, tuple[float, float]]`（在既有簽名上新增 `season` 必要參數，與 `get_league_aggregated_stats` 同步）。
- Consumes：`constants.SEASON`（既有，作為 sidebar 找不到任何 season 時的 fallback 預設值，`prediction.py` 呼叫端也用它作為 `ctx` 缺少 `season` 時的防禦性預設值）。

- [ ] **Step 1: 寫失敗測試 —— 核心 season 過濾邏輯**

```python
# tests/test_season_filtering.py
"""
賽季過濾迴歸測試：確保換季後，聯盟聚合與週報彙整不會把不同賽季的同名球員混在一起
（對應計畫一「已知風險 #6」：跨賽季後聯盟 PR 頁同一人出現兩筆）。
"""

import pandas as pd
from sqlalchemy import text

from src.etl.db_loader import insert_players, insert_teams
from src.etl.stats_crawler import upsert_stats
from src.etl.weekly_report import gather_weekly_data, get_match_weeks


def _seed_two_seasons(engine) -> None:
    df = pd.DataFrame([{
        "team_id": 1, "team_name": "屏東台電", "gender": "M",
        "jersey_number": 4, "name": "李元", "position": "OH",
        "dob": None, "height_cm": None, "weight_kg": None,
    }])
    insert_teams(engine, df)

    for season, match_date in [("2025-26", "2026-01-05"), ("2026-27", "2026-11-10")]:
        insert_players(engine, df, season=season)
        with engine.begin() as conn:
            pid = conn.execute(
                text("SELECT player_id FROM players WHERE name = '李元' AND season = :s"),
                {"s": season},
            ).scalar_one()
        row = dict(
            match_date=match_date, opponent="雲林美津濃", sets_played=3,
            attack_total=10, attack_points=5, block_points=1,
            serve_total=5, serve_points=1, receive_total=5, receive_excellent=3,
            dig_total=5, dig_excellent=2, set_total=0, set_excellent=0,
            total_points=7, is_golden_set=0,
        )
        upsert_stats(engine, pid, [row], season)


def test_get_league_aggregated_stats_filters_by_season(sqlite_engine):
    from src.app.helpers import get_league_aggregated_stats

    _seed_two_seasons(sqlite_engine)
    get_league_aggregated_stats.clear()

    df_2025 = get_league_aggregated_stats("M", "2025-26")
    df_2026 = get_league_aggregated_stats("M", "2026-27")

    # 每個賽季各自只看到「李元」一筆，不會因為換季而出現兩筆同名球員
    assert len(df_2025[df_2025["name"] == "李元"]) == 1
    assert len(df_2026[df_2026["name"] == "李元"]) == 1
    assert df_2025["player_id"].iloc[0] != df_2026["player_id"].iloc[0]


def test_get_match_weeks_filters_by_season(sqlite_engine):
    _seed_two_seasons(sqlite_engine)

    weeks_2025 = get_match_weeks("2025-26")
    weeks_2026 = get_match_weeks("2026-27")

    assert weeks_2025 == [("2026-01-05", "2026-01-05")]
    assert weeks_2026 == [("2026-11-10", "2026-11-10")]


def test_gather_weekly_data_filters_by_season(sqlite_engine):
    _seed_two_seasons(sqlite_engine)

    result_2025 = gather_weekly_data("2026-01-01", "2026-01-10", "2025-26")
    result_2026 = gather_weekly_data("2026-11-01", "2026-11-15", "2026-27")

    assert len(result_2025["matches"]) == 1
    assert len(result_2026["matches"]) == 1

    # 舊賽季範圍查詢新賽季時應為空（season 過濾優先於日期範圍巧合重疊的風險）
    cross_season = gather_weekly_data("2026-01-01", "2026-01-10", "2026-27")
    assert cross_season["matches"] == []
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_season_filtering.py -v`
Expected: FAIL — `TypeError: get_league_aggregated_stats() takes 1 positional argument but 2 were given`；`get_match_weeks() takes 0 positional arguments but 1 was given`；`gather_weekly_data() takes from 2 to 3 positional arguments but 4 were given`

- [ ] **Step 3: `helpers.py` —— `get_league_aggregated_stats` 加 `season` 參數**

第 205-239 行（Task 1 版本基礎上）改為：

```python
@st.cache_data(ttl=3600)
def get_league_aggregated_stats(gender_code: str, season: str) -> pd.DataFrame:
    """
    撈取指定賽季、該組別所有球員的聚合統計數據，JOIN players + teams 取得姓名/球隊/位置。
    僅保留總局數 >= 5 的球員，排除極端值。season 過濾避免跨季後同一人出現兩筆（不同 player_id）。
    """
    raw = load_data(
        """
        SELECT p.player_id,
               p.name,
               p.position,
               t.team_name,
               SUM(s.sets_played)       AS total_sets,
               SUM(s.attack_points)     AS atk_pts,
               SUM(s.attack_total)      AS atk_tot,
               SUM(s.block_points)      AS blk_pts,
               SUM(s.serve_points)      AS srv_pts,
               SUM(s.serve_total)       AS srv_tot,
               SUM(s.receive_excellent) AS rcv_exc,
               SUM(s.receive_total)     AS rcv_tot,
               SUM(s.dig_excellent)     AS dig_exc,
               SUM(s.dig_total)         AS dig_tot,
               SUM(s.set_excellent)     AS set_exc,
               SUM(s.set_total)         AS set_tot,
               SUM(s.total_points)      AS total_points,
               COUNT(*)                 AS n_games
        FROM player_match_stats s
        JOIN players p ON s.player_id = p.player_id
        JOIN teams   t ON p.team_id = t.team_id AND p.gender = t.gender
        WHERE p.gender = :gender_code AND p.season = :season
        GROUP BY p.player_id
        HAVING SUM(s.sets_played) >= 5
        """,
        {"gender_code": gender_code, "season": season},
    )
    raw["asr"] = vec_pct(raw["atk_pts"], raw["atk_tot"])
    raw["gp_pct"] = vec_pct(raw["rcv_exc"], raw["rcv_tot"])
    raw["ace_pct"] = vec_pct(raw["srv_pts"], raw["srv_tot"])
    raw["dig_pct"] = vec_pct(raw["dig_exc"], raw["dig_tot"])
    raw["ppg"] = np.where(raw["n_games"] > 0, raw["total_points"] / raw["n_games"], 0.0)
    raw["set_pct"] = vec_pct(raw["set_exc"], raw["set_tot"])
    raw["blk_per_set"] = np.where(raw["total_sets"] > 0, raw["blk_pts"] / raw["total_sets"], 0.0)
    raw["def_load"] = raw["rcv_tot"] + raw["dig_tot"]
    raw["def_pct"] = vec_pct(raw["rcv_exc"] + raw["dig_exc"], raw["rcv_tot"] + raw["dig_tot"])

    pr_cols = ["asr", "gp_pct", "ace_pct", "dig_pct", "set_pct", "blk_per_set", "def_pct"]
    for col in pr_cols:
        raw[f"{col}_pr"] = (
            raw.groupby("position")[col]
            .rank(pct=True)
            .mul(100)
            .round(1)
        )

    return raw
```

- [ ] **Step 4: 寫失敗測試 —— `prediction.py` 的 `_get_data_ranges` 呼叫端**

`get_league_aggregated_stats` 的簽名在 Step 3 已改為 `(gender_code, season)`，但 `src/app/tabs/prediction.py:22-23` 的 `_get_data_ranges` 仍以單一參數呼叫它；`_get_data_ranges` 整段包在 `except Exception: return {}` 裡，簽名不符產生的 `TypeError` 會被靜默吞掉、回傳 `{}`，讓預測頁的資料驅動 slider 範圍悄悄失效且沒有任何錯誤訊息。以下測試繞開 bare except 的遮蔽，直接斷言回傳值非空：

```python
# tests/test_prediction_slider_config.py（新增在既有檔案末尾）
import pandas as pd
from sqlalchemy import text

from src.etl.db_loader import insert_players, insert_teams
from src.etl.stats_crawler import upsert_stats


def _seed_league_data_for_ranges(engine, season: str) -> None:
    """插入一筆 sets_played=5（剛好達到 get_league_aggregated_stats 的 HAVING 門檻）的球員數據。"""
    df = pd.DataFrame([{
        "team_id": 1, "team_name": "屏東台電", "gender": "M",
        "jersey_number": 4, "name": "李元", "position": "OH",
        "dob": None, "height_cm": None, "weight_kg": None,
    }])
    insert_teams(engine, df)
    insert_players(engine, df, season=season)
    with engine.begin() as conn:
        pid = conn.execute(
            text("SELECT player_id FROM players WHERE name = '李元' AND season = :s"),
            {"s": season},
        ).scalar_one()
    row = dict(
        match_date="2026-01-05", opponent="雲林美津濃", sets_played=5,
        attack_total=20, attack_points=10, block_points=2,
        serve_total=10, serve_points=2, receive_total=10, receive_excellent=6,
        dig_total=10, dig_excellent=4, set_total=0, set_excellent=0,
        total_points=14, is_golden_set=0,
    )
    upsert_stats(engine, pid, [row], season)


def test_get_data_ranges_does_not_silently_swallow_season_mismatch(sqlite_engine):
    """
    迴歸測試：_get_data_ranges 若沒有同步傳入 season，get_league_aggregated_stats 會拋
    TypeError，且該 TypeError 會被 _get_data_ranges 的 bare except 吞掉、回傳 {}。
    這裡斷言回傳值「非空」，確保呼叫端簽名確實同步更新，而不是被 except 悄悄蓋過去。
    """
    from src.app.tabs.prediction import _get_data_ranges

    _seed_league_data_for_ranges(sqlite_engine, season="2025-26")
    _get_data_ranges.clear()

    ranges = _get_data_ranges("M", "2025-26")

    assert ranges != {}, "_get_data_ranges 回傳空字典——很可能是呼叫端與 season 簽名不同步，TypeError 被 bare except 吞掉"
    assert "ASR" in ranges
```

- [ ] **Step 5: 執行測試確認失敗**

Run: `pytest tests/test_prediction_slider_config.py::test_get_data_ranges_does_not_silently_swallow_season_mismatch -v`
Expected: FAIL — `assert ranges != {}` 失敗（`_get_data_ranges("M", "2025-26")` 目前簽名是 `_get_data_ranges(gender_code)` 只接受一個參數，呼叫時傳兩個會直接 `TypeError: _get_data_ranges() takes 1 positional argument but 2 were given`，測試本身也會因此 FAIL，只是失敗形式是例外而非斷言）

- [ ] **Step 6: 修改 `prediction.py` 的 `_get_data_ranges` 與呼叫端**

第 18-23 行改為：

```python
from src.utils.constants import SEASON

@st.cache_data
def _get_data_ranges(gender_code: str, season: str) -> dict[str, tuple[float, float]]:
    """從指定賽季的聯盟聚合數據計算各滑桿指標的實際 (min, max)，加 10% 緩衝。"""
    try:
        from src.app.helpers import get_league_aggregated_stats
        df = get_league_aggregated_stats(gender_code, season)
        if df.empty:
            return {}
```

（`from src.utils.constants import SEASON` 加在檔案頂部既有 import 區塊，`import numpy as np` 等既有 import 之後；`_get_data_ranges` 函式本體其餘部分——`_range` 內部函式與 `ranges` dict 組裝——維持不變。）

第 142 行（`render()` 內呼叫端）改為：

```python
    _data_ranges = _get_data_ranges(ctx.get("gender_code", "M"), ctx.get("season", SEASON))
```

- [ ] **Step 7: 執行測試確認通過**

Run: `pytest tests/test_prediction_slider_config.py -v`
Expected: 全部 PASS（含 Task 3 既有的 `_select_slider_config` 測試與本 Step 新增的測試）

- [ ] **Step 8: `src/etl/weekly_report.py` —— `get_match_weeks`/`gather_weekly_data` 加 `season` 參數**

```python
def get_match_weeks(season: str) -> list[tuple[str, str]]:
    """
    回傳指定賽季所有比賽周次的 (week_start, week_end) 列表。
    以 ISO 周次分組，方便使用者選擇。
    """
    engine = get_engine()
    dates = pd.read_sql(
        text("SELECT DISTINCT match_date FROM player_match_stats WHERE season = :season ORDER BY match_date"),
        engine,
        params={"season": season},
    )["match_date"].tolist()

    if not dates:
        return []

    weeks: dict[tuple[int, int], list[str]] = {}
    for d in dates:
        dt = datetime.strptime(d, "%Y-%m-%d")
        iso_year, iso_week, _ = dt.isocalendar()
        key = (iso_year, iso_week)
        weeks.setdefault(key, []).append(d)

    result = []
    for (_y, _w), ds in sorted(weeks.items()):
        result.append((min(ds), max(ds)))
    return result


def gather_weekly_data(
    date_from: str, date_to: str, season: str, gender_filter: str | None = None
) -> dict:
    """
    彙整指定賽季、指定日期範圍內的所有比賽數據，回傳結構化 dict。

    Parameters
    ----------
    date_from : 起始日期 (YYYY-MM-DD)
    date_to : 結束日期 (YYYY-MM-DD)
    season : 賽季字串（如 "2025-26"），避免跨季後同一人（不同 player_id）混入同一份彙整
    gender_filter : "M", "F", or None (全部)

    Returns
    -------
    dict with keys: "period", "matches"
    """
    engine = get_engine()
    gender_clause = "AND p.gender = :gender_filter" if gender_filter else ""

    params: dict = {"date_from": date_from, "date_to": date_to, "season": season}
    if gender_filter:
        params["gender_filter"] = gender_filter

    raw = pd.read_sql(
        text(f"""
            SELECT s.match_date, s.opponent,
                   p.player_id, p.name, p.position, p.gender,
                   t.team_id, t.team_name,
                   s.sets_played,
                   s.attack_total, s.attack_points,
                   s.block_points,
                   s.serve_total, s.serve_points,
                   s.receive_total, s.receive_excellent,
                   s.dig_total, s.dig_excellent,
                   s.set_total, s.set_excellent,
                   s.total_points,
                   s.is_golden_set
            FROM player_match_stats s
            JOIN players p ON s.player_id = p.player_id
            JOIN teams   t ON p.team_id = t.team_id AND p.gender = t.gender
            WHERE s.match_date BETWEEN :date_from AND :date_to
              AND s.season = :season
            {gender_clause}
            ORDER BY s.match_date, t.team_name
        """),
        engine,
        params=params,
    )

    season_params: dict = {"date_to": date_to, "season": season}
    if gender_filter:
        season_params["gender_filter"] = gender_filter

    season_agg = pd.read_sql(
        text(f"""
            SELECT p.player_id, p.name, p.position, p.gender,
                   t.team_name,
                   COUNT(*) AS season_games,
                   SUM(s.sets_played) AS season_sets,
                   SUM(s.attack_points) AS season_atk_pts,
                   SUM(s.attack_total) AS season_atk_tot,
                   SUM(s.block_points) AS season_blk_pts,
                   SUM(s.serve_points) AS season_srv_pts,
                   SUM(s.serve_total) AS season_srv_tot,
                   SUM(s.receive_excellent) AS season_rcv_exc,
                   SUM(s.receive_total) AS season_rcv_tot,
                   SUM(s.dig_excellent) AS season_dig_exc,
                   SUM(s.dig_total) AS season_dig_tot,
                   SUM(s.set_excellent) AS season_set_exc,
                   SUM(s.set_total) AS season_set_tot,
                   SUM(s.total_points) AS season_total_pts
            FROM player_match_stats s
            JOIN players p ON s.player_id = p.player_id
            JOIN teams   t ON p.team_id = t.team_id AND p.gender = t.gender
            WHERE s.match_date <= :date_to AND s.is_golden_set = 0
              AND s.season = :season
            {gender_clause}
            GROUP BY p.player_id
            HAVING COUNT(*) >= 2
        """),
        engine,
        params=season_params,
    )

    if raw.empty:
        return {"period": f"{date_from} ~ {date_to}", "matches": []}

    # ── 以下（球員賽季平均查找表、正規賽/黃金局分組、matches 組裝）與既有邏輯完全相同，不變 ──
    season_lookup = {}
    for _, row in season_agg.iterrows():
        pid = row["player_id"]
        g = row["season_games"]
        atk_tot = row["season_atk_tot"] or 0
        season_lookup[pid] = {
            "season_games": int(g),
            "season_ppg": round(row["season_total_pts"] / g, 1) if g else 0,
            "season_asr": round(
                row["season_atk_pts"] / atk_tot * 100, 1
            ) if atk_tot > 0 else None,
        }

    def _safe_pct(num, den):
        return round(num / den * 100, 1) if den and den > 0 else None

    raw_regular = raw[raw["is_golden_set"] == 0]
    raw_golden = raw[raw["is_golden_set"] == 1]

    golden_lookup: dict[tuple, dict] = {}
    for (date, team_name, opponent), grp in raw_golden.groupby(
        ["match_date", "team_name", "opponent"]
    ):
        gs_team_stats = {
            "total_points": int(grp["total_points"].sum()),
            "attack_points": int(grp["attack_points"].sum()),
            "attack_total": int(grp["attack_total"].sum()),
            "block_points": int(grp["block_points"].sum()),
            "serve_points": int(grp["serve_points"].sum()),
        }
        gs_players = []
        for _, p in grp.sort_values("total_points", ascending=False).iterrows():
            if p["total_points"] == 0:
                continue
            gs_players.append({
                "name": p["name"],
                "position": p["position"],
                "total_points": int(p["total_points"]),
                "attack_points": int(p["attack_points"]),
                "block_points": int(p["block_points"]),
                "serve_points": int(p["serve_points"]),
            })
        golden_lookup[(date, team_name, opponent)] = {
            "team_stats": gs_team_stats,
            "players": gs_players,
        }

    matches = []
    for (date, team_name, opponent), grp in raw_regular.groupby(
        ["match_date", "team_name", "opponent"]
    ):
        gender = grp.iloc[0]["gender"]
        gender_label = "男子組" if gender == "M" else "女子組"

        team_stats = {
            "total_points": int(grp["total_points"].sum()),
            "attack_points": int(grp["attack_points"].sum()),
            "attack_total": int(grp["attack_total"].sum()),
            "attack_rate": _safe_pct(grp["attack_points"].sum(), grp["attack_total"].sum()),
            "block_points": int(grp["block_points"].sum()),
            "serve_points": int(grp["serve_points"].sum()),
            "serve_total": int(grp["serve_total"].sum()),
        }

        players = []
        for _, p in grp.sort_values("total_points", ascending=False).iterrows():
            if p["sets_played"] == 0:
                continue
            pid = int(p["player_id"])
            pdata = {
                "name": p["name"],
                "position": p["position"],
                "sets_played": int(p["sets_played"]),
                "total_points": int(p["total_points"]),
                "attack_points": int(p["attack_points"]),
                "attack_total": int(p["attack_total"]),
                "attack_rate": _safe_pct(p["attack_points"], p["attack_total"]),
                "block_points": int(p["block_points"]),
                "serve_points": int(p["serve_points"]),
                "receive_excellent": int(p["receive_excellent"]),
                "receive_total": int(p["receive_total"]),
                "receive_rate": _safe_pct(p["receive_excellent"], p["receive_total"]),
                "dig_excellent": int(p["dig_excellent"]),
                "dig_total": int(p["dig_total"]),
            }
            if pid in season_lookup:
                sl = season_lookup[pid]
                pdata["season_ppg"] = sl["season_ppg"]
                pdata["season_asr"] = sl["season_asr"]
                pdata["vs_season_ppg"] = round(
                    p["total_points"] - sl["season_ppg"], 1
                )
            players.append(pdata)

        match_entry = {
            "date": date,
            "gender": gender_label,
            "team_name": team_name,
            "opponent": opponent,
            "team_stats": team_stats,
            "players": players,
        }

        gs_key = (date, team_name, opponent)
        if gs_key in golden_lookup:
            match_entry["golden_set"] = golden_lookup[gs_key]

        matches.append(match_entry)

    return {
        "period": f"{date_from} ~ {date_to}",
        "matches": matches,
    }
```

- [ ] **Step 9: 執行核心測試確認通過**

Run: `pytest tests/test_season_filtering.py -v`
Expected: 全部 PASS

- [ ] **Step 10: `tests/test_weekly_report.py` —— 更新既有呼叫端以符合新簽名**

```python
# tests/test_weekly_report.py
from sqlalchemy import text

from src.etl.db_loader import insert_players, insert_teams
from src.etl.stats_crawler import upsert_stats
from src.etl.weekly_report import gather_weekly_data, get_match_weeks
import pandas as pd


def _seed(engine) -> int:
    df = pd.DataFrame([{
        "team_id": 1, "team_name": "屏東台電", "gender": "M",
        "jersey_number": 4, "name": "李元", "position": "OH",
        "dob": None, "height_cm": None, "weight_kg": None,
    }])
    insert_teams(engine, df)
    insert_players(engine, df, season="2025-26")
    with engine.begin() as conn:
        pid = conn.execute(
            text("SELECT player_id FROM players WHERE name = '李元'")
        ).scalar_one()

    row = dict(
        match_date="2026-01-05", opponent="雲林美津濃", sets_played=3,
        attack_total=10, attack_points=5, block_points=1,
        serve_total=5, serve_points=1, receive_total=5, receive_excellent=3,
        dig_total=5, dig_excellent=2, set_total=0, set_excellent=0,
        total_points=7, is_golden_set=0,
    )
    upsert_stats(engine, pid, [row], "2025-26")
    return pid


def test_get_match_weeks_returns_week_ranges(sqlite_engine):
    _seed(sqlite_engine)
    weeks = get_match_weeks("2025-26")
    assert weeks == [("2026-01-05", "2026-01-05")]


def test_gather_weekly_data_filters_by_date_range(sqlite_engine):
    _seed(sqlite_engine)
    result = gather_weekly_data("2026-01-01", "2026-01-10", "2025-26")
    assert result["period"] == "2026-01-01 ~ 2026-01-10"
    assert len(result["matches"]) == 1
    assert result["matches"][0]["team_name"] == "屏東台電"
    assert result["matches"][0]["opponent"] == "雲林美津濃"


def test_gather_weekly_data_filters_by_gender(sqlite_engine):
    _seed(sqlite_engine)
    result_f = gather_weekly_data("2026-01-01", "2026-01-10", "2025-26", gender_filter="F")
    assert result_f["matches"] == []

    result_m = gather_weekly_data("2026-01-01", "2026-01-10", "2025-26", gender_filter="M")
    assert len(result_m["matches"]) == 1
```

- [ ] **Step 11: `league_pr.py` —— 讀取 `ctx["season"]` 並傳入**

第 18-34 行（`render()` 開頭）改為：

```python
def render(ctx: dict) -> None:
    """Render the league PR tab.

    Parameters
    ----------
    ctx : dict
        Must contain keys: player_id, player_name, player_position,
        gender_code, gender, team_name, season.
    """
    player_id = ctx["player_id"]
    player_name = ctx["player_name"]
    player_position = ctx["player_position"]
    gender_code = ctx["gender_code"]
    gender = ctx["gender"]
    team_name = ctx["team_name"]
    season = ctx["season"]

    league_all = get_league_aggregated_stats(gender_code, season)
```

- [ ] **Step 12: `player_deep.py` —— 讀取 `ctx["season"]` 並傳入 `_load_league_agg`**

第 28-44 行（`_load_league_agg`）加上 `season` 參數：

```python
def _load_league_agg(gender_code: str, season: str, pos_filter: str = "", params: dict | None = None):
    """撈取指定賽季的聯盟聚合數據（全組別或特定位置）。"""
    return load_data(
        f"""
        SELECT SUM(s.attack_points) AS atk_pts, SUM(s.attack_total) AS atk_tot,
               SUM(s.receive_excellent) AS rcv_exc, SUM(s.receive_total) AS rcv_tot,
               SUM(s.serve_points) AS srv_pts, SUM(s.serve_total) AS srv_tot,
               SUM(s.dig_excellent) AS dig_exc, SUM(s.dig_total) AS dig_tot,
               SUM(s.set_excellent) AS set_exc, SUM(s.set_total) AS set_tot,
               SUM(s.block_points) AS blk_pts, SUM(s.sets_played) AS tot_sets,
               SUM(s.total_points) AS tot_pts, COUNT(*) AS n_games
        FROM player_match_stats s
        JOIN players p ON s.player_id = p.player_id
        WHERE p.gender = :gender_code AND p.season = :season {pos_filter}
        """,
        params if params is not None else {"gender_code": gender_code, "season": season},
    ).iloc[0]
```

第 66-72 行（`render()` 開頭）加上 `season = ctx["season"]`：

```python
def render(ctx: dict):
    player_id = ctx["player_id"]
    player_name = ctx["player_name"]
    player_position = ctx["player_position"]
    gender_code = ctx["gender_code"]
    gender = ctx["gender"]
    season = ctx["season"]
```

第 109 行（全聯盟平均）：

```python
    la = _parse_agg(_load_league_agg(gender_code, season, params={"gender_code": gender_code, "season": season}))
```

第 164-167 行（同位置平均）：

```python
    pos_filter = "AND p.position = :position" if player_position else ""
    pos_params = (
        {"gender_code": gender_code, "season": season, "position": player_position}
        if player_position else {"gender_code": gender_code, "season": season}
    )
    lg = _parse_agg(_load_league_agg(gender_code, season, pos_filter, pos_params))
```

- [ ] **Step 13: `box_score.py` —— 讀取 `ctx["season"]` 並加入 3 處查詢**

第 57-68 行（`render()` 開頭 + `bs_teams` 查詢，season 無關但先取出）改為：

```python
def render(ctx: dict):
    season = ctx["season"]

    # ── 篩選器：性別 → 球隊 → 比賽場次 ──────────────────────
    f1, f2, f3 = st.columns(3)

    with f1:
        bs_gender = st.selectbox("組別", ["男子組", "女子組"], key="bs_gender")
    bs_gender_code = "M" if bs_gender == "男子組" else "F"

    bs_teams = load_data(
        "SELECT team_id, team_name FROM teams WHERE gender = :gender_code ORDER BY team_id",
        {"gender_code": bs_gender_code},
    )
```

第 80-89 行（`matches_df`）加上 `s.season`：

```python
    matches_df = load_data(
        """
        SELECT DISTINCT s.match_date, s.opponent
        FROM player_match_stats s
        JOIN players p ON s.player_id = p.player_id
        WHERE p.team_id = :team_id AND p.gender = :gender_code AND s.season = :season
        ORDER BY s.match_date
        """,
        {"team_id": bs_team_id, "gender_code": bs_gender_code, "season": season},
    )
```

第 155-172 行（`team_a_df`）加上 `s.season`：

```python
    team_a_df = load_data(
        """
        SELECT p.name, p.position, s.sets_played,
               s.attack_points, s.attack_total,
               s.block_points,
               s.serve_points, s.serve_total,
               s.receive_excellent, s.receive_total,
               s.dig_excellent, s.dig_total,
               s.set_excellent, s.set_total,
               s.total_points
        FROM player_match_stats s
        JOIN players p ON s.player_id = p.player_id
        WHERE p.team_id = :team_id AND p.gender = :gender_code AND s.season = :season
          AND s.match_date = :match_date AND s.opponent = :opponent
        ORDER BY s.total_points DESC
        """,
        {
            "team_id": bs_team_id, "gender_code": bs_gender_code, "season": season,
            "match_date": sel_date, "opponent": sel_opponent,
        },
    )
```

第 180-197 行（`team_b_df`）加上 `s.season`：

```python
        team_b_df = load_data(
            """
            SELECT p.name, p.position, s.sets_played,
                   s.attack_points, s.attack_total,
                   s.block_points,
                   s.serve_points, s.serve_total,
                   s.receive_excellent, s.receive_total,
                   s.dig_excellent, s.dig_total,
                   s.set_excellent, s.set_total,
                   s.total_points
            FROM player_match_stats s
            JOIN players p ON s.player_id = p.player_id
            WHERE p.team_id = :team_id AND p.gender = :gender_code AND s.season = :season
              AND s.match_date = :match_date
            ORDER BY s.total_points DESC
            """,
            {"team_id": opp_team_id, "gender_code": opp_gender, "season": season, "match_date": sel_date},
        )
```

- [ ] **Step 14: `weekly_report_tab.py` —— 讀取 `ctx["season"]` 並傳入 `get_match_weeks`/`gather_weekly_data`**

第 314-345 行（`render()` 開頭至撈取資料段落）改為：

```python
def render(ctx):
    season = ctx["season"]

    st.subheader("每周戰報")
    st.caption("根據比賽數據，透過 MLIS AI 自動產生結構化中文戰報。")

    # ── 周次選擇器 ────────────────────────────────────────────
    weeks = get_match_weeks(season)
    if not weeks:
        st.info("資料庫中尚無比賽紀錄。")
        return

    week_labels = [
        f"第 {i+1} 周：{w[0]} ~ {w[1]}" for i, w in enumerate(weeks)
    ]

    wr_col1, wr_col2 = st.columns([3, 1])
    with wr_col1:
        selected_week_label = st.selectbox(
            "選擇比賽周次", week_labels, index=len(week_labels) - 1,
            key="wr_week",
        )
    week_idx = week_labels.index(selected_week_label)
    date_from, date_to = weeks[week_idx]

    with wr_col2:
        wr_gender = st.selectbox(
            "組別", ["全部", "男子組", "女子組"], key="wr_gender"
        )
    wr_gender_code = {"男子組": "M", "女子組": "F"}.get(wr_gender)

    # ── 撈取資料 ──────────────────────────────────────────────
    weekly_data = gather_weekly_data(date_from, date_to, season, wr_gender_code)
```

（`_attach_set_scores` 的呼叫與其內部查詢維持不變——`matches` 表的比對是以 `match_date` + 隊名 frozenset 定位，同一天同兩隊在單一賽季內不會重複，暫不加 season 過濾，避免過度工程化；已在下方風險清單註明。）

- [ ] **Step 15: `main.py` —— sidebar 加賽季下拉並傳入 `ctx`**

第 1-6 行（import 區塊）新增：

```python
from src.utils.constants import SEASON
```

第 80-101 行改為（在「選擇組別」之前插入賽季下拉）：

```python
st.sidebar.title("TVL 進階數據儀表板")
st.sidebar.markdown("---")

seasons_df = load_data("SELECT DISTINCT season FROM players ORDER BY season DESC", {})
season_options = seasons_df["season"].tolist() if not seasons_df.empty else [SEASON]
selected_season = st.sidebar.selectbox("選擇賽季", season_options, index=0)

gender = st.sidebar.selectbox("選擇組別", ["男子組", "女子組"])
gender_code = "M" if gender == "男子組" else "F"

teams_df = load_data(
    "SELECT team_id, team_name FROM teams WHERE gender = :gender_code ORDER BY team_id",
    {"gender_code": gender_code},
)
if teams_df.empty:
    st.warning("該組別目前沒有球隊資料。")
    st.stop()

team_name = st.sidebar.selectbox("選擇球隊", teams_df["team_name"].tolist())
team_id = int(teams_df.loc[teams_df["team_name"] == team_name, "team_id"].iloc[0])

players_df = load_data(
    "SELECT player_id, jersey_number, name, position FROM players "
    "WHERE team_id = :team_id AND gender = :gender_code AND season = :season ORDER BY jersey_number",
    {"team_id": team_id, "gender_code": gender_code, "season": selected_season},
)
if players_df.empty:
    st.warning("該球隊目前沒有球員資料。")
    st.stop()
```

第 125-133 行（`ctx` dict）加上 `"season"`：

```python
ctx = {
    "player_id": player_id,
    "player_name": player_name,
    "player_position": player_position,
    "gender_code": gender_code,
    "gender": gender,
    "team_name": team_name,
    "team_id": team_id,
    "season": selected_season,
}
```

- [ ] **Step 16: 更新 Task 2/7 建立的 AppTest harness，補上 `ctx["season"]`**

`tests/test_streamlit_ui.py` 中 `test_box_score_empty_teams_returns_without_stopping_script`、`test_league_pr_empty_league_returns_without_stopping_script`、`test_player_deep_empty_data_returns_without_stopping_script`、`test_weekly_report_no_weeks_returns_without_stopping_script` 這 4 個 harness 的 ctx dict 需補上 `"season": "2025-26"`：

```python
def test_box_score_empty_teams_returns_without_stopping_script(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        import streamlit as st
        from src.app.tabs import box_score

        box_score.render({"season": "2025-26"})
        st.text("MARKER_AFTER_RENDER")

    _assert_returns_without_stopping_script(_harness)


def test_league_pr_empty_league_returns_without_stopping_script(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        import streamlit as st
        from src.app.tabs import league_pr

        ctx = {
            "player_id": 1, "player_name": "測試球員", "player_position": "OH",
            "gender_code": "M", "gender": "男子組", "team_name": "測試隊",
            "season": "2025-26",
        }
        league_pr.render(ctx)
        st.text("MARKER_AFTER_RENDER")

    _assert_returns_without_stopping_script(_harness)


def test_player_deep_empty_data_returns_without_stopping_script(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        import streamlit as st
        from src.app.tabs import player_deep

        ctx = {
            "player_id": 1, "player_name": "測試球員", "player_position": "OH",
            "gender_code": "M", "gender": "男子組", "season": "2025-26",
        }
        player_deep.render(ctx)
        st.text("MARKER_AFTER_RENDER")

    _assert_returns_without_stopping_script(_harness)


def test_weekly_report_no_weeks_returns_without_stopping_script(sqlite_engine):
    def _harness():
        import sys
        sys.path.insert(0, ".")
        import streamlit as st
        from src.app.tabs import weekly_report_tab

        weekly_report_tab.render({"season": "2025-26"})
        st.text("MARKER_AFTER_RENDER")

    _assert_returns_without_stopping_script(_harness)
```

（`test_match_trend_empty_data_returns_without_stopping_script` 不需修改：`match_trend.py` 的查詢只以 `player_id` 過濾，`player_id` 本身已隱含唯一賽季，不需要 `ctx["season"]`。）

`tests/test_weekly_report_tab_mlis.py` 的兩個 harness 呼叫 `weekly_report_tab.render({})` 改為 `weekly_report_tab.render({"season": "2025-26"})`。

- [ ] **Step 17: 執行完整測試套件確認通過**

Run: `pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 18: Commit**

```bash
git add src/app/main.py src/app/helpers.py src/app/tabs/league_pr.py \
        src/app/tabs/player_deep.py src/app/tabs/box_score.py src/app/tabs/weekly_report_tab.py \
        src/app/tabs/prediction.py src/etl/weekly_report.py tests/test_weekly_report.py \
        tests/test_streamlit_ui.py tests/test_weekly_report_tab_mlis.py \
        tests/test_prediction_slider_config.py tests/test_season_filtering.py
git commit -m "feat: sidebar 新增賽季選擇器，所有查詢加 season 過濾，修正跨季 PR 頁重複列問題"
```

---

### Task 10: 整合驗證 + README 更新

收尾：跑一次完整測試套件確認計畫一既有的 48 個測試 + 本計畫新增測試全數通過，並更新 README 反映新的 7 分頁結構、MLIS 環境變數與系統設定說明。

**Files:**
- Modify: `README.md`

**Interfaces:**
- 無新介面（本 Task 僅跑既有測試套件與更新文件，不新增/變更任何函式簽名）。

- [ ] **Step 1: 跑完整測試套件**

Run:
```bash
cd /mnt/d/HPE-PCAI/TVL-Analysis
source .venv/bin/activate
pytest tests/ -v
```
Expected: 全部 PASS（計畫一 48 個測試 + 本計畫新增測試，含 `test_helpers_db.py`、`test_streamlit_ui.py`、`test_weekly_report_tab_dotenv.py`、`test_prediction_slider_config.py`、`test_schema.py`、`test_settings_store.py`、`test_llm_client.py`、`test_weekly_report_tab_mlis.py`、`test_settings_tab.py`、`test_season_filtering.py`、`test_weekly_report.py`）

- [ ] **Step 2: 更新 README —— 分頁表格與功能特色**

第 12 行（AI 戰報功能特色）改為：

```markdown
- **AI 戰報**：透過 PCAI MLIS（OpenAI 相容 API）自動產生每周結構化中文戰報
```

第 14-25 行（儀表板分頁表格）改為：

```markdown
## 儀表板分頁
> 🔗 **Live Demo**: [點此查看互動儀表板](https://tvl-analysis-jggmoeky3gnjrdbcc4kzrc.streamlit.app/)

| 分頁 | 功能 |
|------|------|
| 球員個人深度 | KPI 卡片、雷達圖、逐場趨勢（依位置自動調整） |
| 聯盟 PR 值與分佈 | 百分位排名、象限散佈圖（16 種指標可選） |
| 逐場趨勢 | 熱力資料表、對戰對手績效分佈 |
| 單場 Box Score | 雙方並列 Box Score、局比分、Top-10 排行 |
| 賽果預測 | ML 滑桿模擬器、SHAP 戰術診斷圖 |
| 每周戰報 | 視覺化比賽卡片、MLIS AI 自動撰寫戰報 |
| 系統設定 | 設定 MLIS endpoint/model/API key，測試連線 |

sidebar 最上層可選擇賽季（預設最新），下方所有分頁的資料皆依所選賽季過濾。
```

- [ ] **Step 3: 更新 README —— 專案結構區塊**

第 30-51 行的 `tabs/` 清單與 `app/` 區塊加上新檔案：

```markdown
├── src/
│   ├── app/
│   │   ├── main.py            # Streamlit 入口（路由 + sidebar，含賽季選擇器）
│   │   ├── helpers.py         # 共用函式（DB 查詢、指標計算、外部 API）
│   │   ├── llm_client.py      # MLIS（OpenAI 相容 API）呼叫層
│   │   ├── settings_store.py  # app_settings key-value 表存取
│   │   └── tabs/               # 七個分頁模組
│   │       ├── player_deep.py
│   │       ├── league_pr.py
│   │       ├── match_trend.py
│   │       ├── box_score.py
│   │       ├── prediction.py
│   │       ├── weekly_report_tab.py
│   │       └── settings_tab.py
```

- [ ] **Step 4: 更新 README —— 環境變數區塊**

第 74-75 行後新增（MLIS 相關環境變數）：

```markdown
- `DATABASE_URL`：連線目標，未設定時 fallback 至本地 `data/db/tvl_database.db`（SQLite）；指向 PostgreSQL 時使用 `postgresql+psycopg://...` 格式。
- `SEASON`：目前賽季字串（如 `2025-26`），未設定時預設 `2025-26`，ETL 寫入資料時以此標記賽季。
- `MLIS_BASE_URL` / `MLIS_API_KEY` / `MLIS_MODEL`：PCAI MLIS 的 OpenAI 相容 endpoint 設定，供「每周戰報」分頁產生 AI 戰報。可在「系統設定」分頁的 UI 設定（存於 Postgres/SQLite 的 `app_settings` 表，優先於環境變數），或直接設定環境變數；兩者皆未設定時戰報頁會顯示引導訊息。
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: 更新 README 反映七分頁結構、MLIS 環境變數與賽季選擇器"
```

---

## 完成後的檔案清單

```
src/app/main.py                          [改動：具名參數、st.stop 保留（頂層合理）、cache_resource、賽季選擇器、7 個 tab]
src/app/helpers.py                       [改動：load_data/get_league_aggregated_stats 具名參數 + ttl + season、import 排序]
src/app/llm_client.py                    [新增]
src/app/settings_store.py                [新增]
src/app/tabs/box_score.py                [改動：具名參數、return、season 過濾]
src/app/tabs/league_pr.py                [改動：return、season 過濾]
src/app/tabs/match_trend.py              [改動：具名參數、return]
src/app/tabs/player_deep.py              [改動：具名參數、return、season 過濾]
src/app/tabs/prediction.py               [改動：feature_cols 鍵名修正（Task 3）+ _get_data_ranges 加 season 參數（Task 9）]
src/app/tabs/weekly_report_tab.py        [改動：return、load_dotenv 選用、改接 MLIS、season 過濾]
src/app/tabs/settings_tab.py             [新增]
src/etl/weekly_report.py                 [改動：get_match_weeks/gather_weekly_data 加 season 參數]
sql/schema.sql                           [改動：新增 app_settings 表]
sql/schema_postgres.sql                  [改動：新增 app_settings 表]
requirements.txt                         [改動：移除 google-genai，新增 openai==2.53.0]
requirements-dev.txt                     [改動：新增 httpx==0.28.1]
README.md                                [改動：分頁表格、專案結構、環境變數]
tests/test_helpers_db.py                 [改動]
tests/test_streamlit_ui.py               [新增]
tests/test_weekly_report_tab_dotenv.py   [新增]
tests/test_prediction_slider_config.py   [新增]
tests/test_schema.py                     [改動]
tests/test_settings_store.py             [新增]
tests/test_llm_client.py                 [新增]
tests/test_weekly_report_tab_mlis.py     [新增]
tests/test_settings_tab.py               [新增]
tests/test_season_filtering.py           [新增]
tests/test_weekly_report.py              [改動]
```

## 已知風險（發現但不在本計畫處理範圍）

1. **`weekly_report_tab.py` 的 `_attach_set_scores` 未加 season 過濾**：`matches` 表查詢僅以 `match_date IN (...)` 過濾，未加 `season = :season`。理論上同一天同兩隊在單一賽季內不會重複比賽，跨季時同一天同對戰組合的機率也極低，暫不視為需要立即修正的問題，但若未來排程出現「跨季重複賽程」的邊界情況，需要補上此過濾。
2. **`box_score.py` 的組別/球隊選擇器與 sidebar 的賽季選擇器彼此獨立**：`box_score.py` 有自己的「組別」「球隊」下拉（`bs_gender`/`bs_team_name`），與 sidebar 的 `gender`/`team_name` 是兩套獨立狀態；本計畫只讓它讀取 sidebar 的 `season`，未統一這兩套選擇器。這是既有設計（計畫一之前就如此），不在本計畫範圍內一併重構，但使用者體驗上容易誤解「怎麼跟 sidebar 選的不一樣」。
3. **MLIS 真實 endpoint 未經真實伺服器驗證**：本機無法連線 PCAI 叢集，`llm_client.py` 的正確性僅透過 `httpx.MockTransport` 驗證 HTTP 呼叫層的邏輯（重試次數、錯誤處理、設定讀取順序），真實 endpoint 的 base_url 格式、認證方式、model 名稱等細節需在 PCAI 上實際部署 MLIS 後驗證，可能需要微調 `llm_client.py`（例如若 MLIS 的認證方式不是標準 Bearer token）。
4. **`settings_tab.py` 的 API key 欄位「留空表示不變更既有金鑰」的 UX 隱含假設**：使用者若誤以為留空會清空金鑰，可能造成困惑；目前僅以欄位說明文字（`"API Key（留空表示不變更既有金鑰）"`）提示，未做更明確的視覺區分（例如額外的「清除金鑰」按鈕）。
5. **`AppTest.from_function` 的 harness 是把函式原始碼抽出重新執行，非真正的 closure**：測試撰寫時容易誤用外層變數（例如誤把 pytest fixture 回傳值直接寫進 harness），如此撰寫會在執行期才報錯（`NameError`），而非在撰寫時被型別檢查抓到。本計畫的 harness 皆已避免此陷阱（一律用字面值或透過 `sqlite_engine` fixture 間接建立好的 DB 狀態），但後續維護者新增測試時需留意此限制。
6. **spec §4d 原文「8 處 `st.stop()`」與實際核對的 9 處不一致**：已在「專案現況」段落記錄，本計畫以實際核對的 9 處（含精確行號）為準，此為 spec 撰寫時的算術筆誤，不影響功能正確性。

## AppTest 可行性結論

`streamlit.testing.v1.AppTest`（streamlit 1.61 內建）在本專案完全可行：已實測 `AppTest.from_file()` 可完整跑通含 sidebar 三層連動與 6 個既有 tab 的 `main.py`（無例外，widget 可讀取與互動），`AppTest.from_function()` 可對單一 tab 的 `render(ctx)` 做隔離測試並正確捕捉 `st.stop()`/`return` 差異與 `st.form`/`st.button` 互動（含 `FormSubmitter:<form_key>-<label>` 的按鈕 key 慣例），故本計畫全面採用 AppTest 驗證 UI 層行為，未退回「僅測試抽出的純函式」的備案。
