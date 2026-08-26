# TVL Phase 1 基礎清理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 清掉 TVL-Analysis repo 的地雷與技術債（刪除每周戰報功能、消除 7 處 import fallback、requirements.txt 釘版本、拆掉 db_loader 的 DROP TABLE 地雷），讓 repo 進入可安全承接 Phase 2 schema 重建模的乾淨狀態。
**Architecture:** ETL 爬蟲（`src/etl/`）→ SQLite（`data/db/tvl_database.db`，commit 進 git）→ Streamlit 儀表板（`src/app/`）。本 phase 不改資料模型，只做刪除、收斂 import、釘版本、修 DDL 執行方式四類機械性但高風險的清理。
**Tech Stack:** Python 3、Streamlit、SQLite3、BeautifulSoup4、pandas、pytest（新增，測試骨架）。
**Spec:** roadmap 記憶檔 `tvl-optimization-roadmap` + 本 session 決策：Phase 1 涵蓋 (a) 刪除每周戰報功能 (b) 消除 7 處 import fallback (c) requirements.txt 釘版本 (d) 拆 db_loader 的 DROP TABLE 地雷、DDL 收斂到 `sql/schema.sql`。

## Global Constraints

- **行尾一律 LF**：本 repo 在 WSL 的 `/mnt/d` 上，Windows 工具容易把檔案轉成 CRLF。每個 task 收尾前跑 `git diff --stat -w` 確認沒有純行尾雜訊。
- **繁體中文**：UI 文案、程式註解（如有必要才加）、commit message 一律繁體中文；程式碼識別字（變數/函式名）維持英文。
- **DB 連線一律用 `src.utils.db_config.get_connection()`**（已開 `PRAGMA foreign_keys = ON`），不要在任何檔案裡裸用 `sqlite3.connect()`。
- **DDL 唯一來源是 `sql/schema.sql`**：任何 `.py` 檔都不得自己重寫 `CREATE TABLE`；需要建表時一律 `executescript()` 讀 `sql/schema.sql`，或用該表既有的 `CREATE TABLE IF NOT EXISTS` 陳述式。
- **requirements.txt 只放 Streamlit Cloud 執行期依賴**：測試依賴（pytest）放 `requirements-dev.txt`；notebook 專用套件（optuna）不進任何一份 requirements。
- **資料品質原則：只標記不插補**：任何清洗/驗證程式碼遇到異常值只能記警告，不能自動修正或捏造數值。
- **`.db` 與 `.pkl` 不得進 `.gitignore`**：Streamlit Cloud 靠 repo 內的 `data/db/tvl_database.db` 與 `src/models/match_predictor.pkl` 運作，任何 task 都不得把它們排除在版控外。
- **commit 前需徵求使用者同意**：本專案偏好是「commit 前先問過使用者」。每個 task 走到 commit 步驟時，執行者要停下來把預計的 commit message 貼給使用者確認，取得同意後才執行 `git commit`。在使用者尚未回覆前不要往下一個 task 推進 commit。
- **小步驟、每步可驗證**：每個 step 完成後立刻跑對應的驗證命令，不要累積到 task 結尾一次驗證。
- **不過度工程化**：函式小而專注、early return 優先，不加不必要的註解與 docstring（既有檔案已有的 docstring 風格可維持一致，但新增程式碼不用刻意補模板化註解）。

## 前置說明（已於規劃 session 實查，供所有 task 參照）

- 本 repo 目前無任何測試、無 pytest/lint/typecheck 設定（`find . -name conftest.py` 等指令查無結果，`python3 -m pytest --version` 回傳 `No module named pytest`）。Task 0 會建立最小骨架。
- `sql/schema.sql` 目前開頭是：
  ```sql
  DROP TABLE IF EXISTS player_match_stats;
  DROP TABLE IF EXISTS players;
  DROP TABLE IF EXISTS teams;
  ```
  `src/etl/db_loader.py:37-41` 的 `init_db()` 用 `conn.executescript(schema_sql)` 執行整份 `schema.sql`，因此每次跑 `python -m src.etl.db_loader` 都會把 `player_match_stats`（目前 3,807 筆逐場統計）連同 `players`/`teams` 一起清空重建。這是 Task 4 要拆的地雷。
- `src/etl/stats_crawler.py:78-82` 的 `init_stats_table()` 自己內嵌一份 `DROP TABLE IF EXISTS player_match_stats` + `CREATE TABLE`，與 `schema.sql` 重複定義；`src/etl/match_crawler.py:63-97` 的 `init_matches_table()` 也自己內嵌一份 `CREATE TABLE IF NOT EXISTS matches`，同樣與 `schema.sql:51-77` 重複。Task 4 一併收斂。
- 7 處 import fallback（皆為 `try: from src... except ModuleNotFoundError: ...` 形態，且多數 fallback 分支本身不完整、真的觸發會 `NameError`）：
  - `src/etl/db_loader.py:11-29`
  - `src/etl/crawler.py:13-24`（連帶 `src/etl/crawler.py:151-161` 的 `if TEAM_NAME_SHORT is None: ...` 死區塊要一併刪）
  - `src/etl/cleaner.py:12-20`
  - `src/etl/match_crawler.py:15-38`
  - `src/etl/stats_crawler.py:16-46`
  - `src/etl/weekly_report.py:13-18`（隨 Task 1 整檔刪除，不必單獨處理）
  - `src/app/helpers.py:67-78`
- `src/app/main.py:10-13` 的 `sys.path.insert` 是 Streamlit Cloud 部署需要的機制，**不是**要消除的 fallback，Task 2 不要動它。

---

## Task 0：建立最小測試骨架

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Produces：`tests/conftest.py` 提供 pytest fixture `tmp_db_path(tmp_path)`，回傳一個乾淨的 `Path`（尚未建檔），供後續 task 的測試建立隔離的 SQLite 檔案，不去動 `data/db/tvl_database.db`。
  ```python
  @pytest.fixture
  def tmp_db_path(tmp_path) -> Path:
      return tmp_path / "test.db"
  ```
- Consumes：無（本 task 是骨架起點）。

**步驟：**

- [ ] **Step 1:** 建立 `requirements-dev.txt`（純測試依賴，不進 `requirements.txt`）：
   ```
   pytest
   ```
   > 版本先不釘，待 Task 3 統一以 `pip freeze` 的實際解析結果釘版本（見 Task 3 步驟 2）。

- [ ] **Step 2:** 建立 `tests/__init__.py`（空檔，讓 `tests` 成為 package，避免 pytest import 路徑問題）：
   ```python
   ```
   （檔案內容留空即可）

- [ ] **Step 3:** 建立 `tests/conftest.py`：
   ```python
   from pathlib import Path

   import pytest


   @pytest.fixture
   def tmp_db_path(tmp_path) -> Path:
       """回傳隔離的 SQLite 檔案路徑，測試用，不觸碰正式 DB。"""
       return tmp_path / "test.db"
   ```

- [ ] **Step 4:** 建立 `tests/test_smoke.py`：
   ```python
   import sqlite3

   from src.utils.db_config import get_connection


   def test_get_connection_enables_foreign_keys():
       conn = get_connection()
       try:
           row = conn.execute("PRAGMA foreign_keys").fetchone()
           assert row[0] == 1
       finally:
           conn.close()


   def test_schema_sql_is_valid_sqlite():
       from pathlib import Path

       schema_path = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
       conn = sqlite3.connect(":memory:")
       conn.executescript(schema_path.read_text(encoding="utf-8"))
           # 能無錯執行到底即代表語法正確
       conn.close()
   ```

- [ ] **Step 5:** 在 repo 根目錄安裝並執行測試，確認骨架可跑（尚未安裝 pytest 時先裝）：
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt
   python -m pytest tests/ -v
   ```
   **預期輸出**：2 個測試皆 `PASSED`（`test_get_connection_enables_foreign_keys`、`test_schema_sql_is_valid_sqlite`）。若失敗，先確認是否在 repo 根目錄執行（import 走 `from src...` 需要根目錄在 `sys.path`，pytest 預設會把 rootdir 加進去，因為 `tests/__init__.py` 存在且 rootdir 有 `src/__init__.py`）。

   **未實測**：本規劃 session 未安裝套件實跑（環境為 externally-managed system Python，且未建 venv）。此步驟需執行者在自己的 venv 中實跑並確認上述輸出。

- [ ] **Step 6:** 確認行尾與 diff 乾淨：
   ```bash
   git status
   git diff --stat -w
   git add requirements-dev.txt tests/
   ```

- [ ] **Step 7:** **[STOP：等待使用者同意 commit]** 向使用者展示以下 commit message 草案，取得同意後再執行：
   ```
   test: 建立最小 pytest 測試骨架

   新增 requirements-dev.txt 與 tests/ 目錄，作為後續 Phase 1/2 各項清理與遷移改動的測試地基。
   ```
   同意後執行：
   ```bash
   git commit -m "$(cat <<'EOF'
   test: 建立最小 pytest 測試骨架

   新增 requirements-dev.txt 與 tests/ 目錄，作為後續 Phase 1/2 各項清理與遷移改動的測試地基。
   EOF
   )"
   git status
   ```
   **驗證命令**：`git log -1 --stat`
   **預期輸出**：最新一筆 commit 顯示新增 `requirements-dev.txt`、`tests/__init__.py`、`tests/conftest.py`、`tests/test_smoke.py` 四個檔案。

---

## Task 1：整個刪除「每周戰報」功能

**Files:**
- Delete: `src/etl/weekly_report.py`
- Delete: `src/app/tabs/weekly_report_tab.py`
- Modify: `src/app/main.py`（3 處：L74 import、L142-145 tabs 清單、L147-163 tab 解包與 with 區塊；tabs 宣告與解包同一行故合併計算）
- Modify: `requirements.txt`（移除 `google-genai`）
- Modify: `README.md`（6 處：L12、L24、L40、L46、L107、L127）
- Create: `tests/test_main_tabs.py`

**Interfaces:**
- Consumes：無新介面依賴（純刪除）。
- Produces：`src/app/main.py` 收斂為 5 個 tab（`tab1..tab5`），`ctx` dict 介面維持不變：
  ```python
  ctx = {
      "player_id": int, "player_name": str, "player_position": str | None,
      "gender_code": str, "gender": str, "team_name": str, "team_id": int,
  }
  ```
  後續 task／Phase 2 皆沿用此 `ctx` 介面，不得新增或改名既有 key。

**步驟：**

- [ ] **Step 1:** 先跑一次現況基準測試，確認目前是綠的：
   ```bash
   python -m pytest tests/ -v
   ```
   **預期輸出**：2 個既有測試 PASSED。

- [ ] **Step 2:** 刪除 `src/etl/weekly_report.py` 與 `src/app/tabs/weekly_report_tab.py`：
   ```bash
   git rm src/etl/weekly_report.py src/app/tabs/weekly_report_tab.py
   ```

- [ ] **Step 3:** 編輯 `src/app/main.py`，移除 tab6 相關的 3 處（tabs 宣告與解包同一行故合併計算）：

   3a. 第 74 行的 import，原文：
   ```python
   from src.app.tabs import player_deep, league_pr, match_trend, box_score, prediction, weekly_report_tab
   ```
   改為：
   ```python
   from src.app.tabs import player_deep, league_pr, match_trend, box_score, prediction
   ```

   3b. 第 142-145 行的 tabs 清單，原文：
   ```python
   tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
       "球員個人深度", "聯盟 PR 值與分佈", "逐場趨勢", "單場 Box Score", "賽果預測",
       "每周戰報",
   ])
   ```
   改為：
   ```python
   tab1, tab2, tab3, tab4, tab5 = st.tabs([
       "球員個人深度", "聯盟 PR 值與分佈", "逐場趨勢", "單場 Box Score", "賽果預測",
   ])
   ```

   3c. 第 162-163 行，原文：
   ```python
   with tab6:
       weekly_report_tab.render(ctx)
   ```
   整段刪除（連同前面第 161 行的空行一併確認不要留兩個空行，維持 `with tab5:` 區塊結束後檔案自然結尾）。

- [ ] **Step 4:** 驗證 `main.py` 語法正確且 import 不再指向已刪除模組：
   ```bash
   python -c "import ast; ast.parse(open('src/app/main.py', encoding='utf-8').read())"
   grep -n "weekly_report" src/app/main.py
   ```
   **預期輸出**：`ast.parse` 無輸出無錯誤；`grep` 無任何匹配（exit code 1）。

- [ ] **Step 5:** 編輯 `requirements.txt`，移除 `google-genai` 這一行（該套件僅供 weekly_report.py 呼叫 Gemini API 使用）。

- [ ] **Step 6:** 編輯 `README.md` 六處：

   6a. 第 12 行，原文：
   ```
   - **AI 戰報**：透過 Gemini API（免費）自動產生每周結構化中文戰報
   ```
   整行刪除。

   6b. 第 24 行，原文：
   ```
   | 每周戰報 | 視覺化比賽卡片、Gemini AI 自動撰寫戰報 |
   ```
   整行刪除。同時把第 10 行「六分頁」改為「五分頁」：
   ```
   - **互動式儀表板**：Streamlit + Plotly 打造的六分頁視覺化分析介面
   ```
   改為：
   ```
   - **互動式儀表板**：Streamlit + Plotly 打造的五分頁視覺化分析介面
   ```

   6c. 第 40 行（tabs 清單中的檔名），原文：
   ```
   │   │       ├── prediction.py
   │   │       └── weekly_report_tab.py
   ```
   改為：
   ```
   │   │       └── prediction.py
   ```

   6d. 第 46 行，原文：
   ```
   │   │   ├── db_loader.py       # 資料庫載入
   │   │   └── weekly_report.py   # 週報資料彙整
   ```
   改為：
   ```
   │   │   └── db_loader.py       # 資料庫載入
   ```

   6e. 第 105-107 行整段「AI 戰報（選用）」刪除，原文：
   ```
   ### AI 戰報（選用）

   在 `.env` 中設定 `GEMINI_API_KEY=...`（從 [Google AI Studio](https://aistudio.google.com/) 免費取得），即可在儀表板「每周戰報」分頁使用 Gemini 2.0 Flash 自動產生戰報。
   ```
   整段刪除（含前後多餘空行，確認刪除後上下段落間只留一個空行）。

   6f. 第 127 行，原文：
   ```
   - **AI 戰報**：Google Gemini 2.0 Flash（免費）
   ```
   整行刪除。

- [ ] **Step 7:** 驗證 README 與 requirements 不再殘留任何戰報/Gemini 字樣：
   ```bash
   grep -in "weekly_report\|gemini\|google-genai\|每周戰報\|每週戰報" README.md requirements.txt src/app/main.py
   ```
   **預期輸出**：無匹配（exit code 1）。

- [ ] **Step 8:** 新增 `tests/test_main_tabs.py`，驗證 `main.py` 原始碼中不再引用已刪除模組、且 tab 數量正確（用原始碼靜態檢查，不用真的跑 Streamlit runtime，因為 `main.py` 是 script-level 執行，不適合直接 import 測試）：
   ```python
   from pathlib import Path

   MAIN_PY = Path(__file__).resolve().parents[1] / "src" / "app" / "main.py"


   def test_weekly_report_tab_removed():
       source = MAIN_PY.read_text(encoding="utf-8")
       assert "weekly_report" not in source


   def test_five_tabs_declared():
       source = MAIN_PY.read_text(encoding="utf-8")
       assert "tab1, tab2, tab3, tab4, tab5 = st.tabs" in source
       assert "tab6" not in source


   def test_weekly_report_files_deleted():
       root = MAIN_PY.resolve().parents[3]
       assert not (root / "src" / "etl" / "weekly_report.py").exists()
       assert not (root / "src" / "app" / "tabs" / "weekly_report_tab.py").exists()
   ```

- [ ] **Step 9:** 跑測試：
   ```bash
   python -m pytest tests/ -v
   ```
   **預期輸出**：5 個測試（Task 0 的 2 個 + 本 task 的 3 個）全部 PASSED。

   **未實測**：本規劃 session 環境無 streamlit/pandas 等套件可實際 import 執行 pytest，執行者需在自己 venv 中實跑確認。

- [ ] **Step 10:** 確認行尾乾淨：
    ```bash
    git diff --stat -w
    ```
    **預期輸出**：無 CRLF/LF 雜訊行（若有大量 +/- 但內容相同的行，代表行尾被改動，需重新檢查編輯器設定）。

- [ ] **Step 11:** **[STOP：等待使用者同意 commit]** 展示 commit message 草案：
    ```bash
    git add -A -- src/app/main.py src/app/tabs/weekly_report_tab.py src/etl/weekly_report.py requirements.txt README.md tests/test_main_tabs.py
    git status
    ```
    ```
    feat: 整個刪除每周戰報功能（含 Gemini API 依賴）

    每周戰報 tab 已定案不做，改用 optuna/PCAI 之外的資源；移除
    weekly_report.py、weekly_report_tab.py、google-genai 依賴，
    main.py 收斂為五分頁，README 同步更新。
    ```
    同意後執行 commit，並跑 `git status` 確認乾淨。

---

## Task 2：消除 7 處 import fallback

**Files:**
- Modify: `src/etl/db_loader.py`（L1-29 → 改為直接絕對 import）
- Modify: `src/etl/crawler.py`（L1-26、L149-161）
- Modify: `src/etl/cleaner.py`（L1-22）
- Modify: `src/etl/match_crawler.py`（L1-40）
- Modify: `src/etl/stats_crawler.py`（L1-48）
- Modify: `src/app/helpers.py`（L1-80）
- Create: `tests/test_no_import_fallback.py`

**Interfaces:**
- Consumes：`src.utils.constants`（`POSITION_MAP`, `EXT_HEADERS`, `TEAM_NAME_SHORT`, `VALID_POSITIONS`, `VALID_GENDERS`, `EXT_BASE`, `EXT_CUP_ID`, `SEASON_YEAR_MAP`, `DEFAULT_YEAR`, `EXT_TEAM_MAP`, `OPP_SHORT_TO_TEAM`）、`src.utils.db_config`（`PROJECT_ROOT`, `DB_PATH`, `get_connection`）、`src.utils.logger`（`get_logger`）— 皆為既有介面，本 task 不改動其簽名，只改「怎麼 import」。
- Produces：每個檔案頂部只剩一組乾淨的 `from src...` import，不再有 `try/except ModuleNotFoundError`。

**前提**：此 task 假設所有爬蟲/etl 一律以 `python -m src.etl.xxx` 方式從 repo 根目錄執行（README 與 CLAUDE.md 已明文規定），因此絕對 import 必然可行，fallback 分支從未真正需要過。

**步驟：**

- [ ] **Step 1:** 編輯 `src/etl/db_loader.py`，把第 1-29 行：
   ```python
   import sqlite3
   import numpy as np
   import pandas as pd
   from pathlib import Path

   try:
       from src.etl.cleaner import load_raw, clean, quality_report
       from src.utils.db_config import PROJECT_ROOT, DB_PATH, get_connection
       from src.utils.logger import get_logger
   except ModuleNotFoundError:
       from cleaner import load_raw, clean, quality_report
       import logging
       logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
       get_logger = logging.getLogger
       PROJECT_ROOT = Path(__file__).resolve().parents[2]
       DB_PATH = PROJECT_ROOT / "data" / "db" / "tvl_database.db"

       def get_connection(foreign_keys=True):
           import sqlite3
           DB_PATH.parent.mkdir(parents=True, exist_ok=True)
           conn = sqlite3.connect(DB_PATH)
           if foreign_keys:
               conn.execute("PRAGMA foreign_keys = ON")
           return conn
   ```
   改為：
   ```python
   import sqlite3
   import numpy as np
   import pandas as pd
   from pathlib import Path

   from src.etl.cleaner import load_raw, clean, quality_report
   from src.utils.db_config import PROJECT_ROOT, DB_PATH, get_connection
   from src.utils.logger import get_logger
   ```

- [ ] **Step 2:** 編輯 `src/etl/crawler.py`：
   把第 13-24 行：
   ```python
   try:
       from src.utils.logger import get_logger
       from src.utils.constants import POSITION_MAP, EXT_HEADERS as HEADERS, TEAM_NAME_SHORT
   except ModuleNotFoundError:
       import logging
       logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
       get_logger = logging.getLogger
       POSITION_MAP = {"主攻手": "OH", "中間手": "MB", "副攻手": "OP", "舉球員": "S", "自由球員": "L"}
       HEADERS = {
           "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
       }
       TEAM_NAME_SHORT = None  # will be defined below as local fallback
   ```
   改為：
   ```python
   from src.utils.logger import get_logger
   from src.utils.constants import POSITION_MAP, EXT_HEADERS as HEADERS, TEAM_NAME_SHORT
   ```
   再把第 149-161 行（原本緊接在 `GENDER_MAP` 之後的死區塊）：
   ```python
   GENDER_MAP = {"team": "M", "wteam": "F"}

   # 官網全名 → 簡寫對應（fallback：constants.py import 失敗時使用）
   if TEAM_NAME_SHORT is None:
       TEAM_NAME_SHORT = {
           "臺北鯨華女子排球隊": "臺北鯨華",
           "新北中國人纖企業女子排球隊": "新北中纖",
           "台灣電力公司女子排球隊": "高雄台電",
           "義力營造女子排球隊": "義力營造",
           "台灣電力公司男子排球隊": "屏東台電",
           "美津濃男子排球隊": "雲林美津濃",
           "桃園臺產隼鷹排球隊": "桃園臺產",
       }
   ```
   改為：
   ```python
   GENDER_MAP = {"team": "M", "wteam": "F"}
   ```

- [ ] **Step 3:** 編輯 `src/etl/cleaner.py`，把第 12-20 行：
   ```python
   try:
       from src.utils.logger import get_logger
       from src.utils.constants import VALID_POSITIONS, VALID_GENDERS
   except ModuleNotFoundError:
       import logging
       logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
       get_logger = logging.getLogger
       VALID_POSITIONS = {"OH", "MB", "OP", "S", "L"}
       VALID_GENDERS = {"M", "F"}
   ```
   改為：
   ```python
   from src.utils.logger import get_logger
   from src.utils.constants import VALID_POSITIONS, VALID_GENDERS
   ```

- [ ] **Step 4:** 編輯 `src/etl/match_crawler.py`，把第 15-38 行：
   ```python
   try:
       from src.utils.db_config import DB_PATH, get_connection
       from src.utils.logger import get_logger
       from src.utils.constants import EXT_HEADERS as HEADERS
   except ModuleNotFoundError:
       import logging
       logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
       get_logger = logging.getLogger
       DB_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "tvl_database.db"

       def get_connection(foreign_keys=True):
           DB_PATH.parent.mkdir(parents=True, exist_ok=True)
           conn = sqlite3.connect(DB_PATH)
           if foreign_keys:
               conn.execute("PRAGMA foreign_keys = ON")
           return conn

       HEADERS = {
           "User-Agent": (
               "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "AppleWebKit/537.36 (KHTML, like Gecko) "
               "Chrome/120.0.0.0 Safari/537.36"
           )
       }
   ```
   改為：
   ```python
   from src.utils.db_config import DB_PATH, get_connection
   from src.utils.logger import get_logger
   from src.utils.constants import EXT_HEADERS as HEADERS
   ```

- [ ] **Step 5:** 編輯 `src/etl/stats_crawler.py`，把第 16-46 行：
   ```python
   try:
       from src.utils.db_config import DB_PATH, get_connection
       from src.utils.logger import get_logger
       from src.utils.constants import (
           EXT_BASE, EXT_CUP_ID as CUP_ID, EXT_HEADERS as HEADERS,
           SEASON_YEAR_MAP, DEFAULT_YEAR, EXT_TEAM_MAP,
       )
   except ModuleNotFoundError:
       import logging
       logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
       get_logger = logging.getLogger
       DB_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "tvl_database.db"

       def get_connection(foreign_keys=True):
           DB_PATH.parent.mkdir(parents=True, exist_ok=True)
           conn = sqlite3.connect(DB_PATH)
           if foreign_keys:
               conn.execute("PRAGMA foreign_keys = ON")
           return conn

       EXT_BASE = "http://114.35.229.141"
       CUP_ID = 21
       SEASON_YEAR_MAP = {11: 2025, 12: 2025}
       DEFAULT_YEAR = 2026
       HEADERS = {
           "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
       }
       EXT_TEAM_MAP = {
           1: (1, "M"), 2: (2, "M"), 3: (7, "M"), 4: (4, "M"), 5: (5, "M"),
           6: (4, "F"), 7: (3, "F"), 8: (5, "F"), 9: (7, "F"),
       }
   ```
   改為：
   ```python
   from src.utils.db_config import DB_PATH, get_connection
   from src.utils.logger import get_logger
   from src.utils.constants import (
       EXT_BASE, EXT_CUP_ID as CUP_ID, EXT_HEADERS as HEADERS,
       SEASON_YEAR_MAP, DEFAULT_YEAR, EXT_TEAM_MAP,
   )
   ```

- [ ] **Step 6:** 編輯 `src/app/helpers.py`，把第 67-78 行：
   ```python
   try:
       from src.utils.constants import (
           EXT_BASE, EXT_CUP_ID, EXT_HEADERS, SEASON_YEAR_MAP, OPP_SHORT_TO_TEAM,
       )
   except ModuleNotFoundError:
       EXT_BASE = "http://114.35.229.141"
       EXT_CUP_ID = 21
       EXT_HEADERS = {
           "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
       }
       SEASON_YEAR_MAP = {11: 2025, 12: 2025}
       OPP_SHORT_TO_TEAM = {}
   ```
   改為：
   ```python
   from src.utils.constants import (
       EXT_BASE, EXT_CUP_ID, EXT_HEADERS, SEASON_YEAR_MAP, OPP_SHORT_TO_TEAM,
   )
   ```
   同時把這段 import 移到檔案最上方的 import 區塊（與 `re`, `sqlite3`, `numpy` 等並列），不要留在 `compact_margin()` 函式定義之後 —— 純粹是風格整理，不影響行為。改後檔案開頭 import 區塊應為：
   ```python
   import re
   import sqlite3
   from pathlib import Path

   import numpy as np
   import pandas as pd
   import requests
   import streamlit as st
   from bs4 import BeautifulSoup

   from src.utils.constants import (
       EXT_BASE, EXT_CUP_ID, EXT_HEADERS, SEASON_YEAR_MAP, OPP_SHORT_TO_TEAM,
   )
   ```

- [ ] **Step 7:** 全域搜尋確認 7 處 fallback 已全部清除（`weekly_report.py` 已於 Task 1 隨檔刪除，故此處只檢查剩餘 6 個檔案）：
   ```bash
   grep -rn "except ModuleNotFoundError" src/
   ```
   **預期輸出**：無匹配（exit code 1）。

- [ ] **Step 8:** 靜態語法檢查全部改動檔案：
   ```bash
   for f in src/etl/db_loader.py src/etl/crawler.py src/etl/cleaner.py src/etl/match_crawler.py src/etl/stats_crawler.py src/app/helpers.py; do
     python3 -c "import ast; ast.parse(open('$f', encoding='utf-8').read())" && echo "OK: $f"
   done
   ```
   **預期輸出**：6 行 `OK: <檔名>`。

- [ ] **Step 9:** 新增 `tests/test_no_import_fallback.py`：
   ```python
   from pathlib import Path

   ETL_DIR = Path(__file__).resolve().parents[1] / "src" / "etl"
   APP_DIR = Path(__file__).resolve().parents[1] / "src" / "app"

   FILES_TO_CHECK = [
       ETL_DIR / "db_loader.py",
       ETL_DIR / "crawler.py",
       ETL_DIR / "cleaner.py",
       ETL_DIR / "match_crawler.py",
       ETL_DIR / "stats_crawler.py",
       APP_DIR / "helpers.py",
   ]


   def test_no_module_not_found_fallback():
       for path in FILES_TO_CHECK:
           source = path.read_text(encoding="utf-8")
           assert "ModuleNotFoundError" not in source, f"{path} 仍有 fallback"


   def test_crawler_has_no_dead_fallback_block():
       source = (ETL_DIR / "crawler.py").read_text(encoding="utf-8")
       assert "if TEAM_NAME_SHORT is None" not in source
   ```

- [ ] **Step 10:** 跑完整測試：
    ```bash
    python -m pytest tests/ -v
    ```
    **預期輸出**：全部 PASSED（Task 0: 2 + Task 1: 3 + 本 task: 2 = 7 個測試）。

    **未實測**：需執行者在有完整依賴的 venv 中實跑確認。

- [ ] **Step 11:** 用實際爬蟲跑一次最小驗證（確認絕對 import 在真實執行路徑下沒問題；只測 import 與函式可呼叫，不必真的發網路請求）：
    ```bash
    python -c "from src.etl import db_loader, crawler, cleaner, match_crawler, stats_crawler; from src.app import helpers; print('all modules import OK')"
    ```
    **預期輸出**：`all modules import OK`

    **未實測**：同上，需在有依賴的環境跑。

- [ ] **Step 12:** 確認行尾乾淨：`git diff --stat -w`

- [ ] **Step 13:** **[STOP：等待使用者同意 commit]**：
    ```
    refactor: 消除 6 處 import fallback，統一走絕對 import

    專案一律以 `python -m src.etl.xxx` 從根目錄執行，try/except
    ModuleNotFoundError 的 fallback 分支從未真正被觸發，且多數
    fallback 本身不完整（觸發即 NameError）。統一改為
    `from src...` 絕對 import，並清掉 crawler.py 對應的死區塊。
    ```

---

## Task 3：requirements.txt 釘版本 + 建立 requirements-dev.txt 內容

**Files:**
- Modify: `requirements.txt`
- Modify: `requirements-dev.txt`（Task 0 已建立空骨架，本 task 補上釘版本內容）
- Create: `tests/test_requirements_pinned.py`

**Interfaces:**
- 無程式介面變動，本 task 純粹是依賴宣告檔。

**前提查證（本規劃 session 已實測）：**
- 本規劃環境的系統 Python 為 externally-managed（`pip install` 直接失敗，需要 venv），且未安裝任何專案依賴（`pandas`/`numpy`/`streamlit`/`xgboost` 等在系統 Python 中皆未安裝，`pip show` 對它們都查無結果）。因此**無法在本規劃 session 中直接產生「目前實際安裝版本」**，必須由執行者在自己的 venv 中安裝、`pip freeze` 取得真實可重現版本。
- 本規劃 session 有對外網路（`curl https://pypi.org` 成功），並實際查詢過 PyPI 各套件當前最新版（2026-08-26 查證，僅供起始參考，不是最終要釘的版本 —— 最終版本以下面步驟 2 的 `pip freeze` 實跑結果為準）：
  ```
  beautifulsoup4==4.15.0  joblib==1.5.3      matplotlib==3.11.1
  numpy==2.5.2            pandas==3.0.5      plotly==7.0.0
  python-dotenv==1.2.3    requests==2.34.2   scikit-learn==1.9.0
  shap==0.52.0            streamlit==1.62.0  xgboost==3.4.1
  optuna==4.9.0（不進 requirements，僅 notebook 用）
  pytest==9.1.1
  ```
- `packages.txt`（Streamlit Cloud apt 清單）目前只有 `fonts-noto-cjk`，本 task 不動它。

**步驟：**

- [ ] **Step 1:** 建立乾淨 venv，安裝目前未釘版本的 `requirements.txt`，取得可重現的解析結果：
   ```bash
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   pip freeze > /tmp/tvl_freeze_before.txt
   cat /tmp/tvl_freeze_before.txt
   ```
   **預期輸出**：一份完整的 `pip freeze` 清單，包含 `requirements.txt` 中列出的 8 個直接依賴（`google-genai` 已於 Task 1 移除，故只剩 `beautifulsoup4, joblib, matplotlib, numpy, pandas, plotly, python-dotenv, requests, scikit-learn, shap, streamlit, xgboost`）加上它們的遞移依賴。

   **未實測**：需執行者實跑並貼上真實輸出。若此步驟因網路或套件衝突失敗，記錄失敗原因，不要用記憶中的版本號硬填。

- [ ] **Step 2:** 從 `pip freeze` 結果中，只挑出 `requirements.txt` 原本就列出的 12 個直接依賴（不含遞移依賴），把每個套件改成 `==` 釘死的形式。範例（**實際版本號以步驟 1 的真實 freeze 輸出為準，取代下面 `<FROM_FREEZE>` 占位**，執行者不得憑記憶或本文件的參考版本填入，必須抄自己剛跑出來的 `pip freeze`）：
   ```
   beautifulsoup4==<FROM_FREEZE>
   joblib==<FROM_FREEZE>
   matplotlib==<FROM_FREEZE>
   numpy==<FROM_FREEZE>
   pandas==<FROM_FREEZE>
   plotly==<FROM_FREEZE>
   python-dotenv==<FROM_FREEZE>
   requests==<FROM_FREEZE>
   scikit-learn==<FROM_FREEZE>
   shap==<FROM_FREEZE>
   streamlit>=1.45.0,<FROM_FREEZE_major_pin>
   xgboost==<FROM_FREEZE>
   ```
   > 說明：`streamlit` 原本是 `>=1.45.0`（下限版本，非精確釘版），保留下限語意但補上實測到的版本作為精確釘版 `streamlit==<FROM_FREEZE>`（改為精確釘版，因為「requirements.txt 釘版本」的決策就是要讓 Streamlit Cloud 每次重建都拿到同一組版本，不要讓 `>=` 造成未來重建時版本漂移）。

- [ ] **Step 3:** 用同一個 venv 驗證釘死版本後仍可正常安裝（重新建一個乾淨 venv 驗證，避免受第一次安裝殘留影響）：
   ```bash
   deactivate
   rm -rf venv2
   python3 -m venv venv2
   source venv2/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   echo "exit code: $?"
   ```
   **預期輸出**：`exit code: 0`，且無版本衝突錯誤訊息。

   **未實測**：需執行者實跑。若安裝失敗（版本衝突），退回步驟 2 調整衝突套件版本，並在 commit message 中記錄調整原因。

- [ ] **Step 4:** 驗證 Streamlit app 至少能被語法解析與 import 到 `st.set_page_config` 之前的程度（不需要真的啟動 server，僅確認依賴打包完整）：
   ```bash
   python -c "
   import matplotlib, numpy, pandas, plotly, requests, sklearn, xgboost, shap, streamlit
   from bs4 import BeautifulSoup
   import joblib
   from dotenv import load_dotenv
   print('all pinned deps import OK')
   "
   ```
   **預期輸出**：`all pinned deps import OK`

   **未實測**：需執行者實跑。

- [ ] **Step 5:** 同樣方式為 `requirements-dev.txt` 釘版本（在 venv2 中額外安裝 pytest 並 freeze）：
   ```bash
   pip install pytest
   pip freeze | grep -i "^pytest=="
   ```
   把結果填入 `requirements-dev.txt`：
   ```
   pytest==<FROM_FREEZE>
   ```

- [ ] **Step 6:** 新增 `tests/test_requirements_pinned.py`（靜態檢查，不依賴網路）：
   ```python
   import re
   from pathlib import Path

   ROOT = Path(__file__).resolve().parents[1]
   PIN_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+==[A-Za-z0-9_.]+")
   RANGE_PIN_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+(>=|<=|~=)[A-Za-z0-9_.]+.*==[A-Za-z0-9_.]+")


   def _lines(path: Path) -> list[str]:
       return [
           line.strip()
           for line in path.read_text(encoding="utf-8").splitlines()
           if line.strip() and not line.strip().startswith("#")
       ]


   def test_requirements_txt_all_pinned():
       for line in _lines(ROOT / "requirements.txt"):
           assert PIN_PATTERN.match(line) or RANGE_PIN_PATTERN.match(line), \
               f"未釘版本：{line}"


   def test_requirements_txt_no_genai():
       source = (ROOT / "requirements.txt").read_text(encoding="utf-8")
       assert "genai" not in source.lower()


   def test_requirements_dev_pinned():
       for line in _lines(ROOT / "requirements-dev.txt"):
           assert PIN_PATTERN.match(line), f"未釘版本：{line}"


   def test_optuna_not_in_any_requirements():
       for fname in ["requirements.txt", "requirements-dev.txt"]:
           source = (ROOT / fname).read_text(encoding="utf-8").lower()
           assert "optuna" not in source
   ```

- [ ] **Step 7:** 跑測試：
   ```bash
   python -m pytest tests/test_requirements_pinned.py -v
   ```
   **預期輸出**：4 個測試 PASSED。

   **未實測**：需執行者實跑（此測試不依賴外部套件安裝，僅讀檔案文字，理論上在任何 Python 3 環境都可跑，包含本規劃 session 若有 pytest 也可驗證格式邏輯本身）。

- [ ] **Step 8:** 清理暫用的 venv/venv2（不要 commit 進 git，`.gitignore` 已排除 `venv/`，`venv2/` 需手動刪除避免殘留）：
   ```bash
   deactivate
   rm -rf venv venv2
   ```

- [ ] **Step 9:** 確認行尾乾淨：`git diff --stat -w`

- [ ] **Step 10:** **[STOP：等待使用者同意 commit]**：
    ```
    build: requirements.txt 釘死版本，新增 requirements-dev.txt

    避免 Streamlit Cloud 重建時因套件版本漂移導致行為不一致；
    測試依賴（pytest）獨立放 requirements-dev.txt，不進執行期依賴。
    ```

---

## Task 4：拆 db_loader 的 DROP TABLE 地雷，DDL 收斂到 schema.sql

**Files:**
- Modify: `sql/schema.sql`（移除頂部 3 個 `DROP TABLE`，`teams`/`players`/`player_match_stats` 改為 `CREATE TABLE IF NOT EXISTS`）
- Modify: `src/etl/db_loader.py`（`insert_teams`/`insert_players` 改為 upsert，不再假設空表）
- Modify: `src/etl/stats_crawler.py`（`init_stats_table()` 移除內嵌 DDL，改讀 `schema.sql`）
- Modify: `src/etl/match_crawler.py`（`init_matches_table()` 移除內嵌 DDL，改讀 `schema.sql`）
- Create: `tests/test_db_loader_idempotent.py`

**Interfaces:**
- Produces：
  ```python
  def init_db(conn: sqlite3.Connection) -> None: ...  # 冪等，執行任意次都不清空既有資料
  def upsert_teams(conn: sqlite3.Connection, df: pd.DataFrame) -> None: ...  # 取代 insert_teams
  def upsert_players(conn: sqlite3.Connection, df: pd.DataFrame) -> None: ...  # 取代 insert_players
  ```
- Consumes（沿用既有介面，不改動）：`src.utils.db_config.get_connection`、`src.etl.cleaner.load_raw/clean/quality_report`。

**設計說明（為什麼不能單純把 DROP 改成 CREATE IF NOT EXISTS 就收工）：**
`players.player_id` 是 `AUTOINCREMENT`，`player_match_stats.player_id` 又 FK 參考它。如果只是把 DROP 拿掉，`insert_players()` 原本的邏輯（每次都無條件 `INSERT` 全部 CSV 列）會在第二次執行時產生**重複的球員列**（新的 `player_id`），而不是報錯，因為 `players` 表本身沒有唯一約束擋重複。所以本 task 除了拿掉 DROP，還要把 `insert_players` 改成用自然鍵（`team_id, gender, jersey_number, name`）比對既有資料，存在就 `UPDATE`（保留原 `player_id`，讓 `player_match_stats` 的 FK 不受影響），不存在才 `INSERT`。`teams` 表因為已有複合主鍵 `(team_id, gender)`，改用 `INSERT OR IGNORE` 即可冪等。

**步驟：**

- [ ] **Step 1:** 編輯 `sql/schema.sql`，刪除第 4-7 行：
   ```sql
   -- 依 FK 順序先刪除子表，再刪除父表
   DROP TABLE IF EXISTS player_match_stats;
   DROP TABLE IF EXISTS players;
   DROP TABLE IF EXISTS teams;
   ```
   把第 9 行 `CREATE TABLE teams (` 改為 `CREATE TABLE IF NOT EXISTS teams (`。
   把第 16 行 `CREATE TABLE players (` 改為 `CREATE TABLE IF NOT EXISTS players (`。
   把第 29 行 `CREATE TABLE player_match_stats (` 改為 `CREATE TABLE IF NOT EXISTS player_match_stats (`。
   （`matches` 表第 51 行已經是 `CREATE TABLE IF NOT EXISTS`，不用改。）

   改完後 `sql/schema.sql` 開頭應為：
   ```sql
   -- TVL 資料庫 Schema（可重複執行，冪等：僅 CREATE TABLE IF NOT EXISTS，不清空既有資料）
   -- 注意：男女組的 team_id 可能重複，因此 teams 使用複合主鍵 (team_id, gender)

   CREATE TABLE IF NOT EXISTS teams (
       team_id   INTEGER NOT NULL,
       team_name TEXT    NOT NULL,
       gender    TEXT    NOT NULL CHECK (gender IN ('M', 'F')),
       PRIMARY KEY (team_id, gender)
   );
   ```

- [ ] **Step 2:** 驗證 schema.sql 仍是合法 SQLite 語法（可重複執行兩次都不報錯，這是冪等性的核心驗證）：
   ```bash
   python3 -c "
   import sqlite3
   sql = open('sql/schema.sql', encoding='utf-8').read()
   conn = sqlite3.connect(':memory:')
   conn.executescript(sql)
   conn.executescript(sql)  # 第二次執行，驗證冪等
   print('idempotent OK')
   conn.close()
   "
   ```
   **預期輸出**：`idempotent OK`

- [ ] **Step 3:** 編輯 `src/etl/db_loader.py` 的 `init_db()`，函式本體不變（`executescript` 邏輯不用改，因為 schema.sql 本身已經冪等了），但把 docstring 更新反映新行為：
   ```python
   def init_db(conn: sqlite3.Connection) -> None:
       """讀取 schema.sql 建立資料表（CREATE TABLE IF NOT EXISTS，冪等，不清空既有資料）。"""
       schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
       conn.executescript(schema_sql)
       logger.info("資料庫 Schema 已確認存在（未清空既有資料）")
   ```

- [ ] **Step 4:** 把 `insert_teams()`（原第 53-65 行）改名為 `upsert_teams()` 並改用 `INSERT OR IGNORE`：
   ```python
   def upsert_teams(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
       """萃取唯一球隊組合並 upsert 進 teams 表（複合主鍵 team_id + gender，已存在則略過）。"""
       teams = (
           df[["team_id", "team_name", "gender"]]
           .drop_duplicates()
           .sort_values(["gender", "team_id"])
       )
       conn.executemany(
           "INSERT OR IGNORE INTO teams (team_id, team_name, gender) VALUES (?, ?, ?)",
           teams.values.tolist(),
       )
       conn.commit()
       logger.info("已 upsert teams 表：%d 筆", len(teams))
   ```

- [ ] **Step 5:** 把 `insert_players()`（原第 68-82 行）改名為 `upsert_players()`，改用自然鍵比對：
   ```python
   def _find_existing_player_id(
       conn: sqlite3.Connection, team_id: int, gender: str,
       jersey_number, name: str,
   ) -> int | None:
       """用自然鍵 (team_id, gender, jersey_number, name) 找既有 player_id，找不到回傳 None。"""
       row = conn.execute(
           """SELECT player_id FROM players
              WHERE team_id = ? AND gender = ? AND name = ?
                AND (jersey_number = ? OR (jersey_number IS NULL AND ? IS NULL))""",
           (team_id, gender, name, jersey_number, jersey_number),
       ).fetchone()
       return row[0] if row else None


   def upsert_players(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
       """
       用自然鍵 (team_id, gender, jersey_number, name) upsert players 表。
       已存在的球員只更新 position/dob/height_cm/weight_kg，保留原 player_id，
       避免 player_match_stats 的 FK 因 player_id 改變而斷裂。
       """
       player_cols = [
           "team_id", "gender", "jersey_number", "name",
           "position", "dob", "height_cm", "weight_kg",
       ]
       players = df[player_cols]

       n_inserted = 0
       n_updated = 0
       for row in players.itertuples(index=False):
           existing_id = _find_existing_player_id(
               conn, row.team_id, row.gender, row.jersey_number, row.name,
           )
           if existing_id is None:
               conn.execute(
                   """INSERT INTO players
                      (team_id, gender, jersey_number, name, position, dob, height_cm, weight_kg)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                   (row.team_id, row.gender, row.jersey_number, row.name,
                    row.position, row.dob, row.height_cm, row.weight_kg),
               )
               n_inserted += 1
           else:
               conn.execute(
                   """UPDATE players
                      SET position = ?, dob = ?, height_cm = ?, weight_kg = ?
                      WHERE player_id = ?""",
                   (row.position, row.dob, row.height_cm, row.weight_kg, existing_id),
               )
               n_updated += 1

       conn.commit()
       logger.info("players 表 upsert 完成：新增 %d 筆、更新 %d 筆", n_inserted, n_updated)
   ```

- [ ] **Step 6:** 更新 `main()` 呼叫點（原第 101-118 行），把 `insert_teams`/`insert_players` 改成 `upsert_teams`/`upsert_players`：
   ```python
   def main():
       conn = get_connection()

       try:
           init_db(conn)
           df = load_csv()
           upsert_teams(conn, df)
           upsert_players(conn, df)

           result = verify(conn)
           print("\n===== 驗證查詢：女子組舉球員 (S)，身高 > 170cm =====")
           print(result.head(10).to_string(index=False))
       finally:
           conn.close()

       logger.info("資料庫載入完成：%s", DB_PATH)
   ```

- [ ] **Step 7:** 編輯 `src/etl/stats_crawler.py` 的 `init_stats_table()`（原第 78-111 行），移除內嵌 DDL，改成讀 `schema.sql`（只建 `player_match_stats` 這張表所需的部分——因為 `schema.sql` 是整份腳本，這裡改用「先確認表存在」的方式而不是重新 executescript 整份 schema 避免無謂重複建其他表；做法是新增一個共用的 `ensure_schema()` 概念，直接在 `stats_crawler.py` 內對齊 `db_loader.init_db()` 的做法）：
   ```python
   from src.utils.db_config import DB_PATH, get_connection
   from src.utils.logger import get_logger
   from src.utils.constants import (
       EXT_BASE, EXT_CUP_ID as CUP_ID, EXT_HEADERS as HEADERS,
       SEASON_YEAR_MAP, DEFAULT_YEAR, EXT_TEAM_MAP,
   )

   SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"

   logger = get_logger(__name__)


   def init_stats_table(conn: sqlite3.Connection) -> None:
       """確保 player_match_stats 表存在（讀 schema.sql，冪等，不清空既有資料）。"""
       conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
       logger.info("player_match_stats 表已確認存在（DDL 來源：schema.sql）")
   ```
   （`from pathlib import Path` 已在檔案原本的 import 區塊中，不用重複加。）

   同時把 `main()` 裡呼叫 `init_stats_table(conn)` 的分支說明更新（原第 237-240 行邏輯不變，只是現在 `init_stats_table` 不再清空資料，`--incremental` 與非 `--incremental` 兩種模式的差異縮小為「是否重複呼叫 fetch」而非「是否清表」）：
   ```python
   if not incremental:
       init_stats_table(conn)
       logger.warning(
           "全量模式：schema.sql 為冪等 DDL，不會清空既有 player_match_stats；"
           "若需真正重跑全量，請先手動清空該表或改用 --incremental。"
       )
   else:
       init_stats_table(conn)
       logger.info("增量模式：保留既有資料，僅新增缺少的比賽紀錄")
   ```
   > 這裡刻意保留警告訊息：因為 schema 收斂後，`--incremental`/全量兩種模式的行為差異變小了（都不再清表），這是一個對既有使用者可觀察的行為改變，必須用明確的 log 訊息告知，避免有人誤以為全量模式還會重新清空重抓。

- [ ] **Step 8:** 編輯 `src/etl/match_crawler.py` 的 `init_matches_table()`（原第 63-98 行），同樣移除內嵌 DDL：
   ```python
   SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"


   def init_matches_table(conn: sqlite3.Connection) -> None:
       """確保 matches 表存在（讀 schema.sql，冪等）。"""
       conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
       conn.commit()
       logger.info("matches 表已確認存在（DDL 來源：schema.sql）")
   ```

- [ ] **Step 9:** 靜態語法檢查：
   ```bash
   for f in src/etl/db_loader.py src/etl/stats_crawler.py src/etl/match_crawler.py; do
     python3 -c "import ast; ast.parse(open('$f', encoding='utf-8').read())" && echo "OK: $f"
   done
   ```
   **預期輸出**：3 行 `OK: <檔名>`。

- [ ] **Step 10:** 新增 `tests/test_db_loader_idempotent.py`，用隔離的 `tmp_db_path` fixture（Task 0 建立）驗證核心地雷已修好：
    ```python
    import sqlite3

    import pandas as pd
    import pytest

    from src.etl.db_loader import init_db, upsert_teams, upsert_players


    @pytest.fixture
    def conn(tmp_db_path):
        connection = sqlite3.connect(tmp_db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        yield connection
        connection.close()


    SAMPLE_DF = pd.DataFrame([
        {"team_id": 1, "team_name": "測試隊", "gender": "M",
         "jersey_number": 10, "name": "測試球員", "position": "OH",
         "dob": "2000-01-01", "height_cm": 190.0, "weight_kg": 80.0},
    ])


    def test_init_db_is_idempotent(conn):
        init_db(conn)
        init_db(conn)  # 第二次執行不應報錯
        tables = {
            row[0] for row in
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"teams", "players", "player_match_stats", "matches"} <= tables


    def test_init_db_does_not_wipe_existing_stats(conn):
        init_db(conn)
        upsert_teams(conn, SAMPLE_DF)
        upsert_players(conn, SAMPLE_DF)
        player_id = conn.execute(
            "SELECT player_id FROM players WHERE name = '測試球員'"
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO player_match_stats (player_id, match_date, total_points)
               VALUES (?, '2026-01-01', 10)""",
            (player_id,),
        )
        conn.commit()

        # 模擬重跑 db_loader：再次 init_db + upsert
        init_db(conn)
        upsert_teams(conn, SAMPLE_DF)
        upsert_players(conn, SAMPLE_DF)

        remaining = conn.execute(
            "SELECT COUNT(*) FROM player_match_stats"
        ).fetchone()[0]
        assert remaining == 1, "重跑 db_loader 不應清空 player_match_stats"


    def test_upsert_players_preserves_player_id_on_rerun(conn):
        init_db(conn)
        upsert_teams(conn, SAMPLE_DF)
        upsert_players(conn, SAMPLE_DF)
        first_id = conn.execute(
            "SELECT player_id FROM players WHERE name = '測試球員'"
        ).fetchone()[0]

        upsert_players(conn, SAMPLE_DF)  # 重跑一次

        rows = conn.execute(
            "SELECT player_id FROM players WHERE name = '測試球員'"
        ).fetchall()
        assert len(rows) == 1, "同一自然鍵不應產生重複列"
        assert rows[0][0] == first_id, "重跑後 player_id 應保持不變"
    ```

- [ ] **Step 11:** 跑測試：
    ```bash
    python -m pytest tests/test_db_loader_idempotent.py -v
    ```
    **預期輸出**：3 個測試 PASSED。

    **未實測**：需執行者在有 pandas 依賴的環境實跑。

- [ ] **Step 12:** 跑全部測試確認沒有破壞前面的 task：
    ```bash
    python -m pytest tests/ -v
    ```
    **預期輸出**：全部測試 PASSED（Task 0+1+2+3+4 累計）。

- [ ] **Step 13:** 確認行尾乾淨：`git diff --stat -w`

- [ ] **Step 14:** **[STOP：等待使用者同意 commit]**：
    ```
    fix: 拆除 db_loader 的 DROP TABLE 地雷，DDL 收斂到 schema.sql

    schema.sql 改為全面 CREATE TABLE IF NOT EXISTS（冪等，不清空
    既有資料）；db_loader 的 players/teams 寫入改為自然鍵 upsert，
    保留既有 player_id 不讓 player_match_stats 的 FK 斷裂；
    stats_crawler/match_crawler 不再各自內嵌重複的 CREATE TABLE，
    統一讀 schema.sql。
    ```

---

## Phase 1 完工檢查清單

- [ ] `git grep -n "weekly_report\|gemini\|google-genai" -- . ':!docs'` 無匹配
- [ ] `git grep -n "except ModuleNotFoundError" -- src/` 無匹配
- [ ] `sql/schema.sql` 內無任何 `DROP TABLE`
- [ ] `requirements.txt` 每一行都是 `==` 或帶 `==` 精確釘版
- [ ] `requirements-dev.txt` 存在且已釘版本，且未出現在 `requirements.txt`
- [ ] `python -m pytest tests/ -v` 全數 PASSED
- [ ] `git diff --stat -w` 對 4 個 task 的每次 commit 都確認過無行尾雜訊
- [ ] 所有 commit 都經使用者同意後才執行

完成以上檢查後，repo 即進入可承接 `2026-08-26-phase2-schema-remodel.md` 的乾淨基準狀態。
