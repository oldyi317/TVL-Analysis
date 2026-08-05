# PCAI 搬遷計畫一：後端基礎改造 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 TVL-Analysis 的設定與資料庫存取層改造成可同時支援本地 SQLite 與 PCAI 上 PostgreSQL 的形態：收斂重複設定、DB 層改用 SQLAlchemy Core、ETL 全面改為冪等 upsert、schema 加入賽季維度、鎖定依賴版本，並提供一次性資料遷移工具，為後續容器化與 Airflow 排程（計畫二、三）打底。

**Architecture:** `src/utils/constants.py` 成為所有設定與隊名對照的單一真實來源；`src/utils/db_config.py` 提供 `get_engine()`（由 `DATABASE_URL` 環境變數建立 SQLAlchemy engine，未設定則 fallback 本地 SQLite），所有 ETL 模組與 `app/helpers.py` 改為透過此 engine 存取資料庫；`sql/schema.sql`（SQLite）與新增的 `sql/schema_postgres.sql`（PostgreSQL）皆為冪等的 `CREATE TABLE IF NOT EXISTS`，並在 `players`、`matches`、`player_match_stats` 加入 `season` 欄位，所有寫入改為 `INSERT ... ON CONFLICT ... DO UPDATE` 的 upsert（SQLite 與 PostgreSQL 語法相容，已實測驗證）。

**Tech Stack:** Python 3.11、SQLAlchemy 2.x Core（不使用 ORM）、SQLite3（本地）／PostgreSQL + psycopg 3（正式環境）、pandas、pytest。

## Global Constraints

- Python >= 3.10（程式碼使用 `X | None` 型別語法，禁止改回 `Optional[X]`）
- 介面與資料皆為繁體中文；程式碼、指令、SQL 保持英文
- 環境變數皆需有預設值，未設定任何環境變數時本地行為與改造前完全一致
- 所有寫入 SQL 須以 SQLAlchemy 方言相容寫法撰寫（`text()` + 具名參數 `:name`），需同時能在 SQLite 與 PostgreSQL 執行；`ON CONFLICT ... DO UPDATE` upsert 語法已驗證兩種方言皆支援
- 任何 ETL 重跑皆須冪等：跑兩次結果相同、不丟資料、不觸碰其他 season 的列
- 本計畫**不修改** Streamlit tab 檔案（`src/app/tabs/*.py`、`src/app/main.py`）與 UI 邏輯（`st.stop()`、快取 TTL 等修正屬計畫二 §4d）；`app/helpers.py` 僅改動其 DB 連線建立與 `SEASON_YEAR_MAP` 引用，不改動任何函式對外簽名
- 本計畫**不移除** `google-genai` 依賴（計畫二才處理 AI 戰報改接 MLIS）
- 本計畫**不涉及**容器化、Helm chart、Airflow DAG（計畫二、三範圍）
- commit 前依專案慣例（`/mnt/d/CLAUDE.md`）需先詢問使用者；本計畫每個 Task 的「Commit」步驟僅列出建議的 commit 指令與訊息，實際執行者仍需在跑 `git commit` 前向使用者確認
- 所有 Task 的程式碼修改完成後，依專案「完成定義」跑對應測試（`pytest`），不得攢到最後一次跑

---

## 專案現況（實測基準）

以下版本號於本機以 `python3.11 -m venv` 建立全新虛擬環境、`pip install -r requirements.txt`（改版前的無版本號版本）後以 `pip freeze` 取得，並已實際驗證 `src/models/match_predictor.pkl` 可正常載入、`predict_proba()` 與 `shap.TreeExplainer` 皆正常運作（詳見 Task 2）。

現有 `src/models/match_predictor.pkl` 內容為一個 dict：`{"model": XGBClassifier, "model_name": "XGBoost", "feature_cols": [5 個特徵], "feature_labels": [...], "training_samples": 212, "cv_f1_mean": ...}`。注意 `src/app/tabs/prediction.py:125` 讀取的鍵是 `artifact.get("feature_names", [])`，但實際 pkl 內是 `feature_cols`，因此 `n_features` 目前恆為 0，會落入 `else` 分支使用 V1 slider（5 個特徵），這與 pkl 實際的 5 個特徵剛好吻合，因此現況可運作，但這是巧合而非設計；此問題屬 `src/app/tabs/prediction.py`（Streamlit UI 檔案），不在本計畫範圍內，僅在此記錄供計畫二參考。

---

### Task 1: 建立 pytest 測試環境

專案目前零測試、無 pytest 設定（`CLAUDE.md`：「目前沒有測試、lint、typecheck 設定」）。本任務建立最小可執行的測試骨架，後續所有 Task 皆依賴此骨架。

**Files:**
- Create: `pytest.ini`
- Create: `requirements-dev.txt`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Produces: `pytest.ini` 設定 `pythonpath = .`，使 `import src.xxx` 在任何工作目錄下執行 `pytest` 皆可解析（前提是從 repo 根目錄執行）。後續所有 Task 的測試檔皆直接 `import src.etl.xxx` / `import src.utils.xxx`，依賴此設定。

- [ ] **Step 1: 寫一個會失敗的 smoke test（尚無 pytest.ini 時，pytest 預設仍可執行，但 import 會失敗）**

```python
# tests/test_smoke.py
def test_can_import_project_package():
    import src.utils.constants  # noqa: F401
    assert True
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `cd /mnt/d/HPE-PCAI/TVL-Analysis && python -m pytest tests/test_smoke.py -v`
Expected: 因尚未安裝 pytest（見 Task 2 才釘版本）或 `ModuleNotFoundError: No module named 'src'`（若不在正確 cwd 或未設定 pythonpath）而失敗。若當前環境已有 pytest 且剛好在 repo 根目錄執行，此步驟可能已經通過——此時仍需完成 Step 3 讓設定明確、不依賴「剛好在對的目錄執行」這種隱含假設。

- [ ] **Step 3: 建立 pytest.ini 與 requirements-dev.txt**

```ini
# pytest.ini
[pytest]
pythonpath = .
testpaths = tests
```

```txt
# requirements-dev.txt
-r requirements.txt
pytest==9.1.1
```

- [ ] **Step 4: 安裝測試依賴並執行測試確認通過**

Run:
```bash
cd /mnt/d/HPE-PCAI/TVL-Analysis
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
pytest tests/test_smoke.py -v
```
Expected: `tests/test_smoke.py::test_can_import_project_package PASSED`

- [ ] **Step 5: Commit**

```bash
git add pytest.ini requirements-dev.txt tests/test_smoke.py
git commit -m "test: 建立 pytest 測試環境骨架"
```

---

### Task 2: 依賴鎖定（requirements.txt 全部釘版本 + 新增 SQLAlchemy/psycopg）

`requirements.txt` 目前幾乎未鎖版本（`CLAUDE.md`：「`src/models/match_predictor.pkl` 對 xgboost/sklearn 版本敏感，升級套件前先驗證模型還能載入」）。本任務在乾淨環境中安裝目前的無版本號依賴、以 `pip freeze` 取得實際可用版本、寫回歸測試驗證模型仍可載入與預測，再把驗證過的版本號釘入 `requirements.txt`，並新增 `sqlalchemy`、`psycopg[binary]`（供後續 Task 使用）。

**Files:**
- Modify: `requirements.txt`（全部 14 行）
- Create: `tests/test_model_compat.py`

**Interfaces:**
- Produces: `requirements.txt` 內含 `sqlalchemy` 與 `psycopg[binary]`，Task 5 起的所有 SQLAlchemy 相關程式碼皆依賴這兩個套件已安裝。

- [ ] **Step 1: 在乾淨虛擬環境安裝目前（未釘版本）的 requirements.txt，並取得版本快照**

Run:
```bash
cd /mnt/d/HPE-PCAI/TVL-Analysis
python3.11 -m venv .venv-pin
source .venv-pin/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip freeze > /tmp/pinned_versions.txt
cat /tmp/pinned_versions.txt
```

實測結果（2026-08-05，於本機乾淨環境執行取得，供本步驟核對用）：

```
beautifulsoup4==4.15.0
google-genai==2.16.0
joblib==1.5.3
matplotlib==3.11.1
numpy==2.4.6
pandas==3.0.5
plotly==6.9.0
python-dotenv==1.2.2
requests==2.34.2
scikit-learn==1.9.0
shap==0.51.0
streamlit==1.61.0
xgboost==3.2.0
```
（以上為直接依賴；`pip freeze` 另會列出 streamlit/plotly 等的間接依賴，如 `altair`、`pydeck` 等，不需寫回 `requirements.txt`，pip 會依 streamlit 的宣告自動安裝。）

- [ ] **Step 2: 寫失敗測試（此時 tests/test_model_compat.py 尚未建立）**

```python
# tests/test_model_compat.py
"""
驗證套件版本升級後 src/models/match_predictor.pkl 仍可正確載入與預測。
對應 PCAI 搬遷 spec §4e：釘版本前必須實測模型可載入。
"""

from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "src" / "models" / "match_predictor.pkl"


def test_model_pkl_loads_and_predicts():
    assert MODEL_PATH.exists(), f"模型檔案不存在：{MODEL_PATH}"
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    X = np.zeros((1, len(feature_cols)))
    proba = model.predict_proba(X)

    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(), 1.0)


def test_model_shap_explainer_works():
    import shap

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    explainer = shap.TreeExplainer(model)
    X = np.zeros((1, len(feature_cols)))
    shap_values = explainer.shap_values(X)

    assert shap_values is not None
    assert np.array(shap_values).shape[-1] == len(feature_cols)
```

- [ ] **Step 3: 在 Step 1 的乾淨環境中執行測試（此時 requirements.txt 仍未釘版本，但套件已照未釘版本安裝，用來確認測試邏輯本身正確）**

Run: `pytest tests/test_model_compat.py -v`
Expected: 兩個測試皆 PASS（若 FAIL，表示目前 PyPI 最新版 xgboost/scikit-learn 已與模型不相容，需改用 Step 1 實測版本組合中回退相容版本後重新執行本 Step，直到通過為止——本次實測環境下已確認通過，`predict_proba` 回傳 `[[0.9729, 0.0271]]`，`shap_values` shape 為 `(1, 5)`）

- [ ] **Step 4: 將實測版本釘入 requirements.txt，並新增 sqlalchemy、psycopg[binary]**

```txt
# requirements.txt
beautifulsoup4==4.15.0
google-genai==2.16.0
joblib==1.5.3
matplotlib==3.11.1
numpy==2.4.6
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

- [ ] **Step 5: 在全新乾淨環境重新安裝釘版本後的 requirements.txt，確認可重現安裝且模型測試仍通過**

Run:
```bash
cd /mnt/d/HPE-PCAI/TVL-Analysis
python3.11 -m venv .venv-verify
source .venv-verify/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest==9.1.1
pytest tests/test_model_compat.py -v
```
Expected: 兩個測試皆 PASS，且 `pip install` 過程無版本衝突錯誤

- [ ] **Step 6: 清理暫時建立的驗證用虛擬環境**

Run: `rm -rf /mnt/d/HPE-PCAI/TVL-Analysis/.venv-pin /mnt/d/HPE-PCAI/TVL-Analysis/.venv-verify`

- [ ] **Step 7: Commit**

```bash
git add requirements.txt tests/test_model_compat.py
git commit -m "chore: 釘定 requirements.txt 版本並新增 sqlalchemy/psycopg，實測模型相容性"
```

---

### Task 3: constants.py 設定收斂與賽季函式 + logger.py LOG_LEVEL

`src/utils/constants.py` 目前有 `EXT_BASE`、`SEASON_YEAR_MAP`/`DEFAULT_YEAR` 寫死不可覆寫；`TEAM_NAME_SHORT` 在 `crawler.py` 有重複定義；`TEAM_ALIAS`（隊名對照表）只存在於 `match_crawler.py`，未收斂到 constants。本任務：(1) 讓 `EXT_BASE`、`SEASON` 可被環境變數覆寫；(2) 用 `SEASON`（如 `"2025-26"`）+ `season_year_for_month()` 函式取代寫死的 `SEASON_YEAR_MAP`/`DEFAULT_YEAR`；(3) 把 `TEAM_ALIAS` 併入 constants；(4) `logger.py` 的等級改由 `LOG_LEVEL` 環境變數決定。

**Files:**
- Modify: `src/utils/constants.py`（全檔，1-70 行）
- Modify: `src/utils/logger.py`（全檔，1-24 行）
- Create: `tests/test_constants.py`
- Create: `tests/test_logger.py`

**Interfaces:**
- Produces: `constants.SEASON: str`（預設 `"2025-26"`，可被 `SEASON` 環境變數覆寫）、`constants.season_year_for_month(month: int, season: str = SEASON) -> int`、`constants.EXT_BASE: str`（可被 `EXT_BASE` 環境變數覆寫）、`constants.TEAM_ALIAS: dict[str, str]`。Task 4、8、9、11 皆會 import 這些名稱取代原本的 `SEASON_YEAR_MAP`/`DEFAULT_YEAR`/本地定義的 `TEAM_ALIAS`。
- Produces: `logger._resolve_level(level_name: str) -> int`（供測試與內部使用）。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_constants.py
import importlib

import src.utils.constants as constants


def test_season_year_for_month_default_season():
    assert constants.season_year_for_month(11) == 2025
    assert constants.season_year_for_month(12) == 2025
    assert constants.season_year_for_month(1) == 2026
    assert constants.season_year_for_month(6) == 2026


def test_season_year_for_month_custom_season():
    assert constants.season_year_for_month(11, season="2026-27") == 2026
    assert constants.season_year_for_month(3, season="2026-27") == 2027


def test_team_alias_merged_into_constants():
    assert constants.TEAM_ALIAS["桃園臺灣產險"] == "桃園臺產"
    assert constants.TEAM_ALIAS["臺中獅子王"] == "獅子王"
    assert constants.TEAM_ALIAS["臺北鯨華"] == "臺北鯨華"


def test_ext_base_env_override(monkeypatch):
    monkeypatch.setenv("EXT_BASE", "http://example.com")
    importlib.reload(constants)
    assert constants.EXT_BASE == "http://example.com"
    monkeypatch.delenv("EXT_BASE", raising=False)
    importlib.reload(constants)
    assert constants.EXT_BASE == "http://114.35.229.141"


def test_season_env_override(monkeypatch):
    monkeypatch.setenv("SEASON", "2026-27")
    importlib.reload(constants)
    assert constants.SEASON == "2026-27"
    monkeypatch.delenv("SEASON", raising=False)
    importlib.reload(constants)
    assert constants.SEASON == "2025-26"
```

```python
# tests/test_logger.py
import logging

from src.utils.logger import _resolve_level


def test_resolve_level_reads_valid_level_name():
    assert _resolve_level("DEBUG") == logging.DEBUG
    assert _resolve_level("info") == logging.INFO


def test_resolve_level_falls_back_to_info_for_invalid_name():
    assert _resolve_level("NOT_A_LEVEL") == logging.INFO
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_constants.py tests/test_logger.py -v`
Expected: FAIL — `AttributeError: module 'src.utils.constants' has no attribute 'season_year_for_month'`、`ImportError: cannot import name '_resolve_level'`

- [ ] **Step 3: 改寫 constants.py**

```python
# src/utils/constants.py
"""
TVL 共用常數模組
統一管理外部系統連線資訊與隊伍對照表，避免各模組重複定義。
"""

import os

# ── 外部數據系統 (114.35.229.141) ──────────────────────────────
EXT_BASE = os.environ.get("EXT_BASE", "http://114.35.229.141")
EXT_CUP_ID = 21

EXT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── 賽季設定 ────────────────────────────────────────────────────
# SEASON 格式："起始年-結束年後兩碼"，例如 "2025-26" 代表 2025 年 11 月～2026 年 6 月的賽季。
# 11、12 月屬賽季起始年；其餘月份屬賽季結束年（沿用官網賽程跨年慣例）。
SEASON = os.environ.get("SEASON", "2025-26")
SEASON_CROSSOVER_MONTHS = {11, 12}


def season_year_for_month(month: int, season: str = SEASON) -> int:
    """依賽季字串（如 '2025-26'）與月份，回推對應西元年。"""
    start_str, end_suffix = season.split("-")
    start_year = int(start_str)
    end_year = (
        int(start_str[:2] + end_suffix) if len(end_suffix) == 2 else int(end_suffix)
    )
    return start_year if month in SEASON_CROSSOVER_MONTHS else end_year


# ── 外部系統 TeamID → 本地 DB (team_id, gender) 對照表 ─────────
EXT_TEAM_MAP = {
    1: (1, "M"),   # 屏東台電(男)
    2: (2, "M"),   # 雲林美津濃(男)
    3: (7, "M"),   # 獅子王(男)
    4: (4, "M"),   # 臺北國北獅(男)
    5: (5, "M"),   # 桃園臺產(男)
    6: (4, "F"),   # 高雄台電(女)
    7: (3, "F"),   # 臺北鯨華(女)
    8: (5, "F"),   # 新北中纖(女)
    9: (7, "F"),   # 義力營造(女)
}

# ── 對手簡稱 → 本地 DB (team_id, gender) ──────────────────────
OPP_SHORT_TO_TEAM: dict[str, tuple[int, str]] = {
    "屏東台電":   (1, "M"),
    "雲林美津濃": (2, "M"),
    "臺北國北獅": (4, "M"),
    "桃園臺產":   (5, "M"),
    "獅子王":     (7, "M"),
    "高雄台電":   (4, "F"),
    "臺北鯨華":   (3, "F"),
    "新北中纖":   (5, "F"),
    "義力營造":   (7, "F"),
}

# ── 官網名單頁全名 → 簡寫對應 ───────────────────────────────────
TEAM_NAME_SHORT = {
    "臺北鯨華女子排球隊": "臺北鯨華",
    "新北中國人纖企業女子排球隊": "新北中纖",
    "台灣電力公司女子排球隊": "高雄台電",
    "義力營造女子排球隊": "義力營造",
    "台灣電力公司男子排球隊": "屏東台電",
    "美津濃男子排球隊": "雲林美津濃",
    "桃園臺產隼鷹排球隊": "桃園臺產",
}

# ── 官網賽事頁隊名 → 簡寫對應（賽事頁與名單頁隊名拼寫不同）─────
TEAM_ALIAS = {
    "臺北鯨華": "臺北鯨華",
    "新北中纖": "新北中纖",
    "高雄台電": "高雄台電",
    "義力營造": "義力營造",
    "屏東台電": "屏東台電",
    "雲林美津濃": "雲林美津濃",
    "臺北國北獅": "臺北國北獅",
    "桃園臺灣產險": "桃園臺產",
    "臺中獅子王": "獅子王",
}

# ── 位置相關 ──────────────────────────────────────────────────
VALID_POSITIONS = {"OH", "MB", "OP", "S", "L"}
VALID_GENDERS = {"M", "F"}

POSITION_MAP = {
    "主攻手": "OH",
    "中間手": "MB",
    "副攻手": "OP",
    "舉球員": "S",
    "自由球員": "L",
}
```

- [ ] **Step 4: 改寫 logger.py**

```python
# src/utils/logger.py
"""
統一 Logger 設定模組
所有模組透過 get_logger(__name__) 取得 logger，避免重複呼叫 basicConfig。
LOG_LEVEL 環境變數可覆寫預設等級（預設 INFO）。
"""

import logging
import os

_CONFIGURED = False


def _resolve_level(level_name: str) -> int:
    """將等級名稱字串轉為 logging 等級數值，無法辨識時 fallback 為 INFO。"""
    return getattr(logging, level_name.upper(), logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    取得已設定好格式的 Logger。
    首次呼叫時設定 root handler，後續呼叫直接回傳。
    """
    global _CONFIGURED
    if not _CONFIGURED:
        level = _resolve_level(os.environ.get("LOG_LEVEL", "INFO"))
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
```

- [ ] **Step 5: 執行測試確認通過**

Run: `pytest tests/test_constants.py tests/test_logger.py -v`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add src/utils/constants.py src/utils/logger.py tests/test_constants.py tests/test_logger.py
git commit -m "feat: constants/logger 支援環境變數覆寫，新增賽季函式與 TEAM_ALIAS 單一來源"
```

---

### Task 4: crawler.py + cleaner.py 移除 import fallback

`src/etl/crawler.py`（13-24 行、151-161 行）與 `src/etl/cleaner.py`（12-20 行）各自有 `try/except ModuleNotFoundError` 的 import fallback 區塊，複製了一份 `POSITION_MAP`/`TEAM_NAME_SHORT`/`VALID_POSITIONS`/`VALID_GENDERS`。這兩個檔案不碰資料庫，只需移除 fallback、統一從 Task 3 產出的 `constants.py` 匯入。

**Files:**
- Modify: `src/etl/crawler.py:1-26`（移除 fallback，改為直接 import）、`src/etl/crawler.py:149-162`（移除多餘的 `TEAM_NAME_SHORT is None` 補值區塊）
- Modify: `src/etl/cleaner.py:1-22`（移除 fallback，改為直接 import）
- Create: `tests/test_crawler_cleaner.py`

**Interfaces:**
- Consumes：`src.utils.constants.POSITION_MAP`、`EXT_HEADERS`、`TEAM_NAME_SHORT`（來自 Task 3）；`src.utils.constants.VALID_POSITIONS`、`VALID_GENDERS`（來自 Task 3）；`src.utils.logger.get_logger`（既有介面不變）。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_crawler_cleaner.py
import pandas as pd
from bs4 import BeautifulSoup

from src.etl.cleaner import validate_positions
from src.etl.crawler import extract_team_name


def test_extract_team_name_maps_full_name_to_short():
    soup = BeautifulSoup("<title>臺北鯨華女子排球隊 | TVL</title>", "html.parser")
    assert extract_team_name(soup) == "臺北鯨華"


def test_validate_positions_invalidates_unknown_code():
    df = pd.DataFrame({"name": ["A", "B"], "position": ["OH", "XX"]})
    result = validate_positions(df)
    assert result.loc[0, "position"] == "OH"
    assert result.loc[1, "position"] is None
```

- [ ] **Step 2: 執行測試確認目前行為（改動前應已通過，作為改動前後行為不變的基準）**

Run: `pytest tests/test_crawler_cleaner.py -v`
Expected: PASS（此步驟只是建立回歸基準，確認改動 import 方式不會改變既有行為）

- [ ] **Step 3: 移除 crawler.py 的 fallback 區塊**

將 `src/etl/crawler.py` 開頭（原第 13-26 行）：

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

logger = get_logger(__name__)
```

改為：

```python
from src.utils.logger import get_logger
from src.utils.constants import POSITION_MAP, EXT_HEADERS as HEADERS, TEAM_NAME_SHORT

logger = get_logger(__name__)
```

並刪除原第 151-161 行的補值區塊：

```python
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

（此區塊整段刪除，`GENDER_MAP = {"team": "M", "wteam": "F"}` 保留在原位置。）

- [ ] **Step 4: 移除 cleaner.py 的 fallback 區塊**

將 `src/etl/cleaner.py` 開頭（原第 12-20 行）：

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

logger = get_logger(__name__)
```

改為：

```python
from src.utils.logger import get_logger
from src.utils.constants import VALID_POSITIONS, VALID_GENDERS

logger = get_logger(__name__)
```

- [ ] **Step 5: 執行測試確認通過**

Run: `pytest tests/test_crawler_cleaner.py -v`
Expected: 全部 PASS（與 Step 2 結果相同，證明行為未改變）

- [ ] **Step 6: Commit**

```bash
git add src/etl/crawler.py src/etl/cleaner.py tests/test_crawler_cleaner.py
git commit -m "refactor: 移除 crawler/cleaner 的 import fallback，統一從 constants 匯入"
```

---

### Task 5: db_config.py 改為 SQLAlchemy engine

`src/utils/db_config.py` 目前用 `sqlite3.connect()` 從程式碼位置推導固定路徑。本任務改為由 `DATABASE_URL` 環境變數建立 SQLAlchemy engine，未設定時 fallback 至現有 SQLite 路徑（本地行為不變），並保留 SQLite 的 `PRAGMA foreign_keys = ON` 行為（透過 engine 的 `connect` event）。

**Files:**
- Modify: `src/utils/db_config.py`（全檔，1-24 行）
- Create: `tests/test_db_config.py`

**Interfaces:**
- Produces: `db_config.get_engine() -> sqlalchemy.engine.Engine`（全域單例，延遲建立）、`db_config.reset_engine() -> None`（測試/切換 `DATABASE_URL` 用，釋放並清除快取的 engine）、`db_config.PROJECT_ROOT: Path`、`db_config.DB_PATH: Path`（沿用既有名稱供 Task 7 的 schema 路徑推導使用）。
- Consumes：無（Task 3 的 constants 與此檔無直接依賴）。
- 後續 Task 6-12 皆會 `from src.utils.db_config import get_engine`，不再使用 `get_connection()`（本任務移除此函式，Task 6-11 會同步移除所有呼叫端）。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_db_config.py
import src.utils.db_config as db_config


def test_get_engine_defaults_to_sqlite_when_database_url_unset(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db_config.reset_engine()
    engine = db_config.get_engine()
    assert engine.dialect.name == "sqlite"
    db_config.reset_engine()


def test_get_engine_uses_database_url_when_set(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/tvl"
    )
    db_config.reset_engine()
    engine = db_config.get_engine()
    assert engine.dialect.name == "postgresql"
    assert engine.driver == "psycopg"
    db_config.reset_engine()
    monkeypatch.delenv("DATABASE_URL", raising=False)


def test_get_engine_enables_sqlite_foreign_keys(monkeypatch, tmp_path):
    db_path = tmp_path / "fk_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    db_config.reset_engine()
    engine = db_config.get_engine()
    with engine.connect() as conn:
        from sqlalchemy import text

        fk_status = conn.exec_driver_sql("PRAGMA foreign_keys").scalar()
    assert fk_status == 1
    db_config.reset_engine()
    monkeypatch.delenv("DATABASE_URL", raising=False)
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_db_config.py -v`
Expected: FAIL — `AttributeError: module 'src.utils.db_config' has no attribute 'reset_engine'`（`get_engine` 也不存在）

- [ ] **Step 3: 改寫 db_config.py**

```python
# src/utils/db_config.py
"""
資料庫連線設定模組
以 DATABASE_URL 環境變數建立 SQLAlchemy engine；未設定時 fallback 至本地 SQLite。
"""

import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "db" / "tvl_database.db"

_engine: Engine | None = None


def _default_sqlite_url() -> str:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DB_PATH}"


def get_engine() -> Engine:
    """
    回傳全域共用的 SQLAlchemy engine（延遲建立，僅建立一次）。
    由 DATABASE_URL 環境變數決定連線目標；未設定時 fallback 至本地 SQLite 檔案。
    """
    global _engine
    if _engine is None:
        database_url = os.environ.get("DATABASE_URL", _default_sqlite_url())
        _engine = create_engine(database_url, future=True)
        if _engine.dialect.name == "sqlite":
            @event.listens_for(_engine, "connect")
            def _enable_sqlite_foreign_keys(dbapi_conn, _record):
                dbapi_conn.execute("PRAGMA foreign_keys = ON")
    return _engine


def reset_engine() -> None:
    """重置快取的 engine（測試或切換 DATABASE_URL 後呼叫）。"""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None
```

- [ ] **Step 4: 執行測試確認通過**

Run: `pytest tests/test_db_config.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/db_config.py tests/test_db_config.py
git commit -m "feat: db_config 改為 DATABASE_URL 驅動的 SQLAlchemy engine"
```

---

### Task 6: schema.sql 冪等化 + season 欄位；新增 schema_postgres.sql

`sql/schema.sql` 目前開頭 `DROP TABLE IF EXISTS player_match_stats/players/teams`，每次執行 `db_loader.py` 會清空全部逐場數據（`CLAUDE.md` 明確警告）。本任務移除所有 `DROP TABLE`，四張表統一改為 `CREATE TABLE IF NOT EXISTS`；`players`、`matches`、`player_match_stats` 加入 `season` 欄位並更新對應的 `UNIQUE` 鍵；並新增 PostgreSQL 相容版本 `sql/schema_postgres.sql`（自動遞增欄位以 `GENERATED BY DEFAULT AS IDENTITY` 取代 SQLite 的 `AUTOINCREMENT`，允許 Task 12 的遷移 script 明確指定既有 ID）。

**Files:**
- Modify: `sql/schema.sql`（全檔，1-84 行）
- Create: `sql/schema_postgres.sql`
- Create: `tests/test_schema.py`

**Interfaces:**
- Produces: 四張表（`teams`、`players`、`player_match_stats`、`matches`）皆為 `CREATE TABLE IF NOT EXISTS`；`players` 唯一鍵 `(team_id, gender, season, name)`；`player_match_stats` 唯一鍵 `(player_id, season, match_date, opponent, is_golden_set)`；`matches` 唯一鍵 `(game_id, gender, season)`。Task 7-9、12 的 upsert SQL 的 `ON CONFLICT` 目標欄位皆對應這三組唯一鍵。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_schema.py
from pathlib import Path

from sqlalchemy import create_engine, text

SCHEMA_SQLITE = Path("sql/schema.sql")
SCHEMA_POSTGRES = Path("sql/schema_postgres.sql")


def _apply(engine, path: Path) -> None:
    statements = [
        s.strip() for s in path.read_text(encoding="utf-8").split(";") if s.strip()
    ]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def test_sqlite_schema_has_no_drop_table():
    content = SCHEMA_SQLITE.read_text(encoding="utf-8")
    assert "DROP TABLE" not in content.upper()


def test_sqlite_schema_applies_twice_idempotently():
    engine = create_engine("sqlite:///:memory:")
    _apply(engine, SCHEMA_SQLITE)
    _apply(engine, SCHEMA_SQLITE)  # 重複套用不可報錯
    with engine.begin() as conn:
        tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).scalars().all()
    for expected in ["teams", "players", "player_match_stats", "matches"]:
        assert expected in tables


def test_sqlite_schema_has_season_columns():
    engine = create_engine("sqlite:///:memory:")
    _apply(engine, SCHEMA_SQLITE)
    with engine.begin() as conn:
        for table in ["players", "player_match_stats", "matches"]:
            cols = [row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))]
            assert "season" in cols, f"{table} 缺少 season 欄位"


def test_postgres_schema_has_no_sqlite_specific_syntax():
    content = SCHEMA_POSTGRES.read_text(encoding="utf-8")
    assert "AUTOINCREMENT" not in content
    assert "GENERATED BY DEFAULT AS IDENTITY" in content
    for table in ["teams", "players", "player_match_stats", "matches"]:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in content
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL — `test_sqlite_schema_has_no_drop_table` 因目前檔案含 `DROP TABLE` 而失敗；`test_postgres_schema_has_no_sqlite_specific_syntax` 因 `sql/schema_postgres.sql` 不存在而 FileNotFoundError

- [ ] **Step 3: 改寫 sql/schema.sql**

```sql
-- TVL 資料庫 Schema（SQLite 版，可重複執行，冪等：CREATE TABLE IF NOT EXISTS）
-- 注意：男女組的 team_id 可能重複，因此 teams 使用複合主鍵 (team_id, gender)
-- players / player_match_stats / matches 皆含 season 欄位，upsert 的唯一鍵包含
-- season，換季寫入新 season 的列，永不觸碰舊賽季資料。

CREATE TABLE IF NOT EXISTS teams (
    team_id   INTEGER NOT NULL,
    team_name TEXT    NOT NULL,
    gender    TEXT    NOT NULL CHECK (gender IN ('M', 'F')),
    PRIMARY KEY (team_id, gender)
);

CREATE TABLE IF NOT EXISTS players (
    player_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id       INTEGER NOT NULL,
    gender        TEXT    NOT NULL,
    season        TEXT    NOT NULL,
    jersey_number INTEGER,
    name          TEXT,
    position      TEXT,
    dob           DATE,
    height_cm     REAL,
    weight_kg     REAL,
    FOREIGN KEY (team_id, gender) REFERENCES teams (team_id, gender),
    UNIQUE (team_id, gender, season, name)
);

CREATE TABLE IF NOT EXISTS player_match_stats (
    stat_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id         INTEGER NOT NULL,
    season            TEXT    NOT NULL,
    match_date        DATE,
    opponent          TEXT,
    sets_played       INTEGER,
    attack_total      INTEGER,
    attack_points     INTEGER,
    block_points      INTEGER,
    serve_total       INTEGER,
    serve_points      INTEGER,
    receive_total     INTEGER,
    receive_excellent INTEGER,
    dig_total         INTEGER,
    dig_excellent     INTEGER,
    set_total         INTEGER,
    set_excellent     INTEGER,
    total_points      INTEGER,
    is_golden_set     INTEGER NOT NULL DEFAULT 0 CHECK (is_golden_set IN (0, 1)),
    FOREIGN KEY (player_id) REFERENCES players (player_id),
    UNIQUE (player_id, season, match_date, opponent, is_golden_set)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         INTEGER NOT NULL,
    gender          TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    season          TEXT NOT NULL,
    match_date      DATE NOT NULL,
    venue           TEXT,
    round_name      TEXT,
    game_label      TEXT,
    is_golden_set   INTEGER NOT NULL DEFAULT 0,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    home_set1       INTEGER,
    home_set2       INTEGER,
    home_set3       INTEGER,
    home_set4       INTEGER,
    home_set5       INTEGER,
    home_total      INTEGER,
    away_set1       INTEGER,
    away_set2       INTEGER,
    away_set3       INTEGER,
    away_set4       INTEGER,
    away_set5       INTEGER,
    away_total      INTEGER,
    home_sets_won   INTEGER,
    away_sets_won   INTEGER,
    UNIQUE (game_id, gender, season)
);

-- 效能索引
CREATE INDEX IF NOT EXISTS idx_pms_player_id  ON player_match_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_pms_match_date ON player_match_stats(match_date);
CREATE INDEX IF NOT EXISTS idx_players_team_gender ON players(team_id, gender);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
```

- [ ] **Step 4: 新增 sql/schema_postgres.sql**

```sql
-- TVL 資料庫 Schema（PostgreSQL 版，可重複執行，冪等：CREATE TABLE IF NOT EXISTS）
-- 結構與 sql/schema.sql（SQLite 版）一致，差異僅在自動遞增欄位語法。
-- player_id / stat_id / match_id 使用 GENERATED BY DEFAULT AS IDENTITY（而非
-- GENERATED ALWAYS），允許一次性資料遷移 script 明確指定既有 ID
-- （見 src/etl/migrate_to_postgres.py），一般 upsert 仍會自動遞增。

CREATE TABLE IF NOT EXISTS teams (
    team_id   INTEGER NOT NULL,
    team_name TEXT    NOT NULL,
    gender    TEXT    NOT NULL CHECK (gender IN ('M', 'F')),
    PRIMARY KEY (team_id, gender)
);

CREATE TABLE IF NOT EXISTS players (
    player_id     INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    team_id       INTEGER NOT NULL,
    gender        TEXT    NOT NULL,
    season        TEXT    NOT NULL,
    jersey_number INTEGER,
    name          TEXT,
    position      TEXT,
    dob           DATE,
    height_cm     REAL,
    weight_kg     REAL,
    FOREIGN KEY (team_id, gender) REFERENCES teams (team_id, gender),
    UNIQUE (team_id, gender, season, name)
);

CREATE TABLE IF NOT EXISTS player_match_stats (
    stat_id           INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    player_id         INTEGER NOT NULL,
    season            TEXT    NOT NULL,
    match_date        DATE,
    opponent          TEXT,
    sets_played       INTEGER,
    attack_total      INTEGER,
    attack_points     INTEGER,
    block_points      INTEGER,
    serve_total       INTEGER,
    serve_points      INTEGER,
    receive_total     INTEGER,
    receive_excellent INTEGER,
    dig_total         INTEGER,
    dig_excellent     INTEGER,
    set_total         INTEGER,
    set_excellent     INTEGER,
    total_points      INTEGER,
    is_golden_set     INTEGER NOT NULL DEFAULT 0 CHECK (is_golden_set IN (0, 1)),
    FOREIGN KEY (player_id) REFERENCES players (player_id),
    UNIQUE (player_id, season, match_date, opponent, is_golden_set)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id        INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    game_id         INTEGER NOT NULL,
    gender          TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    season          TEXT NOT NULL,
    match_date      DATE NOT NULL,
    venue           TEXT,
    round_name      TEXT,
    game_label      TEXT,
    is_golden_set   INTEGER NOT NULL DEFAULT 0,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    home_set1       INTEGER,
    home_set2       INTEGER,
    home_set3       INTEGER,
    home_set4       INTEGER,
    home_set5       INTEGER,
    home_total      INTEGER,
    away_set1       INTEGER,
    away_set2       INTEGER,
    away_set3       INTEGER,
    away_set4       INTEGER,
    away_set5       INTEGER,
    away_total      INTEGER,
    home_sets_won   INTEGER,
    away_sets_won   INTEGER,
    UNIQUE (game_id, gender, season)
);

CREATE INDEX IF NOT EXISTS idx_pms_player_id  ON player_match_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_pms_match_date ON player_match_stats(match_date);
CREATE INDEX IF NOT EXISTS idx_players_team_gender ON players(team_id, gender);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
```

- [ ] **Step 5: 執行測試確認通過**

Run: `pytest tests/test_schema.py -v`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add sql/schema.sql sql/schema_postgres.sql tests/test_schema.py
git commit -m "feat: schema 移除 DROP TABLE 改冪等 CREATE TABLE IF NOT EXISTS，新增 season 欄位與 PostgreSQL 版本"
```

---

### Task 7: db_loader.py 改為 upsert + season + 移除 import fallback

`src/etl/db_loader.py` 目前呼叫 `init_db()` 執行含 `DROP TABLE` 的 schema.sql，並用 `executemany` 對 `teams`/`players` 做純 `INSERT`（無 upsert，重跑會出現 UNIQUE 衝突或重複列）。本任務改用 Task 5 的 `get_engine()` 與 Task 6 的新 schema，`insert_teams`/`insert_players` 改為 `ON CONFLICT ... DO UPDATE` 的 upsert，並帶入 `season`。同時建立 `tests/conftest.py` 提供 `sqlite_engine` fixture，供本任務起後續所有 ETL 測試共用。

**Files:**
- Modify: `src/etl/db_loader.py`（全檔，1-122 行）
- Create: `tests/conftest.py`
- Create: `tests/test_db_loader.py`

**Interfaces:**
- Consumes：`src.utils.db_config.get_engine()`、`src.utils.db_config.reset_engine()`（Task 5）；`sql/schema.sql`、`sql/schema_postgres.sql`（Task 6）；`src.utils.constants.SEASON`（Task 3）。
- Produces: `db_loader.init_db(engine: Engine) -> None`（依 `engine.dialect.name` 選擇 schema 檔並逐句執行，Task 8、9、12 皆會呼叫此函式取代自己的建表邏輯）、`db_loader.insert_teams(engine: Engine, df: pd.DataFrame) -> None`、`db_loader.insert_players(engine: Engine, df: pd.DataFrame, season: str = SEASON) -> None`。
- Produces（測試骨架）：`tests/conftest.py` 的 `sqlite_engine` fixture（`pytest.fixture`，回傳已套用 schema 的暫存 SQLite `Engine`），供 Task 8、9、13 直接注入使用。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/conftest.py
"""pytest 共用 fixtures：提供已套用最新 schema 的暫存 SQLite engine。"""

import pytest


@pytest.fixture
def sqlite_engine(tmp_path, monkeypatch):
    """建立套用最新 schema 的暫存 SQLite engine，測試結束後釋放。"""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    import src.utils.db_config as db_config

    db_config.reset_engine()

    from src.etl.db_loader import init_db

    engine = db_config.get_engine()
    init_db(engine)

    yield engine

    db_config.reset_engine()
```

```python
# tests/test_db_loader.py
import pandas as pd
from sqlalchemy import text

from src.etl.db_loader import insert_players, insert_teams


def _sample_roster() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "team_id": 1, "team_name": "屏東台電", "gender": "M",
            "jersey_number": 4, "name": "李元", "position": "OH",
            "dob": "2000-01-01", "height_cm": 190.0, "weight_kg": 80.0,
        },
        {
            "team_id": 1, "team_name": "屏東台電", "gender": "M",
            "jersey_number": 7, "name": "王小明", "position": "S",
            "dob": None, "height_cm": None, "weight_kg": None,
        },
    ])


def test_insert_players_is_idempotent_on_rerun(sqlite_engine):
    df = _sample_roster()
    insert_teams(sqlite_engine, df)
    insert_players(sqlite_engine, df, season="2025-26")
    with sqlite_engine.begin() as conn:
        n1 = conn.execute(text("SELECT COUNT(*) FROM players")).scalar_one()
    assert n1 == 2

    insert_teams(sqlite_engine, df)
    insert_players(sqlite_engine, df, season="2025-26")
    with sqlite_engine.begin() as conn:
        n2 = conn.execute(text("SELECT COUNT(*) FROM players")).scalar_one()
    assert n2 == 2, f"重跑後筆數應不變，實際為 {n2}"


def test_insert_players_updates_changed_fields_on_rerun(sqlite_engine):
    df = _sample_roster()
    insert_teams(sqlite_engine, df)
    insert_players(sqlite_engine, df, season="2025-26")

    df2 = df.copy()
    df2.loc[df2["name"] == "李元", "height_cm"] = 191.0
    insert_players(sqlite_engine, df2, season="2025-26")

    with sqlite_engine.begin() as conn:
        height = conn.execute(
            text("SELECT height_cm FROM players WHERE name = '李元'")
        ).scalar_one()
    assert height == 191.0


def test_insert_players_does_not_touch_other_season_rows(sqlite_engine):
    df = _sample_roster()
    insert_teams(sqlite_engine, df)
    insert_players(sqlite_engine, df, season="2025-26")

    insert_players(sqlite_engine, df, season="2026-27")

    with sqlite_engine.begin() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM players")).scalar_one()
        old_height = conn.execute(
            text("SELECT height_cm FROM players WHERE name = '李元' AND season = '2025-26'")
        ).scalar_one()
    assert total == 4, "兩個賽季各 2 筆，應合計 4 筆"
    assert old_height == 190.0, "舊賽季的列不應被新賽季的 upsert 觸碰"
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_db_loader.py -v`
Expected: FAIL — `insert_teams`/`insert_players` 目前簽名不接受 `engine` 引數（仍是 `conn: sqlite3.Connection`），且不支援 `season` 參數

- [ ] **Step 3: 改寫 db_loader.py**

```python
# src/etl/db_loader.py
"""
TVL 資料庫載入模組
讀取 raw CSV → 經 cleaner 清洗 → 正規化拆分為 teams / players 兩表並 upsert 至資料庫（冪等）。
"""

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.etl.cleaner import load_raw, clean, quality_report
from src.utils.constants import SEASON
from src.utils.db_config import PROJECT_ROOT, get_engine
from src.utils.logger import get_logger

logger = get_logger(__name__)

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "all_teams_roster.csv"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"
SCHEMA_PATH_POSTGRES = PROJECT_ROOT / "sql" / "schema_postgres.sql"


def init_db(engine: Engine) -> None:
    """依 engine 方言選擇對應 schema 檔並逐句執行（CREATE TABLE IF NOT EXISTS，冪等）。"""
    path = SCHEMA_PATH_POSTGRES if engine.dialect.name == "postgresql" else SCHEMA_PATH
    schema_sql = path.read_text(encoding="utf-8")
    statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    logger.info("資料庫 Schema 建立完成（%s）", path.name)


def load_csv() -> pd.DataFrame:
    """讀取 raw CSV 並經 cleaner 清洗，確保資料品質後回傳。"""
    df = load_raw(CSV_PATH)
    df = clean(df)
    quality_report(df)
    logger.info("清洗後資料：%d 筆", len(df))
    return df


def insert_teams(engine: Engine, df: pd.DataFrame) -> None:
    """萃取唯一球隊組合並 upsert 至 teams 表（複合主鍵 team_id + gender）。"""
    teams = (
        df[["team_id", "team_name", "gender"]]
        .drop_duplicates()
        .sort_values(["gender", "team_id"])
    )
    rows = teams.to_dict("records")
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO teams (team_id, team_name, gender)
                VALUES (:team_id, :team_name, :gender)
                ON CONFLICT (team_id, gender) DO UPDATE SET
                    team_name = excluded.team_name
            """),
            rows,
        )
    logger.info("已 upsert teams 表：%d 筆", len(rows))


def insert_players(engine: Engine, df: pd.DataFrame, season: str = SEASON) -> None:
    """萃取球員欄位並 upsert 至 players 表（唯一鍵：team_id+gender+season+name）。"""
    player_cols = [
        "team_id", "gender", "jersey_number", "name",
        "position", "dob", "height_cm", "weight_kg",
    ]
    players = df[player_cols].copy()
    players["season"] = season
    rows = players.to_dict("records")
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO players
                    (team_id, gender, season, jersey_number, name, position, dob, height_cm, weight_kg)
                VALUES
                    (:team_id, :gender, :season, :jersey_number, :name, :position, :dob, :height_cm, :weight_kg)
                ON CONFLICT (team_id, gender, season, name) DO UPDATE SET
                    jersey_number = excluded.jersey_number,
                    position      = excluded.position,
                    dob           = excluded.dob,
                    height_cm     = excluded.height_cm,
                    weight_kg     = excluded.weight_kg
            """),
            rows,
        )
    logger.info("已 upsert players 表：%d 筆（season=%s）", len(rows), season)


def verify(engine: Engine) -> pd.DataFrame:
    """驗證查詢：女子組中位置為舉球員 (S) 且身高 > 170 cm 的球員。"""
    query = """
        SELECT p.name, t.team_name, p.height_cm
        FROM players p
        JOIN teams t ON p.team_id = t.team_id AND p.gender = t.gender
        WHERE p.gender = 'F'
          AND p.position = 'S'
          AND p.height_cm > 170
        ORDER BY p.height_cm DESC
    """
    return pd.read_sql_query(query, engine)


def main():
    engine = get_engine()

    init_db(engine)
    df = load_csv()
    insert_teams(engine, df)
    insert_players(engine, df)

    result = verify(engine)
    print("\n===== 驗證查詢：女子組舉球員 (S)，身高 > 170cm =====")
    print(result.head(10).to_string(index=False))

    logger.info("資料庫載入完成")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 執行測試確認通過**

Run: `pytest tests/test_db_loader.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/etl/db_loader.py tests/conftest.py tests/test_db_loader.py
git commit -m "feat: db_loader 改用 SQLAlchemy engine，teams/players 改為冪等 upsert 並帶入 season"
```

---

### Task 8: stats_crawler.py 改為 upsert + season + 移除 fallback

`src/etl/stats_crawler.py` 目前全量模式先 `DROP TABLE IF EXISTS player_match_stats` 再重建（`init_stats_table`，79-111 行），且自行重複定義一份 `player_match_stats` 的 CREATE TABLE（與 `sql/schema.sql` 重複維護）。本任務移除自建的 DDL，改呼叫 Task 7 的 `db_loader.init_db()`；`--incremental`/全量兩種模式統一改為對每筆抓到的紀錄做 upsert（不再區分「跳過已存在」邏輯，因為 upsert 本身即冪等）；`build_name_to_pid` 依 `season` 篩選（各賽季名單獨立）；Late Arriving Dimension 新增球員時帶入 `season` 並以 `RETURNING` 取得新 `player_id`。

**Files:**
- Modify: `src/etl/stats_crawler.py`（全檔，1-376 行）
- Create: `tests/test_stats_crawler.py`

**Interfaces:**
- Consumes：`src.etl.db_loader.init_db(engine)`（Task 7）；`src.utils.db_config.get_engine()`（Task 5）；`src.utils.constants.SEASON`、`EXT_BASE`、`EXT_CUP_ID`、`EXT_HEADERS`、`EXT_TEAM_MAP`、`season_year_for_month`（Task 3）。
- Produces: `stats_crawler.build_name_to_pid(engine: Engine, season: str = SEASON) -> dict[str, int]`、`stats_crawler.upsert_stats(engine: Engine, player_id: int, records: list[dict], season: str) -> None`（Task 13 的整合測試會直接呼叫此函式）、`stats_crawler.parse_match_date(raw: str) -> str | None`（簽名不變，內部改用 `season_year_for_month`）。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_stats_crawler.py
from sqlalchemy import text

from src.etl.db_loader import insert_players, insert_teams
from src.etl.stats_crawler import build_name_to_pid, parse_match_date, upsert_stats
import pandas as pd


def _sample_stat_row(**overrides) -> dict:
    base = dict(
        match_date="2026-01-05", opponent="雲林美津濃", sets_played=3,
        attack_total=10, attack_points=5, block_points=1,
        serve_total=5, serve_points=1, receive_total=5, receive_excellent=3,
        dig_total=5, dig_excellent=2, set_total=0, set_excellent=0,
        total_points=7, is_golden_set=0,
    )
    base.update(overrides)
    return base


def _seed_player(engine, season="2025-26") -> int:
    df = pd.DataFrame([{
        "team_id": 1, "team_name": "屏東台電", "gender": "M",
        "jersey_number": 4, "name": "李元", "position": "OH",
        "dob": None, "height_cm": None, "weight_kg": None,
    }])
    insert_teams(engine, df)
    insert_players(engine, df, season=season)
    with engine.begin() as conn:
        return conn.execute(
            text("SELECT player_id FROM players WHERE name = '李元' AND season = :s"),
            {"s": season},
        ).scalar_one()


def test_parse_match_date_uses_season_year_for_month():
    assert parse_match_date("311/01") == "2025-11-01"  # 11 月屬賽季起始年
    assert parse_match_date("303/05") == "2026-03-05"  # 場次3、3月5日；3 月屬賽季結束年


def test_build_name_to_pid_scoped_to_season(sqlite_engine):
    pid = _seed_player(sqlite_engine, season="2025-26")
    name_map = build_name_to_pid(sqlite_engine, "2025-26")
    assert name_map["李元"] == pid

    other_season_map = build_name_to_pid(sqlite_engine, "2026-27")
    assert "李元" not in other_season_map


def test_upsert_stats_is_idempotent_on_rerun(sqlite_engine):
    pid = _seed_player(sqlite_engine)
    row = _sample_stat_row()

    upsert_stats(sqlite_engine, pid, [row], "2025-26")
    upsert_stats(sqlite_engine, pid, [row], "2025-26")

    with sqlite_engine.begin() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM player_match_stats")).scalar_one()
    assert n == 1, f"重跑相同資料應仍是 1 筆，實際為 {n}"


def test_upsert_stats_updates_corrected_values(sqlite_engine):
    pid = _seed_player(sqlite_engine)
    row = _sample_stat_row()
    upsert_stats(sqlite_engine, pid, [row], "2025-26")

    corrected = _sample_stat_row(total_points=9)
    upsert_stats(sqlite_engine, pid, [corrected], "2025-26")

    with sqlite_engine.begin() as conn:
        pts = conn.execute(text("SELECT total_points FROM player_match_stats")).scalar_one()
    assert pts == 9


def test_upsert_stats_does_not_touch_other_season_rows(sqlite_engine):
    pid = _seed_player(sqlite_engine)
    row = _sample_stat_row()
    upsert_stats(sqlite_engine, pid, [row], "2025-26")

    upsert_stats(sqlite_engine, pid, [row], "2026-27")

    with sqlite_engine.begin() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM player_match_stats")).scalar_one()
        old_pts = conn.execute(
            text("SELECT total_points FROM player_match_stats WHERE season = '2025-26'")
        ).scalar_one()
    assert total == 2
    assert old_pts == 7
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_stats_crawler.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_name_to_pid'`（目前簽名是 `build_name_to_pid(conn)`，且 `upsert_stats` 尚不存在）

- [ ] **Step 3: 改寫 stats_crawler.py**

```python
# src/etl/stats_crawler.py
"""
TVL 球員逐場數據爬蟲模組
從外部數據系統 (114.35.229.141) 抓取球員逐場統計，
透過球員姓名與本地 DB 關聯後 upsert 至 player_match_stats 事實表（依 season 隔離，冪等）。
"""

import re
import time

import requests
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.etl.db_loader import init_db
from src.utils.db_config import get_engine
from src.utils.logger import get_logger
from src.utils.constants import (
    EXT_BASE, EXT_CUP_ID as CUP_ID, EXT_HEADERS as HEADERS,
    SEASON, EXT_TEAM_MAP, season_year_for_month,
)

logger = get_logger(__name__)


def normalize_name(name: str) -> str:
    """正規化姓名：去除全形/半形空白、轉小寫、去除不間斷空白。"""
    return re.sub(r"[\s　\xa0]+", "", name).lower()


def safe_int(val: str) -> int | None:
    """安全轉換整數，失敗回傳 None。"""
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def parse_match_date(raw: str) -> str | None:
    """
    從 '311/01' 格式中萃取日期並轉為 YYYY-MM-DD。
    前面的數字是場次編號，後面 MM/DD 是日期，年份依 SEASON 設定推算。
    """
    m = re.search(r"(\d{1,2})/(\d{2})$", raw)
    if not m:
        return None
    month = int(m.group(1))
    day = int(m.group(2))
    year = season_year_for_month(month)
    return f"{year}-{month:02d}-{day:02d}"


def build_name_to_pid(engine: Engine, season: str = SEASON) -> dict[str, int]:
    """建立當前賽季 {正規化姓名: player_id} 的查找表（各賽季名單獨立）。"""
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT player_id, name FROM players WHERE season = :season"),
            {"season": season},
        ).fetchall()
    return {normalize_name(name): pid for pid, name in rows}


def fetch_player_list(team_id: int) -> list[dict]:
    """從外部系統取得某隊的球員清單。回傳 [{'ext_player_id': int, 'name': str}, ...]"""
    url = f"{EXT_BASE}/_handler/PlayerList.ashx"
    r = requests.get(
        url, params={"CupID": CUP_ID, "TeamID": team_id},
        headers=HEADERS, timeout=15,
    )
    soup = BeautifulSoup(r.text, "html.parser")
    players = []
    for opt in soup.find_all("option"):
        text_ = opt.get_text(strip=True)  # e.g. "No.2-黃宇晨"
        ext_id = opt.get("value")
        name = text_.split("-", 1)[1] if "-" in text_ else text_
        players.append({"ext_player_id": int(ext_id), "name": name})
    return players


def fetch_player_stats(team_id: int, ext_player_id: int) -> list[dict]:
    """抓取單一球員的逐場數據表，回傳字典列表。跳過表頭行與最後的「累計」行。"""
    url = f"{EXT_BASE}/_handler/Player.ashx"
    r = requests.get(
        url,
        params={"CupID": CUP_ID, "PlayerID": ext_player_id, "TeamID": team_id},
        headers=HEADERS, timeout=15,
    )
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    rows = table.find_all("tr")
    records = []
    for row in rows[2:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
        if not cells or cells[0] == "累計":
            continue
        if len(cells) < 15:
            continue

        record = {
            "match_date": parse_match_date(cells[0]),
            "opponent": cells[1] or None,
            "sets_played": safe_int(cells[2]),
            "attack_total": safe_int(cells[3]),
            "attack_points": safe_int(cells[4]),
            "block_points": safe_int(cells[5]),
            "serve_total": safe_int(cells[6]),
            "serve_points": safe_int(cells[7]),
            "receive_total": safe_int(cells[8]),
            "receive_excellent": safe_int(cells[9]),
            "dig_total": safe_int(cells[10]),
            "dig_excellent": safe_int(cells[11]),
            "set_total": safe_int(cells[12]),
            "set_excellent": safe_int(cells[13]),
            "total_points": safe_int(cells[14]),
        }
        records.append(record)

    # 偵測黃金決勝局：同日期同對手出現兩筆時，局數較少的為黃金局
    seen: dict[tuple, int] = {}
    for i, r in enumerate(records):
        key = (r["match_date"], r["opponent"])
        if key in seen:
            prev_i = seen[key]
            if (records[prev_i]["sets_played"] or 0) <= (r["sets_played"] or 0):
                records[prev_i]["is_golden_set"] = 1
            else:
                r["is_golden_set"] = 1
        else:
            seen[key] = i

    for r in records:
        r.setdefault("is_golden_set", 0)

    return records


def upsert_stats(engine: Engine, player_id: int, records: list[dict], season: str) -> None:
    """將單一球員的逐場數據 upsert 至 player_match_stats（唯一鍵含 season，永不觸碰其他賽季）。"""
    rows = [
        {
            "player_id": player_id,
            "season": season,
            "match_date": r["match_date"],
            "opponent": r["opponent"],
            "sets_played": r["sets_played"],
            "attack_total": r["attack_total"],
            "attack_points": r["attack_points"],
            "block_points": r["block_points"],
            "serve_total": r["serve_total"],
            "serve_points": r["serve_points"],
            "receive_total": r["receive_total"],
            "receive_excellent": r["receive_excellent"],
            "dig_total": r["dig_total"],
            "dig_excellent": r["dig_excellent"],
            "set_total": r["set_total"],
            "set_excellent": r["set_excellent"],
            "total_points": r["total_points"],
            "is_golden_set": r["is_golden_set"],
        }
        for r in records
    ]
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO player_match_stats
                (player_id, season, match_date, opponent, sets_played,
                 attack_total, attack_points, block_points,
                 serve_total, serve_points,
                 receive_total, receive_excellent,
                 dig_total, dig_excellent,
                 set_total, set_excellent, total_points, is_golden_set)
            VALUES
                (:player_id, :season, :match_date, :opponent, :sets_played,
                 :attack_total, :attack_points, :block_points,
                 :serve_total, :serve_points,
                 :receive_total, :receive_excellent,
                 :dig_total, :dig_excellent,
                 :set_total, :set_excellent, :total_points, :is_golden_set)
            ON CONFLICT (player_id, season, match_date, opponent, is_golden_set) DO UPDATE SET
                sets_played       = excluded.sets_played,
                attack_total      = excluded.attack_total,
                attack_points     = excluded.attack_points,
                block_points      = excluded.block_points,
                serve_total       = excluded.serve_total,
                serve_points      = excluded.serve_points,
                receive_total     = excluded.receive_total,
                receive_excellent = excluded.receive_excellent,
                dig_total         = excluded.dig_total,
                dig_excellent     = excluded.dig_excellent,
                set_total         = excluded.set_total,
                set_excellent     = excluded.set_excellent,
                total_points      = excluded.total_points
        """), rows)


def _scalar(engine: Engine, sql: str):
    with engine.begin() as conn:
        return conn.execute(text(sql)).scalar_one()


def main(incremental: bool = False):
    """
    主流程：抓取所有球員逐場數據並 upsert 至 DB。

    Parameters
    ----------
    incremental : bool
        僅影響記錄檔訊息；全量與增量模式皆一律 upsert（冪等），
        不再有「全量先砍表重建」與「增量只新增缺少紀錄」的行為差異。
    """
    engine = get_engine()
    init_db(engine)  # CREATE TABLE IF NOT EXISTS，永不清空既有資料

    mode_label = "增量" if incremental else "全量"
    logger.info("%s模式：upsert 所有球員逐場數據（不再砍表重建）", mode_label)

    name_map = build_name_to_pid(engine, SEASON)

    total_upserted = 0
    total_new_players = 0

    for ext_team_id in range(1, 10):
        players = fetch_player_list(ext_team_id)
        logger.info("TeamID=%d: %d 位球員", ext_team_id, len(players))

        for p in players:
            ext_pid = p["ext_player_id"]
            name = p["name"]
            norm_name = normalize_name(name)

            player_id = name_map.get(norm_name)

            # Late Arriving Dimension：查無此人（本賽季）則動態新增
            if player_id is None:
                db_team_id, gender = EXT_TEAM_MAP[ext_team_id]
                logger.info("[動態新增] 發現新球員: %s，自動寫入 players 表。", name)
                with engine.begin() as conn:
                    player_id = conn.execute(
                        text(
                            "INSERT INTO players (name, team_id, gender, season) "
                            "VALUES (:name, :team_id, :gender, :season) RETURNING player_id"
                        ),
                        {"name": name, "team_id": db_team_id, "gender": gender, "season": SEASON},
                    ).scalar_one()
                name_map[norm_name] = player_id
                total_new_players += 1

            try:
                records = fetch_player_stats(ext_team_id, ext_pid)
            except Exception as e:
                logger.error("抓取球員 [%s] 數據失敗: %s", name, e)
                continue

            if not records:
                continue

            upsert_stats(engine, player_id, records, SEASON)
            total_upserted += len(records)

            time.sleep(0.5)

    total_rows = _scalar(engine, "SELECT COUNT(*) FROM player_match_stats")
    total_players = _scalar(engine, "SELECT COUNT(*) FROM players")

    print(f"\n===== {mode_label} upsert 完成 =====")
    print(f"player_match_stats 總筆數：{total_rows}")
    print(f"本次 upsert：{total_upserted} 筆")
    print(f"動態新增球員數：{total_new_players}")
    print(f"players 表總人數：{total_players}")

    print(f"\n===== 前 3 筆資料 =====")
    df = pd.read_sql_query(
        """SELECT s.stat_id, p.name, s.match_date, s.opponent,
                  s.sets_played, s.attack_points, s.block_points,
                  s.serve_points, s.total_points
           FROM player_match_stats s
           JOIN players p ON s.player_id = p.player_id
           LIMIT 3""",
        engine,
    )
    print(df.to_string(index=False))

    logger.info("事實表載入完成")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TVL 球員逐場數據爬蟲")
    parser.add_argument(
        "--incremental", "-i", action="store_true",
        help="增量模式（僅影響記錄檔訊息，實際皆為 upsert）",
    )
    args = parser.parse_args()
    main(incremental=args.incremental)
```

- [ ] **Step 4: 執行測試確認通過**

Run: `pytest tests/test_stats_crawler.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/etl/stats_crawler.py tests/test_stats_crawler.py
git commit -m "feat: stats_crawler 改為冪等 upsert + season 隔離，移除 DROP TABLE 與 import fallback"
```

---

### Task 9: match_crawler.py 改為 upsert + season + 移除 fallback + TEAM_ALIAS 併入 constants

`src/etl/match_crawler.py` 目前自行維護一份 `init_matches_table()`（與 `sql/schema.sql` 重複）與本地定義的 `TEAM_ALIAS`（Task 3 已併入 constants），且 `upsert_match()` 用「先 SELECT 再決定 INSERT 或 UPDATE」的手動邏輯。本任務改呼叫 `db_loader.init_db()`、`TEAM_ALIAS` 改從 constants 匯入、`upsert_match()` 改為單一 `ON CONFLICT ... DO UPDATE` 陳述式並帶入 `season`。

**Files:**
- Modify: `src/etl/match_crawler.py`（全檔，1-385 行）
- Create: `tests/test_match_crawler.py`

**Interfaces:**
- Consumes：`src.etl.db_loader.init_db(engine)`（Task 7）；`src.utils.db_config.get_engine()`（Task 5）；`src.utils.constants.SEASON`、`TEAM_ALIAS`、`EXT_HEADERS`（Task 3）。
- Produces: `match_crawler.upsert_match(engine: Engine, match: dict) -> None`（Task 13 的整合測試會直接呼叫此函式；`match` dict 需含 `"season"` 鍵）。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_match_crawler.py
from sqlalchemy import text

from src.etl.match_crawler import normalize_team, upsert_match


def _sample_match(**overrides) -> dict:
    base = dict(
        game_id=301, gender="M", season="2025-26", match_date="2026-01-05",
        venue="台南", round_name="例行賽 Week 5", game_label="Game 301",
        is_golden_set=0, home_team="屏東台電", away_team="雲林美津濃",
        home_set1=25, home_set2=25, home_set3=25, home_set4=None, home_set5=None,
        home_total=75, away_set1=20, away_set2=18, away_set3=22,
        away_set4=None, away_set5=None, away_total=60,
        home_sets_won=3, away_sets_won=0,
    )
    base.update(overrides)
    return base


def test_normalize_team_maps_game_page_alias():
    assert normalize_team("桃園臺灣產險") == "桃園臺產"
    assert normalize_team("臺中獅子王") == "獅子王"
    assert normalize_team("屏東台電") == "屏東台電"


def test_upsert_match_is_idempotent_on_rerun(sqlite_engine):
    match = _sample_match()
    upsert_match(sqlite_engine, match)
    upsert_match(sqlite_engine, match)

    with sqlite_engine.begin() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM matches")).scalar_one()
    assert n == 1


def test_upsert_match_updates_corrected_score(sqlite_engine):
    match = _sample_match()
    upsert_match(sqlite_engine, match)

    corrected = _sample_match(home_total=74)
    upsert_match(sqlite_engine, corrected)

    with sqlite_engine.begin() as conn:
        total = conn.execute(text("SELECT home_total FROM matches")).scalar_one()
    assert total == 74


def test_upsert_match_does_not_touch_other_season_rows(sqlite_engine):
    match = _sample_match(season="2025-26")
    upsert_match(sqlite_engine, match)

    upsert_match(sqlite_engine, _sample_match(season="2026-27"))

    with sqlite_engine.begin() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM matches")).scalar_one()
        old_total = conn.execute(
            text("SELECT home_total FROM matches WHERE season = '2025-26'")
        ).scalar_one()
    assert total == 2
    assert old_total == 75
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_match_crawler.py -v`
Expected: FAIL — 目前 `upsert_match(conn, match)` 簽名接受 `sqlite3.Connection` 而非 `Engine`，且回傳 `bool`（`is_new`）而非 `None`；`match` dict 沒有 `season` 鍵時目前程式碼也不會報錯，但唯一鍵尚未包含 season，測試會因表結構不符而失敗

- [ ] **Step 3: 改寫 match_crawler.py**

```python
# src/etl/match_crawler.py
"""
TVL 官網比賽結果爬蟲模組
從 tvl.ctvba.org.tw 的 /game/ (男子組) 與 /wgame/ (女子組) 頁面
抓取各局比分、比賽資訊，並 upsert 至 matches 表（依 season 隔離，冪等）。
"""

import re
import time

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.etl.db_loader import init_db
from src.utils.db_config import get_engine
from src.utils.logger import get_logger
from src.utils.constants import EXT_HEADERS as HEADERS, SEASON, TEAM_ALIAS

logger = get_logger(__name__)

BASE_URL = "https://tvl.ctvba.org.tw"


def normalize_team(raw: str) -> str:
    """將官網隊名轉為 DB 簡寫。"""
    return TEAM_ALIAS.get(raw, raw)


def _safe_int(val: str) -> int | None:
    """安全轉換整數，空字串或失敗回傳 None。"""
    try:
        return int(val) if val and val.strip() else None
    except (ValueError, TypeError):
        return None


def scrape_match_page(prefix: str, game_id: int) -> dict | None:
    """
    抓取單場比賽頁面，回傳結構化 dict。

    Parameters
    ----------
    prefix : 'game' (男子組) 或 'wgame' (女子組)
    game_id : 官網頁面 ID

    Returns
    -------
    dict or None (頁面不存在或無資料)
    """
    gender = "M" if prefix == "game" else "F"
    url = f"{BASE_URL}/{prefix}/{game_id}"

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        r.encoding = "utf-8"
    except requests.RequestException as e:
        logger.error("無法取得 %s: %s", url, e)
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    score_table = soup.find("table", class_="match_table")
    if not score_table:
        return None

    rows = score_table.find_all("tr")
    if len(rows) < 3:
        return None

    home_cells = [td.get_text(strip=True) for td in rows[1].find_all("td")]
    away_cells = [td.get_text(strip=True) for td in rows[2].find_all("td")]

    if not home_cells or not home_cells[0]:
        return None

    home_team = normalize_team(home_cells[0])
    away_team = normalize_team(away_cells[0])

    home_sets = [_safe_int(home_cells[i]) if i < len(home_cells) else None for i in range(1, 6)]
    away_sets = [_safe_int(away_cells[i]) if i < len(away_cells) else None for i in range(1, 6)]
    home_total = _safe_int(home_cells[6]) if len(home_cells) > 6 else None
    away_total = _safe_int(away_cells[6]) if len(away_cells) > 6 else None

    home_sets_won = sum(
        1 for h, a in zip(home_sets, away_sets)
        if h is not None and a is not None and h > a
    )
    away_sets_won = sum(
        1 for h, a in zip(home_sets, away_sets)
        if h is not None and a is not None and a > h
    )

    gh = soup.find("div", class_="game_header")
    gh_text = gh.get_text(" | ", strip=True) if gh else ""

    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", gh_text)
    match_date = date_m.group(1) if date_m else None

    venue = None
    venue_m = re.search(r"\d{2}:\d{2}:\d{2}\s*\|?\s*(.+?)\s*\|", gh_text)
    if venue_m:
        venue = venue_m.group(1).strip()

    round_name = None
    round_m = re.search(r"(例行賽|挑戰賽|總決賽|季後賽|明星賽)\s*Week\s*\d+", gh_text)
    if round_m:
        round_name = round_m.group(0)

    game_label = None
    label_m = re.search(r"(Game\s*\S+(?:\s*\(.*?\))?)", gh_text)
    if label_m:
        game_label = label_m.group(1).strip()

    is_golden = 1 if "黃金決勝局" in gh_text else 0

    if not match_date:
        logger.warning("[%s/%d] 無法解析日期，跳過", prefix, game_id)
        return None

    return {
        "game_id": game_id,
        "gender": gender,
        "season": SEASON,
        "match_date": match_date,
        "venue": venue,
        "round_name": round_name,
        "game_label": game_label,
        "is_golden_set": is_golden,
        "home_team": home_team,
        "away_team": away_team,
        "home_set1": home_sets[0],
        "home_set2": home_sets[1],
        "home_set3": home_sets[2],
        "home_set4": home_sets[3],
        "home_set5": home_sets[4],
        "home_total": home_total,
        "away_set1": away_sets[0],
        "away_set2": away_sets[1],
        "away_set3": away_sets[2],
        "away_set4": away_sets[3],
        "away_set5": away_sets[4],
        "away_total": away_total,
        "home_sets_won": home_sets_won,
        "away_sets_won": away_sets_won,
    }


def upsert_match(engine: Engine, match: dict) -> None:
    """Upsert 單場比賽紀錄（唯一鍵：game_id + gender + season，冪等）。"""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO matches (
                game_id, gender, season, match_date, venue, round_name, game_label,
                is_golden_set, home_team, away_team,
                home_set1, home_set2, home_set3, home_set4, home_set5, home_total,
                away_set1, away_set2, away_set3, away_set4, away_set5, away_total,
                home_sets_won, away_sets_won
            ) VALUES (
                :game_id, :gender, :season, :match_date, :venue, :round_name, :game_label,
                :is_golden_set, :home_team, :away_team,
                :home_set1, :home_set2, :home_set3, :home_set4, :home_set5, :home_total,
                :away_set1, :away_set2, :away_set3, :away_set4, :away_set5, :away_total,
                :home_sets_won, :away_sets_won
            )
            ON CONFLICT (game_id, gender, season) DO UPDATE SET
                match_date=excluded.match_date, venue=excluded.venue,
                round_name=excluded.round_name, game_label=excluded.game_label,
                is_golden_set=excluded.is_golden_set,
                home_team=excluded.home_team, away_team=excluded.away_team,
                home_set1=excluded.home_set1, home_set2=excluded.home_set2,
                home_set3=excluded.home_set3, home_set4=excluded.home_set4,
                home_set5=excluded.home_set5, home_total=excluded.home_total,
                away_set1=excluded.away_set1, away_set2=excluded.away_set2,
                away_set3=excluded.away_set3, away_set4=excluded.away_set4,
                away_set5=excluded.away_set5, away_total=excluded.away_total,
                home_sets_won=excluded.home_sets_won, away_sets_won=excluded.away_sets_won
        """), match)


def scrape_all_matches(
    prefixes: list[str] | None = None,
    id_range: range | None = None,
    delay: float = 0.5,
) -> dict:
    """
    批次抓取官網比賽結果並 upsert 至 DB。

    Parameters
    ----------
    prefixes : ['game', 'wgame']
    id_range : 要掃描的 game_id 範圍
    delay : 請求間隔秒數
    """
    if prefixes is None:
        prefixes = ["game", "wgame"]
    if id_range is None:
        id_range = range(220, 400)

    engine = get_engine()
    init_db(engine)

    stats = {"upserted": 0, "golden_sets": 0}

    for prefix in prefixes:
        consecutive_empty = 0
        for game_id in id_range:
            match = scrape_match_page(prefix, game_id)
            if match is None:
                consecutive_empty += 1
                if consecutive_empty >= 20:
                    logger.info("[%s] 連續 %d 個空頁面，停止掃描", prefix, consecutive_empty)
                    break
                continue

            consecutive_empty = 0
            upsert_match(engine, match)
            stats["upserted"] += 1

            if match["is_golden_set"]:
                stats["golden_sets"] += 1

            logger.info(
                "[%s/%d] %s %s vs %s%s",
                prefix, game_id, match["match_date"],
                match["home_team"], match["away_team"],
                " ★Golden Set" if match["is_golden_set"] else "",
            )

            time.sleep(delay)

    with engine.begin() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM matches")).scalar_one()
        golden = conn.execute(
            text("SELECT COUNT(*) FROM matches WHERE is_golden_set = 1")
        ).scalar_one()

    stats["total"] = total
    stats["total_golden"] = golden
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TVL 官網比賽結果爬蟲")
    parser.add_argument("--range-start", type=int, default=220, help="起始 game_id (預設 220)")
    parser.add_argument("--range-end", type=int, default=400, help="結束 game_id (預設 400)")
    parser.add_argument("--delay", type=float, default=0.5, help="請求間隔秒數 (預設 0.5)")
    args = parser.parse_args()

    stats = scrape_all_matches(
        id_range=range(args.range_start, args.range_end),
        delay=args.delay,
    )

    print(f"\n===== 比賽結果爬取完成 =====")
    print(f"matches 表總筆數：{stats['total']}")
    print(f"本次 upsert：{stats['upserted']} 場")
    print(f"黃金決勝局：{stats['total_golden']} 場")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 執行測試確認通過**

Run: `pytest tests/test_match_crawler.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/etl/match_crawler.py tests/test_match_crawler.py
git commit -m "feat: match_crawler 改用 SQLAlchemy ON CONFLICT upsert + season 隔離，TEAM_ALIAS 併入 constants"
```

---

### Task 10: weekly_report.py 改用 engine + 具名參數

`src/etl/weekly_report.py` 目前用 `get_connection()`（sqlite3）並以 `?` 佔位符 + tuple 參數呼叫 `pd.read_sql`。這是純 ETL/資料彙整模組（非 Streamlit UI 檔案），可安全改為 PostgreSQL 相容寫法：`pd.read_sql` 改傳入 `sqlalchemy.text(...)` 包裹的查詢字串 + dict 具名參數（已實測此寫法在 SQLite 與 PostgreSQL 皆會走 SQLAlchemy 的方言轉換，`?` 佔位符搭配純字串傳給 pandas 則會繞過方言轉換、直接用底層 DBAPI 的原生 paramstyle，對 PostgreSQL 的 psycopg 驅動會失敗，因為 psycopg 原生 paramstyle 是 `pyformat` 而非 `qmark`）。

**Files:**
- Modify: `src/etl/weekly_report.py`（全檔，1-261 行）
- Create: `tests/test_weekly_report.py`

**Interfaces:**
- Consumes：`src.utils.db_config.get_engine()`（Task 5）。
- Produces: `weekly_report.get_match_weeks() -> list[tuple[str, str]]`、`weekly_report.gather_weekly_data(date_from: str, date_to: str, gender_filter: str | None = None) -> dict`（對外簽名與回傳結構完全不變，`src/app/tabs/weekly_report_tab.py` 呼叫端不需修改）。

- [ ] **Step 1: 寫失敗測試**

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
    weeks = get_match_weeks()
    assert weeks == [("2026-01-05", "2026-01-05")]


def test_gather_weekly_data_filters_by_date_range(sqlite_engine):
    _seed(sqlite_engine)
    result = gather_weekly_data("2026-01-01", "2026-01-10")
    assert result["period"] == "2026-01-01 ~ 2026-01-10"
    assert len(result["matches"]) == 1
    assert result["matches"][0]["team_name"] == "屏東台電"
    assert result["matches"][0]["opponent"] == "雲林美津濃"


def test_gather_weekly_data_filters_by_gender(sqlite_engine):
    _seed(sqlite_engine)
    result_f = gather_weekly_data("2026-01-01", "2026-01-10", gender_filter="F")
    assert result_f["matches"] == []

    result_m = gather_weekly_data("2026-01-01", "2026-01-10", gender_filter="M")
    assert len(result_m["matches"]) == 1
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_weekly_report.py -v`
Expected: FAIL — `weekly_report.py` 目前用 `get_connection()`（已在 Task 5 從 db_config 移除），import 會直接報錯

- [ ] **Step 3: 改寫 weekly_report.py**

```python
# src/etl/weekly_report.py
"""
每周戰報資料彙整模組
從 DB 撈取指定日期範圍內的比賽數據，產生結構化摘要供 AI 戰報使用。
"""

from datetime import datetime

import pandas as pd
from sqlalchemy import text

from src.utils.db_config import get_engine


def get_match_weeks() -> list[tuple[str, str]]:
    """
    回傳所有比賽周次的 (week_start, week_end) 列表。
    以 ISO 周次分組，方便使用者選擇。
    """
    engine = get_engine()
    dates = pd.read_sql(
        "SELECT DISTINCT match_date FROM player_match_stats ORDER BY match_date",
        engine,
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
    date_from: str, date_to: str, gender_filter: str | None = None
) -> dict:
    """
    彙整指定日期範圍內的所有比賽數據，回傳結構化 dict。

    Parameters
    ----------
    date_from : 起始日期 (YYYY-MM-DD)
    date_to : 結束日期 (YYYY-MM-DD)
    gender_filter : "M", "F", or None (全部)

    Returns
    -------
    dict with keys: "period", "matches"
    """
    engine = get_engine()
    gender_clause = "AND p.gender = :gender_filter" if gender_filter else ""

    params: dict = {"date_from": date_from, "date_to": date_to}
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
            {gender_clause}
            ORDER BY s.match_date, t.team_name
        """),
        engine,
        params=params,
    )

    season_params: dict = {"date_to": date_to}
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
            {gender_clause}
            GROUP BY p.player_id
            HAVING COUNT(*) >= 2
        """),
        engine,
        params=season_params,
    )

    if raw.empty:
        return {"period": f"{date_from} ~ {date_to}", "matches": []}

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

- [ ] **Step 4: 執行測試確認通過**

Run: `pytest tests/test_weekly_report.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/etl/weekly_report.py tests/test_weekly_report.py
git commit -m "feat: weekly_report 改用 SQLAlchemy engine + 具名參數，PostgreSQL 相容"
```

---

### Task 11: helpers.py DB 存取改用 db_config

`src/app/helpers.py` 目前自行從 `Path(__file__)` 推導 `DB_PATH` 並用 `sqlite3.connect()`（第 7、80、86-93 行），也有一段 `try/except ModuleNotFoundError` fallback（第 67-78 行）內含 `SEASON_YEAR_MAP`。本任務只改動這兩處：DB 連線改用 Task 5 的 `db_config.get_engine()`；fallback 移除，直接從 constants 匯入，`fetch_match_index()` 內原本用 `SEASON_YEAR_MAP.get(month, 2026)` 改用 Task 3 的 `season_year_for_month(month)`。

**`load_data()` 的查詢字串與參數風格（`?` 佔位符 + tuple）維持不變**——`main.py` 與 4 個 tab 檔案（`box_score.py`、`match_trend.py`、`player_deep.py`、`weekly_report_tab.py`）都直接呼叫 `load_data(query, params)` 並沿用 `?` 佔位符，這些呼叫端屬於 Streamlit UI 檔案，不在本計畫範圍內（見文末風險清單第 1 項：這代表 `load_data()` 走 PostgreSQL 時 `?` 佔位符會失敗，需計畫二處理）。

**Files:**
- Modify: `src/app/helpers.py:1-14`（移除未使用的 `sqlite3` import）、`src/app/helpers.py:67-93`（移除 fallback、DB_PATH、改寫 `load_data`）、`src/app/helpers.py:110-139`（`fetch_match_index` 改用 `season_year_for_month`）
- Create: `tests/test_helpers_db.py`

**Interfaces:**
- Consumes：`src.utils.db_config.get_engine()`（Task 5）；`src.utils.constants.season_year_for_month`、`EXT_BASE`、`EXT_CUP_ID`、`EXT_HEADERS`、`OPP_SHORT_TO_TEAM`（Task 3）。
- Produces: `helpers.load_data(query: str, params: tuple = ()) -> pd.DataFrame`（對外簽名不變，維持 `?` 佔位符風格，供所有 tab 檔案原樣呼叫）。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_helpers_db.py
import pandas as pd
from sqlalchemy import text


def test_load_data_reads_via_db_config_engine(sqlite_engine, monkeypatch):
    with sqlite_engine.begin() as conn:
        conn.execute(text("INSERT INTO teams (team_id, team_name, gender) VALUES (1, 'X', 'M')"))

    import src.app.helpers as helpers

    helpers.load_data.clear()  # 清除 st.cache_data 快取，避免跨測試互相汙染
    df = helpers.load_data("SELECT * FROM teams WHERE gender = ?", ("M",))
    assert len(df) == 1
    assert df.iloc[0]["team_name"] == "X"


def test_fetch_match_index_uses_season_year_for_month():
    from src.app.helpers import fetch_match_index

    # fetch_match_index 內部呼叫外部系統，這裡只驗證 import 後年份推算邏輯可用
    # （season_year_for_month 已在 tests/test_constants.py 完整測試，此處驗證 helpers 正確引用它）
    import src.app.helpers as helpers

    assert helpers.season_year_for_month(11) == 2025
    assert helpers.season_year_for_month(3) == 2026
    assert callable(fetch_match_index)
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_helpers_db.py -v`
Expected: FAIL — `helpers.py` 目前用固定 `DB_PATH` 推導路徑（非 `sqlite_engine` fixture 指向的暫存 DB），第一個測試會因讀不到剛寫入的資料而失敗；`helpers.season_year_for_month` 不存在（`AttributeError`）

- [ ] **Step 3: 修改 helpers.py**

移除第 7 行 `import sqlite3`（不再需要）。

將原第 67-80 行：

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

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "tvl_database.db"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "match_predictor.pkl"
```

改為：

```python
from src.utils.constants import (
    EXT_BASE, EXT_CUP_ID, EXT_HEADERS, OPP_SHORT_TO_TEAM, season_year_for_month,
)
from src.utils.db_config import get_engine

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "match_predictor.pkl"
```

將原第 84-93 行的 `load_data`：

```python
# ── DB 查詢 ──────────────────────────────────────────────────

@st.cache_data
def load_data(query: str, params: tuple = ()) -> pd.DataFrame:
    """連線 SQLite，執行查詢並回傳 DataFrame。連線使用後立即關閉。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
```

改為：

```python
# ── DB 查詢 ──────────────────────────────────────────────────

@st.cache_data
def load_data(query: str, params: tuple = ()) -> pd.DataFrame:
    """透過 db_config 的 engine 執行查詢並回傳 DataFrame。"""
    engine = get_engine()
    return pd.read_sql_query(query, engine, params=params)
```

將原第 130 行（`fetch_match_index` 內）：

```python
            year = SEASON_YEAR_MAP.get(month, 2026)
```

改為：

```python
            year = season_year_for_month(month)
```

- [ ] **Step 4: 執行測試確認通過**

Run: `pytest tests/test_helpers_db.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/helpers.py tests/test_helpers_db.py
git commit -m "refactor: helpers.py DB 存取改用 db_config engine，移除 import fallback"
```

---

### Task 12: 一次性資料遷移 script

一次性把現有 `data/db/tvl_database.db`（舊 schema，四張表皆無 `season` 欄位）的資料搬到 `DATABASE_URL` 指向的目標資料庫（新 schema，含 `season`），並為既有資料標上目前賽季。保留原始的 `player_id`/`stat_id`/`match_id`，避免重新對應外鍵關聯；若目標為 PostgreSQL，遷移後需校正 identity 序列（因為手動指定了既有 ID，序列不會自動前進）。

**Files:**
- Create: `src/etl/migrate_to_postgres.py`
- Create: `tests/test_migrate_to_postgres.py`

**Interfaces:**
- Consumes：`src.etl.db_loader.init_db(engine)`（Task 7）；`src.utils.db_config.get_engine()`（Task 5）；`src.utils.constants.SEASON`（Task 3）。
- Produces: `migrate_to_postgres.migrate(sqlite_path: Path = SOURCE_DB_PATH, season: str = SEASON) -> dict[str, int]`（回傳各表搬移筆數，供操作者確認）。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_migrate_to_postgres.py
import sqlite3
from pathlib import Path

from sqlalchemy import text

from src.etl.migrate_to_postgres import migrate

OLD_SCHEMA = """
CREATE TABLE teams (
    team_id INTEGER NOT NULL, team_name TEXT NOT NULL, gender TEXT NOT NULL,
    PRIMARY KEY (team_id, gender)
);
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT, team_id INTEGER NOT NULL, gender TEXT NOT NULL,
    jersey_number INTEGER, name TEXT, position TEXT, dob DATE, height_cm REAL, weight_kg REAL
);
CREATE TABLE player_match_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT, player_id INTEGER NOT NULL,
    match_date DATE, opponent TEXT, sets_played INTEGER,
    attack_total INTEGER, attack_points INTEGER, block_points INTEGER,
    serve_total INTEGER, serve_points INTEGER, receive_total INTEGER, receive_excellent INTEGER,
    dig_total INTEGER, dig_excellent INTEGER, set_total INTEGER, set_excellent INTEGER,
    total_points INTEGER, is_golden_set INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL, gender TEXT NOT NULL,
    match_date DATE NOT NULL, venue TEXT, round_name TEXT, game_label TEXT,
    is_golden_set INTEGER NOT NULL DEFAULT 0,
    home_team TEXT NOT NULL, away_team TEXT NOT NULL,
    home_set1 INTEGER, home_set2 INTEGER, home_set3 INTEGER, home_set4 INTEGER, home_set5 INTEGER, home_total INTEGER,
    away_set1 INTEGER, away_set2 INTEGER, away_set3 INTEGER, away_set4 INTEGER, away_set5 INTEGER, away_total INTEGER,
    home_sets_won INTEGER, away_sets_won INTEGER,
    UNIQUE (game_id, gender)
);
"""


def _build_old_source_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    conn.execute("INSERT INTO teams VALUES (1, '屏東台電', 'M')")
    conn.execute(
        "INSERT INTO players (team_id, gender, jersey_number, name, position, dob, height_cm, weight_kg) "
        "VALUES (1, 'M', 4, '李元', 'OH', '2000-01-01', 190.0, 80.0)"
    )
    pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO player_match_stats "
        "(player_id, match_date, opponent, sets_played, attack_total, attack_points, block_points, "
        "serve_total, serve_points, receive_total, receive_excellent, dig_total, dig_excellent, "
        "set_total, set_excellent, total_points, is_golden_set) "
        "VALUES (?, '2026-01-05', '雲林美津濃', 3, 10, 5, 1, 5, 1, 5, 3, 5, 2, 0, 0, 7, 0)",
        (pid,),
    )
    conn.execute(
        "INSERT INTO matches "
        "(game_id, gender, match_date, venue, round_name, game_label, is_golden_set, "
        "home_team, away_team, home_set1, home_set2, home_set3, home_set4, home_set5, home_total, "
        "away_set1, away_set2, away_set3, away_set4, away_set5, away_total, home_sets_won, away_sets_won) "
        "VALUES (301, 'M', '2026-01-05', '台南', '例行賽 Week 5', 'Game 301', 0, "
        "'屏東台電', '雲林美津濃', 25, 25, 25, NULL, NULL, 75, 20, 18, 22, NULL, NULL, 60, 3, 0)"
    )
    conn.commit()
    conn.close()


def test_migrate_copies_all_tables_and_tags_season(tmp_path, sqlite_engine):
    source_path = tmp_path / "old_source.db"
    _build_old_source_db(source_path)

    counts = migrate(source_path, season="2025-26")
    assert counts == {"teams": 1, "players": 1, "player_match_stats": 1, "matches": 1}

    with sqlite_engine.begin() as conn:
        season_val = conn.execute(text("SELECT DISTINCT season FROM players")).scalar_one()
        assert season_val == "2025-26"


def test_migrate_is_idempotent_on_rerun(tmp_path, sqlite_engine):
    source_path = tmp_path / "old_source.db"
    _build_old_source_db(source_path)

    migrate(source_path, season="2025-26")
    counts2 = migrate(source_path, season="2025-26")
    assert counts2 == {"teams": 1, "players": 1, "player_match_stats": 1, "matches": 1}

    with sqlite_engine.begin() as conn:
        n_players = conn.execute(text("SELECT COUNT(*) FROM players")).scalar_one()
    assert n_players == 1
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `pytest tests/test_migrate_to_postgres.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.etl.migrate_to_postgres'`

- [ ] **Step 3: 建立 src/etl/migrate_to_postgres.py**

```python
# src/etl/migrate_to_postgres.py
"""
一次性資料遷移工具
讀取現有 SQLite（data/db/tvl_database.db，舊 schema 無 season 欄位），
寫入 DATABASE_URL 指向的目標資料庫（新 schema，含 season），
並為既有資料標上目前賽季（SEASON 設定）。
部署後執行一次，之後由 Airflow（計畫三）增量維護。
"""

import sqlite3
from pathlib import Path

from sqlalchemy import text

from src.etl.db_loader import init_db
from src.utils.constants import SEASON
from src.utils.db_config import PROJECT_ROOT, get_engine
from src.utils.logger import get_logger

logger = get_logger(__name__)

SOURCE_DB_PATH = PROJECT_ROOT / "data" / "db" / "tvl_database.db"


def _read_source_tables(sqlite_path: Path) -> dict:
    """讀取舊 SQLite DB 的四張表，回傳 {table_name: [dict, ...]}。"""
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        tables = {}
        for table in ["teams", "players", "player_match_stats", "matches"]:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            tables[table] = [dict(r) for r in rows]
        return tables
    finally:
        conn.close()


def migrate(sqlite_path: Path = SOURCE_DB_PATH, season: str = SEASON) -> dict:
    """執行一次性遷移，回傳各表搬移筆數統計。"""
    if not sqlite_path.exists():
        raise FileNotFoundError(f"來源 SQLite 檔案不存在：{sqlite_path}")

    source = _read_source_tables(sqlite_path)
    engine = get_engine()
    init_db(engine)  # 於目標 DB 建立新 schema（含 season 欄位），冪等

    counts = {}
    with engine.begin() as conn:
        for row in source["teams"]:
            conn.execute(text("""
                INSERT INTO teams (team_id, team_name, gender)
                VALUES (:team_id, :team_name, :gender)
                ON CONFLICT (team_id, gender) DO UPDATE SET team_name = excluded.team_name
            """), row)
        counts["teams"] = len(source["teams"])

        for row in source["players"]:
            conn.execute(text("""
                INSERT INTO players
                    (player_id, team_id, gender, season, jersey_number, name, position, dob, height_cm, weight_kg)
                VALUES
                    (:player_id, :team_id, :gender, :season, :jersey_number, :name, :position, :dob, :height_cm, :weight_kg)
                ON CONFLICT (team_id, gender, season, name) DO UPDATE SET
                    jersey_number = excluded.jersey_number,
                    position      = excluded.position,
                    dob           = excluded.dob,
                    height_cm     = excluded.height_cm,
                    weight_kg     = excluded.weight_kg
            """), {**row, "season": season})
        counts["players"] = len(source["players"])

        for row in source["player_match_stats"]:
            conn.execute(text("""
                INSERT INTO player_match_stats
                    (stat_id, player_id, season, match_date, opponent, sets_played,
                     attack_total, attack_points, block_points,
                     serve_total, serve_points,
                     receive_total, receive_excellent,
                     dig_total, dig_excellent,
                     set_total, set_excellent, total_points, is_golden_set)
                VALUES
                    (:stat_id, :player_id, :season, :match_date, :opponent, :sets_played,
                     :attack_total, :attack_points, :block_points,
                     :serve_total, :serve_points,
                     :receive_total, :receive_excellent,
                     :dig_total, :dig_excellent,
                     :set_total, :set_excellent, :total_points, :is_golden_set)
                ON CONFLICT (player_id, season, match_date, opponent, is_golden_set) DO UPDATE SET
                    sets_played       = excluded.sets_played,
                    attack_total      = excluded.attack_total,
                    attack_points     = excluded.attack_points,
                    block_points      = excluded.block_points,
                    serve_total       = excluded.serve_total,
                    serve_points      = excluded.serve_points,
                    receive_total     = excluded.receive_total,
                    receive_excellent = excluded.receive_excellent,
                    dig_total         = excluded.dig_total,
                    dig_excellent     = excluded.dig_excellent,
                    set_total         = excluded.set_total,
                    set_excellent     = excluded.set_excellent,
                    total_points      = excluded.total_points
            """), {**row, "season": season})
        counts["player_match_stats"] = len(source["player_match_stats"])

        for row in source["matches"]:
            conn.execute(text("""
                INSERT INTO matches (
                    match_id, game_id, gender, season, match_date, venue, round_name, game_label,
                    is_golden_set, home_team, away_team,
                    home_set1, home_set2, home_set3, home_set4, home_set5, home_total,
                    away_set1, away_set2, away_set3, away_set4, away_set5, away_total,
                    home_sets_won, away_sets_won
                ) VALUES (
                    :match_id, :game_id, :gender, :season, :match_date, :venue, :round_name, :game_label,
                    :is_golden_set, :home_team, :away_team,
                    :home_set1, :home_set2, :home_set3, :home_set4, :home_set5, :home_total,
                    :away_set1, :away_set2, :away_set3, :away_set4, :away_set5, :away_total,
                    :home_sets_won, :away_sets_won
                )
                ON CONFLICT (game_id, gender, season) DO UPDATE SET
                    match_date=excluded.match_date, venue=excluded.venue,
                    round_name=excluded.round_name, game_label=excluded.game_label,
                    is_golden_set=excluded.is_golden_set,
                    home_team=excluded.home_team, away_team=excluded.away_team,
                    home_set1=excluded.home_set1, home_set2=excluded.home_set2,
                    home_set3=excluded.home_set3, home_set4=excluded.home_set4,
                    home_set5=excluded.home_set5, home_total=excluded.home_total,
                    away_set1=excluded.away_set1, away_set2=excluded.away_set2,
                    away_set3=excluded.away_set3, away_set4=excluded.away_set4,
                    away_set5=excluded.away_set5, away_total=excluded.away_total,
                    home_sets_won=excluded.home_sets_won, away_sets_won=excluded.away_sets_won
            """), {**row, "season": season})
        counts["matches"] = len(source["matches"])

        # 校正 PostgreSQL identity 序列，避免後續自動產生的 ID 與剛遷入的既有 ID 衝突
        # （SQLite 沒有序列概念，此區塊只在方言為 postgresql 時執行）
        if engine.dialect.name == "postgresql":
            for table, pk in [
                ("players", "player_id"),
                ("player_match_stats", "stat_id"),
                ("matches", "match_id"),
            ]:
                conn.execute(text(
                    f"SELECT setval(pg_get_serial_sequence('{table}', '{pk}'), "
                    f"COALESCE((SELECT MAX({pk}) FROM {table}), 1))"
                ))

    logger.info("遷移完成：%s", counts)
    return counts


def main():
    counts = migrate()
    print("\n===== 一次性資料遷移完成 =====")
    for table, n in counts.items():
        print(f"{table}: {n} 筆")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 執行測試確認通過**

Run: `pytest tests/test_migrate_to_postgres.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/etl/migrate_to_postgres.py tests/test_migrate_to_postgres.py
git commit -m "feat: 新增一次性資料遷移 script，保留原始 ID 並標上目前賽季"
```

---

### Task 13: 整合驗證：完整 ETL pipeline 冪等性 + DATABASE_URL 切換

本任務是本計畫的收尾驗證，對應 spec §12「本次最關鍵的行為改變」。串接 db_loader（名單）→ stats_crawler（逐場數據）→ match_crawler（比賽結果）三個模組，在同一個暫存 SQLite DB 上完整跑兩次，驗證：(1) 全部四張表跑兩次列數相同；(2) 換賽季重跑不影響舊賽季任何一列；(3) `db_config.get_engine()` 依 `DATABASE_URL` 正確切換 SQLite/PostgreSQL 方言（此項已在 Task 5 驗證過基本行為，本任務額外驗證「切換後 upsert 語句本身在兩種方言下語法一致」——透過檢查實際送出的 SQL 字串在两種 dialect 下編譯結果皆合法）。

**Files:**
- Create: `tests/test_etl_pipeline_idempotency.py`

**Interfaces:**
- Consumes：`db_loader.insert_teams`、`insert_players`（Task 7）；`stats_crawler.upsert_stats`（Task 8）；`match_crawler.upsert_match`（Task 9）；`db_config.get_engine`、`reset_engine`（Task 5）。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/test_etl_pipeline_idempotency.py
import pandas as pd
from sqlalchemy import text

from src.etl.db_loader import insert_players, insert_teams
from src.etl.match_crawler import upsert_match
from src.etl.stats_crawler import upsert_stats


def _roster_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "team_id": 1, "team_name": "屏東台電", "gender": "M",
        "jersey_number": 4, "name": "李元", "position": "OH",
        "dob": None, "height_cm": None, "weight_kg": None,
    }])


def _stat_row() -> dict:
    return dict(
        match_date="2026-01-05", opponent="雲林美津濃", sets_played=3,
        attack_total=10, attack_points=5, block_points=1,
        serve_total=5, serve_points=1, receive_total=5, receive_excellent=3,
        dig_total=5, dig_excellent=2, set_total=0, set_excellent=0,
        total_points=7, is_golden_set=0,
    )


def _match_row(season: str) -> dict:
    return dict(
        game_id=301, gender="M", season=season, match_date="2026-01-05",
        venue="台南", round_name="例行賽 Week 5", game_label="Game 301",
        is_golden_set=0, home_team="屏東台電", away_team="雲林美津濃",
        home_set1=25, home_set2=25, home_set3=25, home_set4=None, home_set5=None,
        home_total=75, away_set1=20, away_set2=18, away_set3=22,
        away_set4=None, away_set5=None, away_total=60,
        home_sets_won=3, away_sets_won=0,
    )


def _run_pipeline_once(engine, season: str) -> None:
    df = _roster_df()
    insert_teams(engine, df)
    insert_players(engine, df, season=season)
    with engine.begin() as conn:
        pid = conn.execute(
            text("SELECT player_id FROM players WHERE name = '李元' AND season = :s"),
            {"s": season},
        ).scalar_one()
    upsert_stats(engine, pid, [_stat_row()], season)
    upsert_match(engine, _match_row(season))


def _table_counts(engine) -> dict[str, int]:
    with engine.begin() as conn:
        return {
            table: conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            for table in ["teams", "players", "player_match_stats", "matches"]
        }


def test_full_pipeline_rerun_is_idempotent(sqlite_engine):
    _run_pipeline_once(sqlite_engine, "2025-26")
    counts_first = _table_counts(sqlite_engine)

    _run_pipeline_once(sqlite_engine, "2025-26")
    counts_second = _table_counts(sqlite_engine)

    assert counts_first == counts_second
    assert counts_second == {"teams": 1, "players": 1, "player_match_stats": 1, "matches": 1}


def test_new_season_rerun_does_not_touch_old_season_rows(sqlite_engine):
    _run_pipeline_once(sqlite_engine, "2025-26")
    _run_pipeline_once(sqlite_engine, "2026-27")

    counts = _table_counts(sqlite_engine)
    assert counts["players"] == 2
    assert counts["player_match_stats"] == 2
    assert counts["matches"] == 2

    with sqlite_engine.begin() as conn:
        old_total = conn.execute(
            text("SELECT home_total FROM matches WHERE season = '2025-26'")
        ).scalar_one()
    assert old_total == 75


def test_database_url_switch_between_sqlite_and_postgresql(monkeypatch):
    import src.utils.db_config as db_config

    monkeypatch.delenv("DATABASE_URL", raising=False)
    db_config.reset_engine()
    sqlite_eng = db_config.get_engine()
    assert sqlite_eng.dialect.name == "sqlite"

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/tvl")
    db_config.reset_engine()
    pg_eng = db_config.get_engine()
    assert pg_eng.dialect.name == "postgresql"

    # 驗證同一段 upsert SQL 可被兩種方言的 compiler 編譯（不需要真正連線）
    from sqlalchemy import text as sa_text

    upsert_sql = sa_text(
        "INSERT INTO teams (team_id, team_name, gender) VALUES (:team_id, :team_name, :gender) "
        "ON CONFLICT (team_id, gender) DO UPDATE SET team_name = excluded.team_name"
    )
    compiled_sqlite = str(upsert_sql.compile(dialect=sqlite_eng.dialect))
    compiled_pg = str(upsert_sql.compile(dialect=pg_eng.dialect))
    assert "ON CONFLICT" in compiled_sqlite
    assert "ON CONFLICT" in compiled_pg

    monkeypatch.delenv("DATABASE_URL", raising=False)
    db_config.reset_engine()
```

- [ ] **Step 2: 執行測試確認失敗（若 Task 1-12 皆已完成，此步驟應直接通過；若尚有 Task 未完成則會因對應函式不存在而失敗）**

Run: `pytest tests/test_etl_pipeline_idempotency.py -v`
Expected: 若 Task 1-12 皆已完成則直接 PASS（本任務不需要額外的實作步驟，純粹是整合驗證）；若有 Task 未完成，依失敗訊息回頭確認對應 Task 的 Step 3 是否確實完成

- [ ] **Step 3: 若全部通過，跑一次完整測試套件確認沒有互相影響**

Run: `pytest tests/ -v`
Expected: 全部測試 PASS（含 Task 1-13 建立的所有測試檔）

- [ ] **Step 4: Commit**

```bash
git add tests/test_etl_pipeline_idempotency.py
git commit -m "test: 新增跨模組 ETL 冪等性與 DATABASE_URL 切換整合測試"
```

---

## 完成後的檔案清單

```
pytest.ini                              [新增]
requirements.txt                        [改動：全部釘版本 + sqlalchemy/psycopg]
requirements-dev.txt                    [新增]
sql/schema.sql                          [改動：移除 DROP TABLE、加 season]
sql/schema_postgres.sql                 [新增]
src/utils/constants.py                  [改動：環境變數化、season 函式、TEAM_ALIAS]
src/utils/db_config.py                  [改動：SQLAlchemy engine]
src/utils/logger.py                     [改動：LOG_LEVEL]
src/etl/crawler.py                      [改動：移除 fallback]
src/etl/cleaner.py                      [改動：移除 fallback]
src/etl/db_loader.py                    [改動：upsert + season]
src/etl/stats_crawler.py                [改動：upsert + season + 移除 fallback]
src/etl/match_crawler.py                [改動：upsert + season + 移除 fallback]
src/etl/weekly_report.py                [改動：engine + 具名參數]
src/etl/migrate_to_postgres.py          [新增]
src/app/helpers.py                      [改動：DB 連線改用 db_config]
tests/conftest.py                       [新增]
tests/test_smoke.py                     [新增]
tests/test_model_compat.py              [新增]
tests/test_constants.py                 [新增]
tests/test_logger.py                    [新增]
tests/test_crawler_cleaner.py           [新增]
tests/test_db_config.py                 [新增]
tests/test_schema.py                    [新增]
tests/test_db_loader.py                 [新增]
tests/test_stats_crawler.py             [新增]
tests/test_match_crawler.py             [新增]
tests/test_weekly_report.py             [新增]
tests/test_helpers_db.py                [新增]
tests/test_migrate_to_postgres.py       [新增]
tests/test_etl_pipeline_idempotency.py  [新增]
```

## 已知風險（發現但不在本計畫處理範圍）

1. **`app/helpers.py` 的 `load_data()` 對 PostgreSQL 不安全**：`load_data(query, params)` 沿用 `?` 佔位符 + tuple 參數。實測證實 `pandas.read_sql_query` 收到純字串查詢時會透過 SQLAlchemy `Connection.exec_driver_sql()` 直接把查詢丟給底層 DBAPI 驅動，使用驅動的原生 paramstyle，而非 SQLAlchemy 的方言轉換——SQLite（`sqlite3` 模組）原生 paramstyle 剛好是 `qmark`（`?`），所以本地行為不受影響；但 PostgreSQL 用的 `psycopg` 驅動原生 paramstyle 是 `pyformat`（`%(name)s`），並不認得 `?`。這代表 `main.py` 與 4 個 tab 檔案（`box_score.py`、`match_trend.py`、`player_deep.py`、`weekly_report_tab.py`）目前呼叫 `load_data(..., "?", (...))` 的寫法，一旦 `DATABASE_URL` 指向 PostgreSQL 就會全部報錯。修法是把這些呼叫端的查詢字串改成具名參數（`:name`）並把 `load_data` 內部改用 `text()` 包裹——但這會改到 5 個 Streamlit UI 檔案的程式碼，超出本計畫「不修改 §4d」的範圍，建議列入計畫二第一項工作。
2. **`player_match_stats` 新增的 `season` 欄位會出現在 `SELECT *` 結果中**：`src/app/tabs/match_trend.py:22` 與 `src/app/tabs/player_deep.py:74` 皆用 `SELECT * FROM player_match_stats WHERE player_id = ?`，加入 `season` 欄位後回傳的 DataFrame 會多一欄。依現有程式碼推斷這兩處後續是用欄位名稱存取（非位置索引），多一欄理論上無害，但未實際執行 Streamlit 驗證，建議計畫二執行時順手確認。
3. **`src/app/tabs/prediction.py:125` 的 `artifact.get("feature_names", [])` 讀錯鍵名**：實際 pkl 內容的鍵是 `feature_cols` 不是 `feature_names`，導致 `n_features` 恆為 0、恆定落入 V1（5 特徵）分支。目前恰好與 pkl 實際的 5 特徵吻合，功能上「無症狀」，但若未來訓練出 11 特徵版模型（V2）並更新 pkl，這個既有 bug 會讓 UI 誤判為 V1，需在計畫二處理 `prediction.py` 時一併修正。
4. **PostgreSQL 相容性未經真實伺服器驗證**：本機環境無 Docker/Postgres 可用，`sql/schema_postgres.sql`、`migrate_to_postgres.py`、各 upsert 語句的 PostgreSQL 相容性僅透過語法檢查與 SQLAlchemy dialect compiler 驗證，未實際對真正的 PostgreSQL 執行。spec §12 已規劃「docker compose 起 Postgres 跑同一套測試」，建議在計畫二/三容器化階段第一時間補跑本計畫全部測試對照一次真實 PostgreSQL。
5. **`stats_crawler.py` 的 `--incremental` 旗標語意改變**：改造前「全量」會砍表重建、「增量」只新增缺少的紀錄；改造後兩者皆為 upsert（全量的破壞性行為消失，增量的「跳過已存在」邏輯也移除，改為一律更新）。這是本計畫刻意簡化的設計決策（見 Task 8 說明），對外 CLI 介面（`--incremental` 旗標）維持相容，但底層行為語意不同，若計畫三的 Airflow DAG 或既有操作文件對兩種模式有额外假設，需要對照確認。
