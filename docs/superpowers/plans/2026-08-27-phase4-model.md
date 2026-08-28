# Phase 4 模型優化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 勝負標籤改用 matches 表真實局數、訓練流程收進 `src/models/` script、pkl 檔名版本化 + app fail-loud 載入、notebook 收斂至一本 EDA。

**Architecture:** 新增 `src/models/features.py`（純 sqlite3+pandas，不依賴 streamlit，可被測試直接 import）負責球隊層級聚合、真實標籤、滾動特徵；`src/models/train.py` 負責 TimeSeriesSplit + Optuna + XGBoost 訓練與版本化 artifact 匯出；`src/app/tabs/prediction.py` 改為依 artifact 的 `feature_cols` 順序驅動滑杆與特徵向量，版本不認得即停。

**Tech Stack:** Python 3 / sqlite3 / pandas / scikit-learn / XGBoost / Optuna / joblib / pytest

**Spec:** `docs/superpowers/specs/2026-08-27-phase4-model-design.md`

## Global Constraints

- 行尾一律 LF；commit 前 `git diff --stat -w` 確認無行尾雜訊（repo 在 /mnt/d，Windows 工具會製造 CRLF）。
- UI 文案、註解、commit message 用繁體中文；不加不必要的註解與 docstring。
- import 走 `from src....` 絕對路徑；從 repo 根目錄執行；測試用 `python -m pytest`。
- DB 連線：正式程式用 `src.utils.db_config.get_connection()`；測試一律用 `tmp_db_path` fixture 建臨時 DB（`sql/schema.sql` 建表），**不觸碰正式 DB**（除 Task 7 的實跑重訓與驗證，唯讀 + 產 pkl）。
- `requirements.txt` 全部 `==` 精確釘版（`tests/test_requirements_pinned.py` 會驗）；新增依賴以實際安裝版本釘版，不憑記憶填版本號。
- `.db` 與 `.pkl` 刻意 commit 進 git（Streamlit Cloud 依賴），不要加進 `.gitignore`。
- 資料品質原則：只標記與警告，不插補不竄改。
- 每個 task 結尾 commit（本計畫經使用者核准執行即為授權）。
- 已知事實（探索實查）：matches 170 筆（2 筆 golden set）、隊名 14 種（含 `屏東台電男排`/`高雄台電女排` 後綴變體與 `彰化三大有線`/`臺北Conti`/`連莊` 三支 2024 賽季舊隊）；`player_match_stats.opponent` 只有 9 種簡稱 = `OPP_SHORT_TO_TEAM` 鍵集；proxy label 與真實局數在 13/170 場不一致。

---

### Task 1: 隊名正規化常數與函式

**Files:**
- Modify: `src/utils/constants.py`（`TEAM_NAME_SHORT` 區塊後新增兩個常數）
- Create: `src/models/__init__.py`（空檔）
- Create: `src/models/features.py`
- Test: `tests/test_features_labels.py`

**Interfaces:**
- Produces: `constants.MATCH_TEAM_ALIASES: dict[str, str]`、`constants.LEGACY_TEAMS: set[str]`、`features.normalize_match_team(name: str) -> str`（回傳簡稱；未知隊名 raise `ValueError`）。

- [ ] **Step 1: 寫失敗測試**

`tests/test_features_labels.py`：

```python
import pytest

from src.models.features import normalize_match_team
from src.utils.constants import LEGACY_TEAMS, OPP_SHORT_TO_TEAM

# matches 表實際出現過的 14 種隊名（2026-08-27 實查）
MATCH_TEAM_NAMES = [
    "屏東台電", "屏東台電男排", "彰化三大有線", "新北中纖", "桃園臺產",
    "獅子王", "義力營造", "臺北Conti", "臺北國北獅", "臺北鯨華",
    "連莊", "雲林美津濃", "高雄台電", "高雄台電女排",
]


def test_all_match_team_names_normalize():
    for name in MATCH_TEAM_NAMES:
        short = normalize_match_team(name)
        assert short in OPP_SHORT_TO_TEAM or short in LEGACY_TEAMS


def test_suffix_variants_map_to_short():
    assert normalize_match_team("屏東台電男排") == "屏東台電"
    assert normalize_match_team("高雄台電女排") == "高雄台電"


def test_unknown_team_raises():
    with pytest.raises(ValueError, match="未知隊名"):
        normalize_match_team("不存在的隊")
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python -m pytest tests/test_features_labels.py -v`
Expected: FAIL（`ModuleNotFoundError` 或 `ImportError`）

- [ ] **Step 3: 實作**

`src/utils/constants.py` 在 `TEAM_NAME_SHORT` 區塊後新增：

```python
# ── matches 表隊名別名（官網比分頁的後綴變體）→ 簡稱 ──────────
MATCH_TEAM_ALIASES = {
    "屏東台電男排": "屏東台電",
    "高雄台電女排": "高雄台電",
}

# 2024 賽季舊隊：比分留存於 matches，但已不在 teams 表、無逐場統計
LEGACY_TEAMS = {"彰化三大有線", "臺北Conti", "連莊"}
```

`src/models/__init__.py`：空檔。

`src/models/features.py`：

```python
"""
球隊層級特徵與真實勝負標籤（Phase 4）。
不依賴 streamlit，訓練 script 與測試皆直接 import。
"""

from src.utils.constants import LEGACY_TEAMS, MATCH_TEAM_ALIASES, OPP_SHORT_TO_TEAM


def normalize_match_team(name: str) -> str:
    short = MATCH_TEAM_ALIASES.get(name, name)
    if short in OPP_SHORT_TO_TEAM or short in LEGACY_TEAMS:
        return short
    raise ValueError(f"未知隊名：{name}（請補 constants.MATCH_TEAM_ALIASES）")
```

- [ ] **Step 4: 跑測試確認通過**

Run: `python -m pytest tests/test_features_labels.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/constants.py src/models/__init__.py src/models/features.py tests/test_features_labels.py
git commit -m "feat: 隊名正規化——matches 別名映射與 2024 舊隊清單"
```

---

### Task 2: 球隊層級單場聚合（濾 golden set）

**Files:**
- Modify: `src/models/features.py`
- Test: `tests/test_features_labels.py`

**Interfaces:**
- Consumes: `db_config.get_connection()` 型的 `sqlite3.Connection`。
- Produces: `features.GAME_STAT_COLS = ["ASR", "GP_pct", "DIG_pct", "BLK_per_set", "ACE_pct"]`；`features.load_team_match_stats(conn) -> pd.DataFrame`，欄位含 `match_date, team_id, gender, opponent, total_points` 與五指標，每列 = 一隊一場（golden set 已排除）。

- [ ] **Step 1: 寫失敗測試**

追加到 `tests/test_features_labels.py`（模組層加 helper 與 import）：

```python
import sqlite3
from pathlib import Path

import pandas as pd

from src.models.features import GAME_STAT_COLS, load_team_match_stats

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


def _make_db(tmp_db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def _seed_team(conn, team_id, name, gender):
    conn.execute("INSERT INTO teams VALUES (?, ?, ?)", (team_id, name, gender))


def _seed_stat(conn, team_id, gender, match_date, opponent, *,
               atk_pts=10, atk_tot=20, total_points=25, golden=0):
    cur = conn.execute(
        "INSERT INTO players (name, gender) VALUES (?, ?)", (f"p{team_id}{match_date}{golden}", gender))
    pid = cur.lastrowid
    cur = conn.execute(
        """INSERT INTO roster_registrations
           (player_id, team_id, gender, cup_id, week_label, source)
           VALUES (?, ?, ?, 21, ?, 'match_page')""",
        (pid, team_id, gender, f"w{match_date}"))
    rid = cur.lastrowid
    conn.execute(
        """INSERT INTO player_match_stats
           (registration_id, match_date, opponent, sets_played,
            attack_total, attack_points, block_points, serve_total, serve_points,
            receive_total, receive_excellent, dig_total, dig_excellent,
            set_total, set_excellent, total_points, is_golden_set)
           VALUES (?, ?, ?, 3, ?, ?, 2, 10, 1, 10, 5, 10, 3, 10, 5, ?, ?)""",
        (rid, match_date, opponent, atk_tot, atk_pts, total_points, golden))


def test_team_match_aggregation_excludes_golden_set(tmp_db_path):
    conn = _make_db(tmp_db_path)
    _seed_team(conn, 1, "屏東台電", "M")
    _seed_stat(conn, 1, "M", "2026-01-10", "獅子王")
    _seed_stat(conn, 1, "M", "2026-01-10", "獅子王", golden=1)
    df = load_team_match_stats(conn)
    conn.close()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["team_id"] == 1 and row["gender"] == "M"
    assert row["ASR"] == pytest.approx(50.0)  # 10/20*100
    for col in GAME_STAT_COLS:
        assert col in df.columns
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python -m pytest tests/test_features_labels.py -v`
Expected: 新測試 FAIL（`ImportError: load_team_match_stats`）

- [ ] **Step 3: 實作**

`src/models/features.py` 追加：

```python
import pandas as pd

GAME_STAT_COLS = ["ASR", "GP_pct", "DIG_pct", "BLK_per_set", "ACE_pct"]

_TEAM_MATCH_SQL = """
    SELECT s.match_date,
           r.team_id,
           r.gender,
           s.opponent,
           SUM(s.attack_points)     AS atk_pts,
           SUM(s.attack_total)      AS atk_tot,
           SUM(s.block_points)      AS blk_pts,
           SUM(s.serve_points)      AS srv_pts,
           SUM(s.serve_total)       AS srv_tot,
           SUM(s.receive_excellent) AS rcv_exc,
           SUM(s.receive_total)     AS rcv_tot,
           SUM(s.dig_excellent)     AS dig_exc,
           SUM(s.dig_total)         AS dig_tot,
           SUM(s.total_points)      AS total_points,
           MAX(s.sets_played)       AS total_sets
    FROM player_match_stats s
    JOIN roster_registrations r ON s.registration_id = r.registration_id
    WHERE s.is_golden_set = 0
    GROUP BY s.match_date, r.team_id, r.gender, s.opponent
"""


def _pct(num, den):
    return (num / den * 100).where(den > 0, 0.0)


def load_team_match_stats(conn) -> pd.DataFrame:
    df = pd.read_sql_query(_TEAM_MATCH_SQL, conn)
    df["ASR"] = _pct(df["atk_pts"], df["atk_tot"])
    df["GP_pct"] = _pct(df["rcv_exc"], df["rcv_tot"])
    df["DIG_pct"] = _pct(df["dig_exc"], df["dig_tot"])
    df["ACE_pct"] = _pct(df["srv_pts"], df["srv_tot"])
    df["BLK_per_set"] = (df["blk_pts"] / df["total_sets"]).where(df["total_sets"] > 0, 0.0)
    return df
```

- [ ] **Step 4: 跑測試確認通過**

Run: `python -m pytest tests/test_features_labels.py -v`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/features.py tests/test_features_labels.py
git commit -m "feat: 球隊層級單場聚合走 registration 路徑並排除 golden set"
```

---

### Task 3: 真實局數標籤（fail loud）

**Files:**
- Modify: `src/models/features.py`
- Test: `tests/test_features_labels.py`

**Interfaces:**
- Produces: `features.build_match_labels(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict]`（labels 欄位 `match_date, team_id, gender, win`；report 含 `legacy_skipped`/`invalid_skipped` 計數）；`features.attach_labels(team_match, labels) -> pd.DataFrame`（left merge，統計面找不到比分即 raise）。matches DataFrame 欄位依 `sql/schema.sql` 的 matches 表。

- [ ] **Step 1: 寫失敗測試**

追加：

```python
from src.models.features import attach_labels, build_match_labels


def _match_row(home, away, gender, hsw, asw, *, date="2026-01-10",
               h_total=75, a_total=70, golden=0):
    return {
        "match_date": date, "gender": gender, "is_golden_set": golden,
        "home_team": home, "away_team": away,
        "home_total": h_total, "away_total": a_total,
        "home_sets_won": hsw, "away_sets_won": asw,
    }


def test_label_uses_sets_won_not_total_points():
    # 總得分較高但局數落敗（proxy 會標錯的 13 場型態）
    matches = pd.DataFrame([_match_row(
        "屏東台電", "獅子王", "M", 2, 3, h_total=110, a_total=105)])
    labels, _ = build_match_labels(matches)
    home = labels[labels["team_id"] == 1].iloc[0]
    away = labels[labels["team_id"] == 7].iloc[0]
    assert home["win"] == 0 and away["win"] == 1


def test_labels_skip_golden_set_and_legacy():
    matches = pd.DataFrame([
        _match_row("屏東台電", "獅子王", "M", 3, 1),
        _match_row("屏東台電", "獅子王", "M", 1, 0, golden=1, date="2026-01-11"),
        _match_row("臺北Conti", "連莊", "M", 3, 2, date="2024-12-01"),
    ])
    labels, report = build_match_labels(matches)
    assert len(labels) == 2  # 只有第一場的主客兩筆
    assert report["legacy_skipped"] == 1


def test_labels_unknown_team_raises():
    matches = pd.DataFrame([_match_row("怪隊", "獅子王", "M", 3, 0)])
    with pytest.raises(ValueError, match="未知隊名"):
        build_match_labels(matches)


def test_attach_labels_fails_loud_on_missing_score(tmp_db_path):
    conn = _make_db(tmp_db_path)
    _seed_team(conn, 1, "屏東台電", "M")
    _seed_stat(conn, 1, "M", "2026-01-10", "獅子王")
    team_match = load_team_match_stats(conn)
    conn.close()
    empty_labels = pd.DataFrame(columns=["match_date", "team_id", "gender", "win"])
    with pytest.raises(ValueError, match="找不到對應比分"):
        attach_labels(team_match, empty_labels)
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python -m pytest tests/test_features_labels.py -v`
Expected: 新 4 測試 FAIL（ImportError）

- [ ] **Step 3: 實作**

`src/models/features.py` 追加：

```python
def build_match_labels(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"legacy_skipped": 0, "invalid_skipped": 0}
    rows = []
    for _, m in matches.iterrows():
        if m["is_golden_set"] == 1:
            continue
        home = normalize_match_team(m["home_team"])
        away = normalize_match_team(m["away_team"])
        if home in LEGACY_TEAMS or away in LEGACY_TEAMS:
            report["legacy_skipped"] += 1
            continue
        if (pd.isna(m["home_sets_won"]) or pd.isna(m["away_sets_won"])
                or m["home_sets_won"] == m["away_sets_won"]):
            report["invalid_skipped"] += 1
            continue
        home_tid, home_g = OPP_SHORT_TO_TEAM[home]
        away_tid, away_g = OPP_SHORT_TO_TEAM[away]
        if home_g != m["gender"] or away_g != m["gender"]:
            raise ValueError(
                f"隊伍性別對不上 matches.gender：{m['home_team']} vs {m['away_team']}（{m['gender']}）")
        home_win = int(m["home_sets_won"] > m["away_sets_won"])
        rows.append((m["match_date"], home_tid, home_g, home_win))
        rows.append((m["match_date"], away_tid, away_g, 1 - home_win))
    labels = pd.DataFrame(rows, columns=["match_date", "team_id", "gender", "win"])
    dup = labels.duplicated(["match_date", "team_id", "gender"], keep=False)
    if dup.any():
        raise ValueError(f"同日同隊出現多筆標籤：\n{labels[dup].to_string(index=False)}")
    return labels, report


def attach_labels(team_match: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = team_match.merge(labels, on=["match_date", "team_id", "gender"], how="left")
    missing = merged[merged["win"].isna()]
    if not missing.empty:
        detail = missing[["match_date", "team_id", "gender", "opponent"]].to_string(index=False)
        raise ValueError(f"{len(missing)} 筆球隊單場統計找不到對應比分：\n{detail}")
    merged["win"] = merged["win"].astype(int)
    return merged
```

- [ ] **Step 4: 跑測試確認通過**

Run: `python -m pytest tests/test_features_labels.py -v`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/features.py tests/test_features_labels.py
git commit -m "feat: 勝負標籤改用 matches 真實局數，未匹配即報錯"
```

---

### Task 4: 滾動特徵與 win_streak

**Files:**
- Modify: `src/models/features.py`
- Test: `tests/test_features_rolling.py`

**Interfaces:**
- Produces: `features.ROLLING_FEATURES`（11 個，順序 = 5 個 `_roll3` + 5 個 `_roll5` + `win_streak`）、`features.compute_win_streak(wins: pd.Series) -> list[int]`、`features.add_rolling_features(labeled: pd.DataFrame) -> pd.DataFrame`（丟棄各隊首場）、`features.build_training_frame(conn) -> tuple[pd.DataFrame, dict]`（一條龍：聚合 → 標籤 → 滾動特徵）。

- [ ] **Step 1: 寫失敗測試**

`tests/test_features_rolling.py`：

```python
import pandas as pd
import pytest

from src.models.features import (
    GAME_STAT_COLS, ROLLING_FEATURES, add_rolling_features, compute_win_streak,
)


def _labeled_frame():
    rows = []
    for i, (asr, win) in enumerate([(40.0, 1), (50.0, 1), (60.0, 0), (30.0, 1)]):
        row = {"match_date": f"2026-01-{10 + i:02d}", "team_id": 1, "gender": "M",
               "opponent": "獅子王", "win": win,
               "ASR": asr, "GP_pct": 50.0, "DIG_pct": 30.0,
               "BLK_per_set": 1.5, "ACE_pct": 5.0}
        rows.append(row)
    return pd.DataFrame(rows)


def test_rolling_features_order_and_count():
    assert len(ROLLING_FEATURES) == 11
    assert ROLLING_FEATURES[:5] == [f"{c}_roll3" for c in GAME_STAT_COLS]
    assert ROLLING_FEATURES[5:10] == [f"{c}_roll5" for c in GAME_STAT_COLS]
    assert ROLLING_FEATURES[10] == "win_streak"


def test_first_match_dropped_and_shift_excludes_current():
    df = add_rolling_features(_labeled_frame())
    assert len(df) == 3  # 首場無歷史被丟
    # 第二場的 roll3 只含第一場的 ASR=40，不含當場 50
    assert df.iloc[0]["ASR_roll3"] == pytest.approx(40.0)
    # 第四場 roll3 = mean(40, 50, 60)
    assert df.iloc[2]["ASR_roll3"] == pytest.approx(50.0)


def test_win_streak_is_pregame_state():
    wins = pd.Series([1, 1, 0, 0, 1])
    assert compute_win_streak(wins) == [0, 1, 2, -1, -2]


def test_win_streak_from_real_labels():
    df = add_rolling_features(_labeled_frame())
    # 賽前狀態：第二場前連勝1、第三場前連勝2、第四場前連敗1
    assert df["win_streak"].tolist() == [1, 2, -1]
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python -m pytest tests/test_features_rolling.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 實作**

`src/models/features.py` 追加：

```python
ROLLING_FEATURES = (
    [f"{c}_roll3" for c in GAME_STAT_COLS]
    + [f"{c}_roll5" for c in GAME_STAT_COLS]
    + ["win_streak"]
)

_GROUP_KEYS = ["team_id", "gender"]


def compute_win_streak(wins: pd.Series) -> list[int]:
    streaks, current = [], 0
    for w in wins:
        streaks.append(current)
        if w == 1:
            current = current + 1 if current > 0 else 1
        else:
            current = current - 1 if current < 0 else -1
    return streaks


def add_rolling_features(labeled: pd.DataFrame) -> pd.DataFrame:
    df = (labeled.sort_values(_GROUP_KEYS + ["match_date", "opponent"])
          .reset_index(drop=True))
    gkey = df[_GROUP_KEYS].apply(tuple, axis=1)
    for col in GAME_STAT_COLS:
        shifted = df.groupby(_GROUP_KEYS)[col].shift(1)
        df[f"{col}_roll3"] = shifted.groupby(gkey).transform(
            lambda x: x.rolling(3, min_periods=1).mean())
        df[f"{col}_roll5"] = shifted.groupby(gkey).transform(
            lambda x: x.rolling(5, min_periods=1).mean())
    df["win_streak"] = df.groupby(_GROUP_KEYS)["win"].transform(compute_win_streak)
    return (df.dropna(subset=[f"{GAME_STAT_COLS[0]}_roll3"])
            .reset_index(drop=True))


def build_training_frame(conn) -> tuple[pd.DataFrame, dict]:
    team_match = load_team_match_stats(conn)
    matches = pd.read_sql_query("SELECT * FROM matches", conn)
    labels, report = build_match_labels(matches)
    labeled = attach_labels(team_match, labels)
    report["team_match_rows"] = len(labeled)
    frame = add_rolling_features(labeled)
    report["training_rows"] = len(frame)
    return frame, report
```

- [ ] **Step 4: 跑測試確認通過**

Run: `python -m pytest tests/test_features_rolling.py tests/test_features_labels.py -v`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/features.py tests/test_features_rolling.py
git commit -m "feat: 滾動特徵與 win_streak 改由真實標籤計算"
```

---

### Task 5: 訓練 script（train.py + optuna 釘版）

**Files:**
- Create: `src/models/train.py`
- Modify: `requirements.txt`（加 optuna）
- Test: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `features.build_training_frame`、`features.ROLLING_FEATURES`、`features.GAME_STAT_COLS`。
- Produces: `train.tune_and_train(frame, n_trials, seed=42) -> dict`（回傳 artifact dict，尚未落地）；`train.main(argv=None)`（CLI：`--trials`（預設 50）、`--output`（預設 `src/models/match_predictor_v2.pkl`））。artifact 必含 key：`model, model_name, version, feature_cols, feature_labels, label_source, best_params, optuna_best_f1, cv_f1_mean, training_samples, trained_at, xgboost_version, game_stat_cols`，其中 `version == "v2"`、`label_source == "matches.sets_won"`。

- [ ] **Step 1: 安裝並釘版 optuna**

```bash
pip install optuna
pip show optuna | head -2
```

把實際顯示的版本以 `optuna==<實際版本>` 加入 `requirements.txt`（字母序位置）。**不憑記憶填版本號。**

Run: `python -m pytest tests/test_requirements_pinned.py -v`
Expected: PASS

- [ ] **Step 2: 寫失敗測試**

`tests/test_train_smoke.py`：

```python
import numpy as np
import pandas as pd

from src.models.features import GAME_STAT_COLS, ROLLING_FEATURES
from src.models.train import tune_and_train

REQUIRED_KEYS = {
    "model", "model_name", "version", "feature_cols", "feature_labels",
    "label_source", "best_params", "optuna_best_f1", "cv_f1_mean",
    "training_samples", "trained_at", "xgboost_version", "game_stat_cols",
}


def _fake_frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.uniform(20, 60, n) for f in ROLLING_FEATURES})
    df["win_streak"] = rng.integers(-3, 4, n)
    df["match_date"] = pd.date_range("2026-01-01", periods=n).astype(str)
    df["team_id"] = rng.integers(1, 5, n)
    df["win"] = rng.integers(0, 2, n)
    return df


def test_tune_and_train_artifact_contract():
    artifact = tune_and_train(_fake_frame(), n_trials=2)
    assert REQUIRED_KEYS <= set(artifact)
    assert artifact["version"] == "v2"
    assert artifact["label_source"] == "matches.sets_won"
    assert artifact["feature_cols"] == ROLLING_FEATURES
    assert artifact["game_stat_cols"] == GAME_STAT_COLS
    assert artifact["training_samples"] == 40
    proba = artifact["model"].predict_proba(
        _fake_frame(seed=1)[ROLLING_FEATURES].values[:3])
    assert proba.shape == (3, 2)
```

- [ ] **Step 3: 跑測試確認失敗**

Run: `python -m pytest tests/test_train_smoke.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 4: 實作**

`src/models/train.py`：

```python
"""
賽果預測模型訓練（v2）：TimeSeriesSplit + Optuna + XGBoost。
用法：python -m src.models.train [--trials 50] [--output src/models/match_predictor_v2.pkl]
"""

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

from src.models.features import GAME_STAT_COLS, ROLLING_FEATURES, build_training_frame
from src.utils.db_config import get_connection

SEED = 42
N_SPLITS = 5

ROLLING_LABELS = {
    "ASR_roll3": "近3場 攻擊率", "ASR_roll5": "近5場 攻擊率",
    "GP_pct_roll3": "近3場 接發率", "GP_pct_roll5": "近5場 接發率",
    "DIG_pct_roll3": "近3場 防守率", "DIG_pct_roll5": "近5場 防守率",
    "BLK_per_set_roll3": "近3場 局均攔網", "BLK_per_set_roll5": "近5場 局均攔網",
    "ACE_pct_roll3": "近3場 發球率", "ACE_pct_roll5": "近5場 發球率",
    "win_streak": "連勝/連敗",
}


def tune_and_train(frame, n_trials, seed=SEED) -> dict:
    df_ts = frame.sort_values(["match_date", "team_id"]).reset_index(drop=True)
    X_all = df_ts[ROLLING_FEATURES].values
    y_all = df_ts["win"].values
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": seed,
            "eval_metric": "logloss",
        }
        model = xgb.XGBClassifier(**params)
        scores = []
        for train_idx, test_idx in tscv.split(X_all):
            model.fit(X_all[train_idx], y_all[train_idx])
            scores.append(f1_score(y_all[test_idx], model.predict(X_all[test_idx]),
                                   zero_division=0))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best_params = {**study.best_params, "random_state": seed, "eval_metric": "logloss"}
    cv_scores = []
    model = xgb.XGBClassifier(**best_params)
    for train_idx, test_idx in tscv.split(X_all):
        model.fit(X_all[train_idx], y_all[train_idx])
        cv_scores.append(f1_score(y_all[test_idx], model.predict(X_all[test_idx]),
                                  zero_division=0))
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_all, y_all)

    return {
        "model": final_model,
        "model_name": "XGBoost (Optuna-tuned, v2)",
        "version": "v2",
        "feature_cols": ROLLING_FEATURES,
        "feature_labels": [ROLLING_LABELS[c] for c in ROLLING_FEATURES],
        "label_source": "matches.sets_won",
        "best_params": best_params,
        "optuna_best_f1": float(study.best_value),
        "cv_f1_mean": float(np.mean(cv_scores)),
        "training_samples": len(df_ts),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "xgboost_version": xgb.__version__,
        "game_stat_cols": GAME_STAT_COLS,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="重訓賽果預測模型（v2）")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--output", type=Path,
                        default=Path("src/models/match_predictor_v2.pkl"))
    args = parser.parse_args(argv)

    conn = get_connection()
    try:
        frame, report = build_training_frame(conn)
    finally:
        conn.close()
    print(f"資料報告：{report}")

    artifact = tune_and_train(frame, n_trials=args.trials)
    joblib.dump(artifact, args.output)
    print(f"樣本數：{artifact['training_samples']}｜"
          f"Optuna 最佳 F1：{artifact['optuna_best_f1']:.3f}｜"
          f"最佳參數重驗 cv F1：{artifact['cv_f1_mean']:.3f}")
    print(f"已輸出：{args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 跑測試確認通過**

Run: `python -m pytest tests/test_train_smoke.py -v`
Expected: PASS（2 trials，應在一分鐘內）

- [ ] **Step 6: Commit**

```bash
git add src/models/train.py requirements.txt tests/test_train_smoke.py
git commit -m "feat: v2 訓練流程收進 script——TimeSeriesSplit + Optuna 一鍵重訓"
```

---

### Task 6: app 端——fail-loud 載入、artifact 順序驅動滑杆、球隊層級範圍

**Files:**
- Modify: `src/app/helpers.py`（`MODEL_PATH` 一行）
- Modify: `src/app/tabs/prediction.py`（大改）
- Test: `tests/test_prediction_artifact.py`

**Interfaces:**
- Consumes: `features.GAME_STAT_COLS`、`features.load_team_match_stats`、Task 5 的 artifact schema。
- Produces: `helpers.MODEL_PATH` 指向 `src/models/match_predictor_v2.pkl`；`prediction.SLIDER_CFG: dict[str, tuple]`（key = 11 個特徵名，value = `(label, min, max, default, step)`）；`prediction.KNOWN_VERSIONS = {"v2"}`；`prediction._artifact_error(artifact) -> str | None`（純函式，回傳錯誤訊息或 None）。

- [ ] **Step 1: 寫失敗測試**

`tests/test_prediction_artifact.py`：

```python
from src.app.tabs.prediction import KNOWN_VERSIONS, SLIDER_CFG, _artifact_error
from src.models.features import ROLLING_FEATURES


def test_slider_cfg_covers_rolling_features_exactly():
    assert set(SLIDER_CFG) == set(ROLLING_FEATURES)


def test_artifact_error_accepts_v2():
    artifact = {"version": "v2", "feature_cols": ROLLING_FEATURES}
    assert _artifact_error(artifact) is None


def test_artifact_error_rejects_unknown_version():
    assert _artifact_error({"version": None, "feature_cols": []}) is not None
    assert _artifact_error({"version": "v1", "feature_cols": ROLLING_FEATURES}) is not None


def test_artifact_error_rejects_feature_mismatch():
    artifact = {"version": "v2", "feature_cols": ROLLING_FEATURES[:5]}
    assert _artifact_error(artifact) is not None
    artifact = {"version": "v2", "feature_cols": ROLLING_FEATURES + ["extra"]}
    assert _artifact_error(artifact) is not None
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python -m pytest tests/test_prediction_artifact.py -v`
Expected: FAIL（ImportError：無 `SLIDER_CFG`/`_artifact_error`）

- [ ] **Step 3: 實作**

`src/app/helpers.py:73` 改為：

```python
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "match_predictor_v2.pkl"
```

`src/app/tabs/prediction.py` 改寫（保留 SHAP 段與結果顯示段不動）：

1. 刪除 `V1_SLIDER_CFG` 與 `V2_SLIDER_CFG`，改為單一 dict（順序無所謂，組裝順序以 artifact 為準）：

```python
KNOWN_VERSIONS = {"v2"}

SLIDER_CFG = {
    "ASR_roll3":         ("近3場 攻擊率 (%)",   25.0, 65.0, 42.0, 0.5),
    "ASR_roll5":         ("近5場 攻擊率 (%)",   25.0, 65.0, 42.0, 0.5),
    "GP_pct_roll3":      ("近3場 接發率 (%)",   15.0, 85.0, 50.0, 0.5),
    "GP_pct_roll5":      ("近5場 接發率 (%)",   15.0, 85.0, 50.0, 0.5),
    "DIG_pct_roll3":     ("近3場 防守率 (%)",    5.0, 75.0, 32.0, 0.5),
    "DIG_pct_roll5":     ("近5場 防守率 (%)",    5.0, 75.0, 32.0, 0.5),
    "BLK_per_set_roll3": ("近3場 局均攔網",      0.0,  5.0,  1.8, 0.1),
    "BLK_per_set_roll5": ("近5場 局均攔網",      0.0,  5.0,  1.8, 0.1),
    "ACE_pct_roll3":     ("近3場 發球率 (%)",    0.0, 18.0,  4.0, 0.5),
    "ACE_pct_roll5":     ("近5場 發球率 (%)",    0.0, 18.0,  4.0, 0.5),
    "win_streak":        ("連勝/連敗 (正=連勝)", -8.0,  8.0,  0.0, 1.0),
}


def _artifact_error(artifact) -> str | None:
    version = artifact.get("version")
    if version not in KNOWN_VERSIONS:
        return f"未知的模型版本：{version!r}，請以 python -m src.models.train 重訓"
    feature_cols = artifact.get("feature_cols", [])
    if set(feature_cols) != set(SLIDER_CFG) or len(feature_cols) != len(SLIDER_CFG):
        return f"模型特徵與滑桿設定不一致：{sorted(set(feature_cols) ^ set(SLIDER_CFG))}"
    return None
```

2. `_get_data_ranges` 改為球隊層級（取代整個函式；`FEAT_LABELS_MAP` 不再被用到就刪）：

```python
@st.cache_data
def _get_data_ranges(gender_code: str) -> dict[str, tuple[float, float]]:
    """球隊層級單場指標的實際 (min, max)，加 10% 緩衝。"""
    import sqlite3
    from src.models.features import GAME_STAT_COLS, load_team_match_stats
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            df = load_team_match_stats(conn)
        finally:
            conn.close()
        df = df[df["gender"] == gender_code]
        if df.empty:
            return {}
        ranges = {}
        for col in GAME_STAT_COLS:
            lo, hi = float(df[col].min()), float(df[col].max())
            buf = (hi - lo) * 0.1 if hi > lo else 1.0
            ranges[col] = (round(max(0.0, lo - buf), 1), round(hi + buf, 1))
        return ranges
    except Exception:
        return {}
```

（`DB_PATH` 自 `src.app.helpers` import；檔頭 import 區同步整理：移除未用的 `vec_pct`。）

3. `render()` 內載入段改為：

```python
    artifact, model, explainer = _load_model_and_explainer()
    err = _artifact_error(artifact)
    if err:
        st.error(err)
        return
    feature_cols = artifact["feature_cols"]
    st.caption(
        f"模型版本：{artifact['version']}｜特徵數：{len(feature_cols)}｜"
        f"訓練樣本：{artifact.get('training_samples', '—')}｜"
        f"訓練時間：{artifact.get('trained_at', '—')}")
```

4. 滑杆迴圈改為依 `feature_cols` 順序，範圍查表用基底指標名：

```python
    _data_ranges = _get_data_ranges(ctx.get("gender_code", "M"))

    input_values = {}
    cols = st.columns(2)
    for idx, key in enumerate(feature_cols):
        label, min_v, max_v, default_v, step = SLIDER_CFG[key]
        base = key.rsplit("_roll", 1)[0]
        if base in _data_ranges:
            d_min, d_max = _data_ranges[base]
            min_v = min(min_v, d_min)
            max_v = max(max_v, d_max)
        import math
        min_v = round(math.floor(min_v / step) * step, 4)
        max_v = round(math.ceil(max_v / step) * step, 4)
        default_v = round(round(default_v / step) * step, 4)
        default_v = float(max(min_v, min(default_v, max_v)))
        min_v, max_v, step = float(min_v), float(max_v), float(step)
        with cols[idx % 2]:
            input_values[key] = st.slider(
                label, min_value=min_v, max_value=max_v,
                value=default_v, step=step, key=f"pred_{key}",
            )
```

5. 特徵向量與 SHAP 顯示名稱改以 artifact 為準：

```python
    X = np.array([[input_values[k] for k in feature_cols]])
    ...
    display_names = artifact.get("feature_labels") or feature_cols
```

（SHAP 段原本的 `display_names = [FEAT_LABELS_MAP.get(k, k) for k, *_ in slider_cfg]` 改為上行。）

- [ ] **Step 4: 跑測試確認通過**

Run: `python -m pytest tests/test_prediction_artifact.py tests/test_main_tabs.py tests/test_tab_queries_phase2.py -v`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/helpers.py src/app/tabs/prediction.py tests/test_prediction_artifact.py
git commit -m "fix: 預測 tab 改依 artifact 特徵順序驅動，未知版本即報錯"
```

---

### Task 7: 實跑重訓、產出 v2 pkl、移除舊 pkl、契約測試

**Files:**
- Create: `src/models/match_predictor_v2.pkl`（訓練產物，刻意 commit）
- Delete: `src/models/match_predictor.pkl`
- Test: `tests/test_prediction_artifact.py`（追加對實體 pkl 的契約測試）

**Interfaces:**
- Consumes: Task 5 的 `train.main`、Task 6 的 `SLIDER_CFG`。

- [ ] **Step 1: 標籤差異驗證（真實 vs proxy，唯讀正式 DB）**

```bash
python - <<'EOF'
import pandas as pd
from src.utils.db_config import get_connection
conn = get_connection()
m = pd.read_sql_query("SELECT * FROM matches WHERE is_golden_set=0", conn)
conn.close()
diff = m[(m.home_total > m.away_total) != (m.home_sets_won > m.away_sets_won)]
print(f"總得分 proxy 與真實局數不一致：{len(diff)}/{len(m)} 場")
print(diff[["match_date","home_team","away_team","home_total","away_total","home_sets_won","away_sets_won"]].to_string(index=False))
EOF
```

Expected: 約 13 場不一致（探索實查值；若數字不同，記下實際值，這是換標籤的效益證據）。

- [ ] **Step 2: 實跑重訓**

```bash
python -m src.models.train --trials 100
```

Expected: 印出資料報告（`legacy_skipped` 應對應 2024 舊隊場次、`training_rows` > 0）、cv F1、輸出 `src/models/match_predictor_v2.pkl`。若 `build_training_frame` raise（統計面找不到比分），停下調查——那是 fail-loud 在工作，不准降級為 warning。把 cv F1 與舊 v1 的 0.717（leakage 版，不可直接比）並列記進 commit message。

- [ ] **Step 3: 追加實體 pkl 契約測試**

`tests/test_prediction_artifact.py` 追加：

```python
from pathlib import Path

import joblib
import numpy as np

PKL_PATH = Path(__file__).resolve().parents[1] / "src" / "models" / "match_predictor_v2.pkl"

REQUIRED_KEYS = {
    "model", "model_name", "version", "feature_cols", "feature_labels",
    "label_source", "best_params", "optuna_best_f1", "cv_f1_mean",
    "training_samples", "trained_at", "xgboost_version", "game_stat_cols",
}


def test_shipped_pkl_matches_app_contract():
    artifact = joblib.load(PKL_PATH)
    assert REQUIRED_KEYS <= set(artifact)
    assert _artifact_error(artifact) is None
    X = np.array([[40.0] * 10 + [0.0]])
    proba = artifact["model"].predict_proba(X)
    assert proba.shape == (1, 2)


def test_old_v1_pkl_removed():
    assert not (PKL_PATH.parent / "match_predictor.pkl").exists()
```

- [ ] **Step 4: 移除舊 pkl 並跑全測試**

```bash
git rm src/models/match_predictor.pkl
python -m pytest -v
```

Expected: 全 PASS（含既有全部測試不退步）。

- [ ] **Step 5: Commit**

```bash
git add src/models/match_predictor_v2.pkl tests/test_prediction_artifact.py
git commit -m "feat: 真實局數標籤重訓 v2 模型上線，移除 leakage 版 v1 pkl"
```

（commit message 內文附：樣本數、cv F1、proxy 差異場數。）

---

### Task 8: notebook 收斂與文件同步

**Files:**
- Delete: `notebooks/02_ml_match_prediction.ipynb`、`notebooks/03_ml_v2_prediction_engine.ipynb`
- Modify: `notebooks/01_eda.ipynb`（SQL cell 改走新 schema）
- Modify: `README.md`（模型/notebook 說明）、`CLAUDE.md`（地雷條目）、`docs/ops/season-switch.md`（重訓步驟）

**Interfaces:**
- Consumes: Task 2 的 registration join 寫法。

- [ ] **Step 1: 修 01_eda 的 SQL**

讀 `notebooks/01_eda.ipynb` 找出用 `JOIN players p ON s.player_id`／`p.team_id` 的 SQL cell（探索記錄在 cell 4，以實際內容為準），保持選取欄位語意不變，join 路徑改為：

```sql
FROM player_match_stats s
JOIN roster_registrations r ON s.registration_id = r.registration_id
JOIN players p ON r.player_id = p.player_id
JOIN teams t ON t.team_id = r.team_id AND t.gender = r.gender
```

（原 `p.team_id`/`p.gender` 改 `r.team_id`/`r.gender`；球員屬性 `p.name/p.dob/p.height_cm/p.weight_kg` 照舊。）

- [ ] **Step 2: 驗證 SQL 可跑**

把改好的 SQL 抽出，對正式 DB 唯讀執行：

```bash
python - <<'EOF'
import json, pandas as pd
from src.utils.db_config import get_connection
nb = json.load(open("notebooks/01_eda.ipynb"))
conn = get_connection()
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "read_sql_query" in src and "player_match_stats" in src:
        start = src.index('"""') + 3
        end = src.index('"""', start)
        df = pd.read_sql_query(src[start:end], conn)
        print(f"OK, {len(df)} rows")
conn.close()
EOF
```

Expected: 印出 `OK, <n> rows`，無 OperationalError。（若 notebook 內 SQL 字串不是 `"""` 包裹，依實際格式調整抽取方式。）

- [ ] **Step 3: 刪兩本訓練 notebook**

```bash
git rm notebooks/02_ml_match_prediction.ipynb notebooks/03_ml_v2_prediction_engine.ipynb
```

- [ ] **Step 4: 文件同步**

- `README.md`：模型段落改為「`python -m src.models.train` 重訓（真實局數標籤）」；notebook 清單只剩 `01_eda.ipynb`。
- `CLAUDE.md` 常用指令加一行 `python -m src.models.train`；地雷區「.db 與 .pkl 刻意 commit」條目的 pkl 檔名更新為 `match_predictor_v2.pkl`，並加一句：artifact 需含 `version`/`feature_cols`，app 對未知版本會直接報錯，重訓務必走 `src.models.train`。
- `docs/ops/season-switch.md` 的重訓 checklist 項改為具體指令：`python -m src.models.train --trials 100`，並註明「產出後跑 `python -m pytest tests/test_prediction_artifact.py` 驗契約」。

- [ ] **Step 5: 跑全測試 + 行尾檢查**

```bash
python -m pytest -v
git diff --stat -w
```

Expected: 全 PASS；diff 無純行尾雜訊。

- [ ] **Step 6: Commit**

```bash
git add notebooks/01_eda.ipynb README.md CLAUDE.md docs/ops/season-switch.md
git commit -m "docs: notebook 收斂至 EDA 一本，重訓流程文件化"
```

---

### Task 9: 端到端驗收

**Files:** 無新增（驗證性 task）

- [ ] **Step 1: 全測試**

```bash
python -m pytest -v
```

Expected: 全 PASS（既有 51+ 與本 Phase 新增測試）。

- [ ] **Step 2: Streamlit 實測**

```bash
streamlit run src/app/main.py
```

人工檢查預測 tab：載入 `match_predictor_v2.pkl` 顯示「模型版本：v2｜特徵數：11」、11 個滑杆、範圍合理（對照球隊層級指標的實際分佈）、預測與 SHAP 瀑布圖正常。男女組都切換測一次。無頭環境改用 `streamlit run src/app/main.py --server.headless true` + 瀏覽器/截圖確認，或由使用者驗收。

- [ ] **Step 3: 收尾檢查**

```bash
git log --oneline main..HEAD
git diff --stat -w main..HEAD
```

確認 commit 序列乾淨、無行尾雜訊。回報使用者驗收（merge/push 由使用者決定）。
