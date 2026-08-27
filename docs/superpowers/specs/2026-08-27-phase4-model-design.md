# Phase 4 模型優化設計（2026-08-27 定案）

## 背景與目標

路線圖 Phase 4：勝負標籤改用 matches 表真實局數、v2 流程重訓、pkl 檔名版本化、
notebook 收斂。成功標準為**正確性優先**：標籤語意正確、流程可重現、可測試；
重訓後準確率持平甚至略降可接受。

### 探索發現的現況問題（2026-08-27 實查）

1. **版本偵測 live bug**：`src/app/tabs/prediction.py:125` 讀 `feature_names`，
   但兩本訓練 notebook 存的 key 是 `feature_cols`，`n_features` 永遠為 0，
   UI 恆顯示「特徵數：0」並 fallback 到 V1 滑杆。直接換 11 特徵 pkl 會 crash。
2. **線上 pkl 是 notebook 03 自判「data leakage、指標不可信」的 v1**
   （5 特徵、212 筆、F1 0.717）；v2 訓練過但產物從未 commit。
3. **真實標籤未被使用**：`matches` 170/170 場皆有 `home_sets_won/away_sets_won`，
   proxy label（總得分比大小）實測 13/170（7.6%）標錯，且 v2 的 `win_streak`
   由 proxy label 滾動計算，誤差被放大。
4. **三本 notebook（01 EDA、02 v1、03 v2）的 SQL 全數停留在 Phase 2 前 schema**
   （`players.team_id`、`s.player_id` 已不存在），現在執行即 OperationalError。
5. **隊名映射不全**：`constants.py` 的 `OPP_SHORT_TO_TEAM`/`TEAM_NAME_SHORT`
   缺 臺北Conti、連莊、彰化三大有線 與「男排/女排」後綴；且 notebook 02/03
   各有一份硬編碼副本（共三份）。
6. **模型／預測零測試覆蓋**。
7. **滑杆範圍尺度不符**：`prediction.py` 拿球員層級聚合（`helpers.py`
   `get_league_aggregated_stats`）的 min/max 當球隊層級特徵的滑杆範圍。

### 已定案的範圍決策

- 成功標準：正確性優先（準確率持平可接受）。
- 訓練流程收進 script；notebook 只留 `01_eda.ipynb`（修 SQL 後保留）。
- 訓練資料：matches × 逐場統計的重疊區間全量（現況即本季，開季後自然累加）。
- 重訓為**手動觸發**，不掛 daily_update.sh / GitHub Actions。
- 滑杆尺度修正納入本 Phase。

## 設計

### 1. 標籤與資料層（`src/models/features.py` 新增）

- 標籤：`matches.home_sets_won` vs `away_sets_won` 比大小，每場產生主客兩筆
  球隊層級樣本（win=1/0），取代總得分 proxy。
- 資料流：`player_match_stats` → `registration_id` → `roster_registrations`
  取 `(team_id, gender)`（Phase 2 新 schema），以 `(match_date, team_id, gender)`
  聚合成球隊層級五指標（ASR、GP_pct、DIG_pct、BLK_per_set、ACE_pct，沿用
  `safe_pct/safe_div` 防禦），再與 matches 主客隊對接。
- 隊名正規化：**只在 `src/utils/constants.py` 維護一份**，補齊
  臺北Conti、連莊、彰化三大有線 與「男排/女排」後綴 → `(team_id, gender)`
  的映射；notebook 內兩份副本隨 notebook 刪除。
- 對接失敗 **fail loud**：隊名映射不到或比分缺漏時，列出未匹配場次清單並
  raise，不默默丟棄。
- `is_golden_set=1` 的場次不進訓練樣本（單局加賽與整場勝負語意不同）。

### 2. 特徵與訓練（`src/models/train.py` 新增）

- 特徵沿用 v2 的 11 個：五指標 × roll3/roll5（`shift(1)` 後 rolling mean，
  防 leakage 機制保留）+ `win_streak`；`win_streak` 改由真實標籤滾動計算。
- 入口：`python -m src.models.train`。TimeSeriesSplit 交叉驗證 + Optuna
  調參（trials 數可參數化，預設較小）+ XGBoost。
- 輸出：印出 cv 指標、訓練樣本數、未匹配場次統計，並匯出版本化 pkl。

### 3. Artifact 與 app 載入

- 統一 artifact schema：`version`（"v2"）、`feature_cols`、`feature_labels`、
  `model_name`、`trained_at`、`label_source`（"matches.sets_won"）、
  `training_samples`、cv 指標、`xgboost_version`。
- 檔名版本化：產出 `src/models/match_predictor_v2.pkl`；`MODEL_PATH`
  常數（`src/app/helpers.py`）指向現行版檔名；舊 `match_predictor.pkl`
  自 repo 移除（git 歷史可回溯）。
- `prediction.py`：改讀 `feature_cols`；驗 `version` 屬已知集合，不認得即
  `st.error` + 停止，不 fallback；V1 滑杆設定與分支刪除。
- 滑杆範圍：由 `features.py` 提供球隊層級特徵分佈查詢，取代球員層級統計。

### 4. Notebook 與文件

- 刪 `notebooks/02_ml_match_prediction.ipynb`（leakage 版）與
  `notebooks/03_ml_v2_prediction_engine.ipynb`（邏輯已進 script）。
- `notebooks/01_eda.ipynb` SQL 改走新 schema 後保留。
- 同步更新：`README.md` 模型/notebook 說明、`CLAUDE.md` 補 pkl 版本化
  地雷條目、`docs/ops/season-switch.md` 重訓步驟改為具體指令。

### 5. 測試與驗收

測試（pytest，接續既有骨架）：
- 標籤產生器單元測試：以 proxy 標錯的 13 場為回歸樣本，驗真實標籤正確。
- 隊名正規化測試：全部 matches 隊名皆可映射到 `(team_id, gender)`。
- Artifact 契約測試：實際載入 pkl，驗 key 齊全且 `len(feature_cols)` 與
  app 滑杆設定長度一致。
- Leakage 防護測試：各隊首場的 rolling 特徵必須被 shift 掉（不含當場資訊）。
- Train 冒煙測試：小參數（少 trials）快速跑通全流程。

驗收：
1. 全測試綠（含既有 51+ 測試不退步）。
2. 實跑 `python -m src.models.train` 產出 v2 pkl，報告樣本數與 cv 指標。
3. 本機 `streamlit run src/app/main.py` 預測 tab 實測：載入新 pkl、滑杆
   範圍合理、預測與 SHAP 圖不炸。

## 明確不做

- 重訓自動化（排程/CI）。
- 新增特徵或特徵工程迭代（沿用 v2 的 11 特徵）。
- 換模型架構（維持 XGBoost）。
- 預測 tab UI 重設計（字級等留 Phase 5）。
