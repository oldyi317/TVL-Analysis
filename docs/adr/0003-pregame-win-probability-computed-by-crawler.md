---
status: accepted
date: 2026-10-15
---

# 賽前勝率在爬蟲端計算並落 DB

第三期要在首頁「下一場」與未開打賽事卡顯示賽前勝率。模型是 `src/models/match_predictor_v2.pkl`（XGBoost，joblib 序列化），載入需要 joblib、xgboost、pandas；但 ADR 0001 定下匯出器 `src/export/site_data.py` 只用標準庫，pages.yml 在雲端 runner 不裝 pip 套件。三個選項：(a) pages.yml 補 pip install，匯出器直接載模型；(b) 自架 runner 跑 `daily_update.sh` 時算好寫進 DB 新表，匯出器只 SELECT；(c) 爬蟲端算好直接寫 JSON 進 `web/src/lib/data/`。決定 **(b)**：新增 `match_predictions` 表（match_id、model_version、home_win_prob、computed_at，UNIQUE(match_id, model_version)），由 `daily_update.sh` 在 `stats_crawler --incremental` 之後新增一步計算，隨 `.db` 一起 commit。

理由：守住匯出器只用標準庫的既有決策，pages.yml 的建置時間與失敗面不變；預測落在 DB 就有歷史，之後可做賽前預測對實際結果的命中率，(c) 做不到；pkl 已在 repo，不新增雲端依賴。代價是預測更新頻率等於 daily_update 頻率而非建置頻率。

同時定下的規則：

- **特徵定義不另寫**：賽前特徵必須呼叫 `src/models/features.py` 訓練時同一套函式（視窗 3 與 5 場、先 shift(1)、分組鍵 team_id+gender、排除黃金決勝局、跨季連續不重置）。改視窗、跨季或黃金局定義屬於重訓模型，另開票，不在第三期。
- **重算與凍結**：每次 daily_update 對所有未開打賽事重算並 upsert；賽事一打完（任一局比分非 NULL）就不再重算，保留最後一次賽前值，不刪不覆寫。不存特徵快照，要除錯時重算即可。預測步驟失敗只記 log，不中止 daily_update。
- **顯示門檻**：兩隊各有至少 1 場有技術統計的歷史賽事（不限賽季）才算得出；算不出時預測步驟不寫列，匯出為 null，卡片顯示「資料不足」。開季初會以上季末近況推估，寫在模型限制說明裡。
- **兩個爬蟲欄位分工**：見 ADR 0008。
- **不加 status 欄**：「未開打」的唯一定義是所有局比分皆 NULL；`match_crawler` 遇各局全空時 sets_won 寫 NULL 而非 0，前端沿用 `sets.length === 0`。

## Consequences

- `daily_update.sh` 多兩步：`schedule_crawler`（排在 `match_crawler` 前）與預測計算（排在 `stats_crawler --incremental` 後）。
- 匯出器在 `matches.json` 每筆加 `home_win_prob`（nullable），`web/schema/matches.schema.json` 同步；模型限制說明文案放前端常數，不進 JSON。
- 前端 `nextMatch` 排序鍵改為 (date, start_time)；百分比以兩隊並列、整數呈現，相加為 100。
- 自架 runner 的 daily-crawl 必須真的在跑，否則預測表永遠空；其修復是第三期前端票的前置。
- 「賽程」在 CONTEXT.md 是避免詞，`schedule_crawler` 只是檔名，UI 文案一律用「賽事」。
