---
status: accepted
date: 2026-10-22
---

# matches 欄位所有權

`matches` 表的日常寫入者有兩個：`src/etl/schedule_crawler.py` 的 `upsert_schedule_match`（列表頁，先跑）與 `src/etl/match_crawler.py` 的 `upsert_match`（單場頁，後跑），由 `daily_update.sh` 依序呼叫。兩者對同一列 (game_id, gender) 的欄位寫入規則曾不一致：`match_crawler` 會以單場頁解析不到的 NULL 蓋掉 `schedule_crawler` 寫下的 round_name 與 game_label。本 ADR 定下每個欄位的擁有者與衝突時的行為，並把 `match_crawler` 的 round_name、game_label 改為 COALESCE(新, 舊)。

## 欄位所有權

| 欄位 | schedule_crawler | match_crawler | 衝突時行為 |
|---|---|---|---|
| cup_id、match_date | 覆寫 | 覆寫 | 後寫者贏 |
| start_time、venue | COALESCE(新, 舊) | COALESCE(舊, 新) | 列表頁新值贏；單場頁只在既有值為 NULL 時補缺 |
| round_name、game_label | COALESCE(新, 舊) | COALESCE(新, 舊) | 有值就更新，NULL 不蓋既有值 |
| is_golden_set | COALESCE(golden, 舊)，列表頁沒有 game_label 時不寫 | 覆寫 | 由 match_crawler 擁有 |
| home_team、away_team、home_team_id、away_team_id | 覆寫 | 覆寫 | 後寫者贏；查不到隊名時 team_id 寫 NULL 並 warning |
| 各局比分、total、sets_won | 不碰 | 覆寫 | 只有 match_crawler 寫 |

## 理由

- **start_time、venue 方向相反**：列表頁是時間與場館的權威來源，所以 `schedule_crawler` 的新值贏；單場頁只當後備，既有值為 NULL 才補缺。
- **round_name、game_label 兩側皆 COALESCE(新, 舊)**：兩個頁面都可能缺這兩個欄位（例如列表頁整頁空白、季後賽排名賽的單場頁不給 round_name），所以有值就更新、NULL 不蓋既有值。
- **is_golden_set 由 match_crawler 擁有**：單場頁的解析結果永遠是 0 或 1，不會有缺值。
- **比分欄只有 match_crawler 寫**：`schedule_crawler` 永不碰比分欄，「未開打」的定義（所有局比分皆 NULL）才不會被列表頁寫入破壞。
- **隊伍欄後寫者贏**：兩邊都以 `resolve_team_id` 查隊名，查不到寫 NULL 並 warning。
- **寫入方式**：全是先 SELECT (game_id, gender) 再 INSERT 或 UPDATE，沒有 ON CONFLICT、沒有 DELETE。

## stats_crawler

`stats_crawler` 不寫 `matches`，只唯讀 round_name 來決定 week_label，所以 `daily_update.sh` 必須先跑 `match_crawler`。違序時 week_label 退化成「未比對-日期」並寫入 `roster_registrations`，之後補上 round_name 也不會清除或遷移那些列。這個問題在 #61 處理。

## 不在範圍

- 一次性 migration 腳本已於 #66 移除。

## Consequences

- `tests/test_matches_write_contract.py` 鎖住兩個爬蟲交錯寫同一場的結果：NULL 不蓋 round_name 與 game_label、start_time 與 venue 的方向、比分只由 `match_crawler` 寫入。
- 改任一爬蟲的 UPDATE 子句時，要同步更新本表與該測試。
- ADR 0003 原本的「兩個爬蟲欄位分工」描述與程式不符，改為指向本 ADR。
