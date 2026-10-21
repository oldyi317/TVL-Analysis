---
status: accepted
date: 2026-10-19
---

# 賽季報名名單另開 season_rosters 表，不寫進 roster_registrations

2026-27 開季時外部統計系統 CupID=22 尚未建立，`roster_registrations`（週登錄）抓不到新季資料，球隊頁與球員頁沒有名單、背號與位置。官網球隊頁有整季的賽季報名名單，要先匯入。選項：(a) 新表 `season_rosters`（player_id、team_id、gender、cup_id、jersey_number、position，UNIQUE(player_id, gender, cup_id)）；(b) 寫進 `roster_registrations`，加 `source='season_roster'` 與一個不會撞到真實週次的 `week_label`。決定 **(a)**。

理由：主要位置（`site_data.py` 依登錄計次）與球員本季球隊／背號（取最新登錄）都直接讀 `roster_registrations` 且不分 source，(b) 要在每個讀者加排除條件，往後新增讀者也得記得；若 `week_label` 撞到真實週次，`stats_crawler --rosters` 的 ON CONFLICT 會覆寫該列。(b) 還要重建表才能改 CHECK。賽季報名名單與週登錄在領域上本來就是兩個概念（見 CONTEXT.md）。

同時定下的規則：

- **只補顯示，不進比較**：球隊頁本季沒有任何週登錄時才顯示賽季報名名單（標題標明是報名名單）；球員頁沒有主要位置時以報名位置與背號補顯示。主要位置、PR 值、同位置分佈只看週登錄。
- **每日同步、整隊覆蓋**：官網球隊頁的爬取與寫入加進 `daily_update.sh`（排在 `schedule_crawler` 後，新季賽程寫入當天名單就歸到新季、不會蓋掉舊季；失敗不中止）；每隊以官網為準覆蓋本季報名名單，該隊抓到 0 人（官網佔位頁）則跳過不動。
- **賽季歸屬**：寫入 `matches` 表中最新的 cup_id，不用今天日期查 `seasons`（休賽期官網仍是舊季名單）；該季 `end_date` 已過則整步略過，休賽期官網先換新季名單時不會蓋掉舊季。賽程先出、官網名單還沒換時，舊名單會暫時寫進新季，官網更新後隔天整隊覆蓋即修正。
- **整站抓不到時中止**：`crawler` 抓到 0 人以非 0 結束，`db_loader` 不會拿 runner 上前一天的 CSV 去寫。
- **隊伍對應**：官網隊伍頁編號與本地 team_id 已脫鉤，一律以隊名經 `team_aliases` 解析；`db_loader` 不再以 CSV 編號寫 `teams`，解析不到的隊伍 log warning 並略過。

## Consequences

- `db_loader` 每日執行，官網的身高、體重與新球員也每日更新進 `players` 身分層（COALESCE，不以空值覆蓋）。
- 球隊列表的球員卡與球員頁一樣，以報名位置補顯示。
- 已知限制（暫不處理）：同一球員在官網同時掛兩隊時以處理順序最後者為準；只抓到部分球員的隊伍仍整隊覆蓋、不警告；匯出器假設 DB 已有 `season_rosters` 表（commit 進 git 的 DB 已建）。
