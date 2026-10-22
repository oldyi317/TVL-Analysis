---
status: accepted
date: 2026-10-12
---

# matches 改存 team_id 並新增 seasons 表

`matches` 以隊名字串存雙方球隊，隊名在四張對照表與一個爬蟲別名表各自維護，官網全名（桃園臺灣產險）與 repo 簡稱（桃園臺產）還不一致；`matches` 也沒有 cup_id，季別只能用日期推。決定：新增 `seasons` 表（cup_id、名稱、起訖日）並讓 `matches` 帶 cup_id；`teams` 加 `short_name`、`official_name`、`color`；新增 `team_aliases` 表收攏所有別名；`matches` 改存 team_id。同時讓 `matches` 的比分欄位允許 NULL 並加 `start_time`，為未開打賽事預留位置。

這打破了 repo「schema 只用 CREATE TABLE IF NOT EXISTS 增、不改既有表」的慣例，需要一次性遷移。換到的是球隊身分與賽季的單一來源，前端匯出器不必各自解析隊名與日期。

## Consequences

- 這是前端動工前的獨立前置 PR；匯出器的 JSON 結構依賴這些欄位。
- `match_crawler` 寫入時需透過 `team_aliases` 解析成 team_id。
- 換季從手動改常數變成在 `seasons` 插一列，但 `EXT_CUP_ID` 等外部系統常數仍需手動更新（ETL 換季本身不在重整範圍）。
- 2024-25 季的舊隊（彰化三大有線、臺北Conti、連莊）需補進 `teams`，標記為非現役，顯示在該季戰績但不給球隊頁。（2026-10-22 補述：#34 回填後資料已齊，非現役隊在其出賽賽季亦有球隊頁，#71。）
