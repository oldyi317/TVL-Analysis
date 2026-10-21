---
status: accepted
date: 2026-10-20
---

# 當季隊名獨立成表，隊徽依賽季落回

球隊改名（2026-27 臺北國北獅→臺北鳳凰、桃園臺灣產險→龍邁建設）後，`teams` 只有一組現行隊名，歷史賽季的戰績、賽事卡、球員頁都顯示新名字與新隊徽。舊名雖在 `team_aliases` 以 `source='former'` 留著，但別名表的職責是「字串解析成哪一隊」，主鍵是別名字串、一隊可有多種寫法，沒有賽季範圍，塞進去要再加「哪個 former 是顯示用」的旗標。週登錄與賽季報名名單只存 team_id，`matches` 的隊名字串只覆蓋有出賽的隊季且寫法不一，都推不出可靠的當季隊名。

決定：新增 `team_season_names (team_id, gender, cup_id, team_name, short_name)`，只收與現行隊名不同的季，由 `sql/team_seed.sql` 種子；改名粒度為賽季，不支援季中改名。匯出器在 `teams.json` 每隊加密集的 `names_by_cup`（`cup_ids` 每季一筆，查無就填現行隊名），落回邏輯只在匯出器做一次。前端以 `seasonTeam(team, cupId)` 把季範圍的資料換成當季隊名，球隊頁標題仍用現行隊名並列出曾用名。隊徽檔名加賽季後綴（`{group}-{team_id}-{cup_id}.png`）表示該季專用，無後綴為現行；該季改過名又沒有專用圖時退回隊色圓標，不顯示現行隊徽。

## Consequences

- `teams.schema.json` 與鎖 teams.json 欄位的測試隨 `names_by_cup` 更新；`team_aliases` 與 ADR-0002 的解析規則不變。
- 改名變成三個動作：`teams` 更新現行隊名、`team_season_names` 補上一季、舊隊徽改名加 cup 後綴；寫進 `docs/ops/season-switch.md`。
- 隊色維持跨季一色（CONTEXT.md「隊色」），只有隊名與隊徽分季。
- 非現役球隊有圖就產隊徽，仍不給球隊頁。
