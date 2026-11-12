# 換季手動 Checklist

每年新賽季開打前逐項執行。未完成前每日排程照跑但抓不到新季資料（`seasons` 表
還沒有新季列），屬預期行為，不需先停 workflow。單場數據與週登錄自 cup 22 起由官網
賽事頁供應（ADR 0009），換季不再有外部統計系統的項目。

## 常數更新（src/utils/constants.py 為主）

- [x] 在 `seasons` 插一列新賽季：於 `sql/schema.sql` 的種子 `INSERT OR IGNORE INTO seasons`
      加一列（`cup_id`、名稱如 `2026-27`、起訖日 `2026-07-01`～`2027-06-30`）；下次
      `match_crawler` 執行時 `init_matches_table` 重跑 schema.sql，`INSERT OR IGNORE`
      會把新列帶進既有 DB 的 `seasons` 表（只補列，不改既有表結構），也可直接用
      sqlite 對 DB 插入。未插入時 `match_crawler` 會因找不到賽季而報錯。
      （2026-27 已於 2026-10-15 插入 cup_id=22，見下方 daily-crawl 維運紀錄；下季再把本項改回未勾。）
- [ ] `match_crawler.py` 的 `--range-start`/`--range-end` argparse 預設值（目前 220–400 已涵蓋 2026-27 開季的 329+／281+，且 `--range-end` 不含端點；不在範圍內才需要改）：
      新季 game_id 起點不用再手動到官網確認，`schedule_crawler`（daily_update 第一步）會先把
      官網列表頁（含「下一年度」頁）的場次寫進 matches，查
      `SELECT gender, min(game_id), max(game_id) FROM matches WHERE cup_id=<新季> GROUP BY gender` 即可。
- [x] 球隊身分（有新隊伍/改名才需要動；2026-27 已於 2026-10-15 補：鳳凰／龍邁延續 4／5，三福／可利為新 team_id 11／12，本地 team_id 自此與官網隊伍頁編號脫鉤，下季再把本項改回未勾）：改 `sql/team_seed.sql`（teams 與 team_aliases），
      再套用：`python -c "from src.utils.db_config import get_connection; from src.utils.teams import apply_team_seed; c=get_connection(); apply_team_seed(c); c.commit()"`。
      過渡期相容：`TEAM_NAME_SHORT`（constants.py）仍被爬蟲使用（`EXT_TEAM_MAPS` 止於 cup 21，新季不補，ADR 0009）；官網隊名的別名只補 `sql/team_seed.sql` 的 team_aliases；`OPP_SHORT_TO_TEAM` 已廢除，單場出賽名單隊名改走 team_aliases。
      `match_crawler` 寫入 matches 時以 `team_aliases`（`source <> 'ext_stats'`）解析 `home_team_id`/`away_team_id`；
      新隊名或新別名未進種子會 log warning 並寫 NULL，記得補 `sql/team_seed.sql`。

- [ ] 隊色：開官網賽程列表（`/schedule`、`/wschedule`），每隊旁色卡是 `<span style="color:#xxxxxx">▇</span>`，
      逐隊核對 `sql/team_seed.sql` 的 `color`（隊色會跨季變，2025-26→2026-27 美津濃就從淺藍改深藍），有差就改 seed 再套用。
- [ ] 隊徽（有新隊伍或拿到舊隊圖才需要動）：原圖放 `assets/team_logo/<隊名>.png`（jpg 也可，縮圖一律輸出 png），縮成最長邊 512px 存到
      `web/static/logos/`。檔名規則：`<men|women>-<team_id>.png` 是現行隊徽；`<men|women>-<team_id>-<cup_id>.png` 是該季專用隊徽
      （該季當季隊名與現行不同時才會用到，見 ADR 0005；前端 `seasonTeam` 先找該季專用圖，沒有且當季隊名等於現行隊名才用現行圖，否則顯示隊色圓標）。
      **有圖就產，不分現役與否**：非現役球隊的圖會出現在歷史賽季的戰績與賽事，且在其出賽賽季有球隊頁（#71）；缺圖的隊顯示隊色圓標，不會壞圖。
      縮圖一行：`~/venvs/tvl/bin/python -c "from PIL import Image; im=Image.open('assets/team_logo/X.png').convert('RGBA'); im.thumbnail((512,512)); im.save('web/static/logos/men-N.png', optimize=True)"`。
      `src/lib/data/logos.json`（隊徽檔名清單）由 `npm run gen` 掃 `static/logos` 產生、不進 git，`check`／`build` 前會自動重跑；本機要先跑單元測試就手動 `npm run gen`。原圖與縮圖一併提交。
- [ ] `STANDINGS_NOTES`（src/export/site_data.py）：以 `(cup_id, gender)` 標記官網收錄不全的
      戰績（2024-25 已於 2026-10-20 補齊，目前為空）。新季正常不用加；若官網缺漏賽事再補一列。

## 季前作業

- [x] 跑 `python -m src.etl.crawler` 抓新季官網名單 → `data/raw/`。（2026-27 已於 2026-10-15 跑：10 隊 161 人；下季再把本項改回未勾）
- [x] 跑 `python -m src.etl.db_loader` 更新 players 身分層與 teams。（2026-10-15 已跑：players 151→222；下季改回未勾）
- [ ] 賽季報名名單（ADR 0004）：`db_loader` 會把官網球隊頁的名單寫進 `season_rosters`，賽季取 `matches` 最新 cup_id（該季已結束則整步略過），`daily_update.sh` 在 `schedule_crawler` 後每日同步。換季時確認新季賽程已進 matches 之後名單才會歸到新季；新隊名要先補 `sql/team_seed.sql`，否則該隊 log warning 並略過。（2026-10-19 首次匯入 cup 22：10 隊各 16 人）
- [ ] 球隊改名（ADR 0005）三步：(a) 更新 `teams` 現行隊名（`sql/team_seed.sql`）；
      (b) `team_season_names` 補上舊名所屬的季（種子同寫在 `sql/team_seed.sql`，
      用 `apply_team_seed` 既有一行套用）；(c) 舊隊徽改名加 cup 後綴再放新圖：
      `git mv web/static/logos/<group>-<id>.png web/static/logos/<group>-<id>-<舊季 cup_id>.png`（舊名用過不只一季就每季各複製一份），
      新隊徽照上面「隊徽」一項縮圖存為無後綴的 `<group>-<id>.png`。少做 (c) 歷史賽季會變成隊色圓標（不會拿新隊徽充數）。
- [ ] 到 Actions 頁確認 daily-crawl workflow 未因 60 天不活動被自動停用（被停用則按 Re-enable；GitHub 停用前會寄警告信）。
- [ ] 手動 `workflow_dispatch` 觸發一次 daily-crawl 驗證整條管道，
      確認新季第一批資料正確落庫（cup_id 應為新季編號）。

## 歷史賽季回填

補舊賽季的週登錄與單場數據（cup 20 / 2024-25 已依此跑過，見 issue #34）。只適用 cup 21 以前：cup 22 起單場數據以官網賽事頁為唯一來源（ADR 0009），`stats_crawler` 對這些賽季會拒絕執行。一次性本機執行，不進每日更新；以下命令都從 repo 根目錄跑，會打外部系統或改 DB。

- [ ] 確認外部系統該屆有資料：`curl -s -o /dev/null -w '%{http_code}' 'http://114.35.229.141/_handler/Match.ashx?CupID=<N>&MatchID=1'`，200 有資料、500 沒有（程式的 `_match_page_exists` 另要求回應 `<h3>` 含「組」，200 但標題無「組」也視為沒有）。
      cup 20 的 MatchID 為 1–102 與 201–210 有資料，其餘回 500，`fetch_match_list` 的逐號探測會處理。
- [ ] 前置資料：DB 要先有該季 `seasons` 列、`matches`（比分；`python -m src.etl.match_crawler --range-start <起> --range-end <迄>`，`--range-end` 不含端點）、
      `season_rosters`（使用者轉錄的名單 CSV：`python -m src.etl.db_loader --csv <路徑> --cup-id <N>`；cup 20 用 `assets/tvl_20_player_list/tvl20_rosters.csv`）。
      沒有 `season_rosters` 時 `PlayerResolver` 退回全域姓名比對＋動態新增球員（log「無賽季報名名單」，cup 21 即此路徑），歷史賽季不建議走這條。
- [ ] 補每屆隊伍對照：`EXT_TEAM_MAPS`（constants.py）以 cup_id 為鍵，外部 TeamID 每屆重編；查無該屆會明確報錯，不落回其他賽季。
      查證：外部球員名單（`No.<背號>-<姓名>` 下拉）對賽季報名名單做姓名重疊，每隊第二高重疊應為 0 才無歧義。
      cup 20 範例：1 屏東台電、2 雲林美津濃、3 彰化三大有線、4 臺北Conti、5 龍邁建設（當季名桃園臺產）、6 連莊（以上男）、7 高雄台電、8 臺北鯨華、9 新北中纖、10 義力營造（以上女）。
      當季舊隊名要在 `team_aliases`（`sql/team_seed.sql`）才解析得出。
- [ ] 執行順序（都是補缺不清表，可重跑）：先 `python -m src.etl.stats_crawler --rosters --cup-id <N>`（週登錄），
      再 `python -m src.etl.stats_crawler --cup-id <N>`（單場數據）。不帶 `--cup-id` 抓 `EXT_CUP_ID`（固定為外部系統最後一季 21）。
      外部日期只有月/日，年份由 `seasons` 起迄推算；兩個年份都落不進區間會 warning 並略過該筆。
- [ ] 球員對應與清單檢視：順序為「同隊姓名 → 同隊背號對 `season_rosters` → 略過（不新增身分）」。結束時 log 兩份清單：
      「姓名不同、以背號對上」與「略過（對不上賽季報名名單）」。維護者看完決定是否把略過者補進 CSV；補了就重跑 `db_loader --csv <路徑> --cup-id <N>`，再重跑上面兩步補缺。
      cup 20 實例：Darin→Darin Pinsuwan、Jarasporn→Jarasporn Bundasak 以背號對上；略過 吳承翰（雲林美津濃 #6）、韓晟曦（臺北Conti #12）、B.BAGUNAS（連莊 #1）、陳芃宣（臺北鯨華 #10）。
- [ ] 輪次補值：官網寫「總冠軍賽」會正規化為「總決賽」。若歷史賽季某些場次 `round_name` 為空，重抓那幾場（cup 20 總決賽為男 game 261–263、女 wgame 225–227）：
      `python -c "from src.utils.db_config import get_connection; from src.etl.match_crawler import scrape_match_page, upsert_match; c=get_connection(); [upsert_match(c, scrape_match_page(p, i)) for p, i in [('game', 261), ('wgame', 225)]]; c.commit()"`
- [ ] 驗證：有單場數據的場次數應等於外部系統有資料的場次（cup 20：112 場，男 65、女 47）；
      `python -c "from src.utils.db_config import get_connection; c=get_connection(); print(c.execute('SELECT rr.cup_id, count(*) FROM player_match_stats p JOIN roster_registrations rr USING (registration_id) GROUP BY rr.cup_id').fetchall())"`
      看各季單場數據筆數；跑 `python -m src.export.site_data` 後 `web/src/lib/data/box_scores.json` 該季場數一致。其他賽季筆數前後不變（cup 21：單場數據 3807、週登錄 2034）。
- [ ] commit：程式與資料分兩個 commit，資料 commit 只含 `data/db/tvl_database.db`，訊息寫來源與筆數。

## 模型（可延後）

- [ ] 新季累積足夠場次後跑 `python -m src.models.train --trials 100` 重訓
      match_predictor_v2；產出後跑 `python -m pytest tests/test_prediction_artifact.py` 驗契約。
