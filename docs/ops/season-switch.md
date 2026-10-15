# 換季手動 Checklist

每年新賽季開打前逐項執行。未完成前每日排程照跑但抓不到新季資料（EXT_CUP_ID
仍指舊季），屬預期行為，不需先停 workflow。

## 常數更新（src/utils/constants.py 為主）

- [ ] `EXT_CUP_ID`（constants.py）：外部系統的 `.aspx` 頁面自 2026-09-29 起全部 404，
      改用 handler 探測：`curl -s -o /dev/null -w '%{http_code}' 'http://114.35.229.141/_handler/Match.ashx?CupID=<N>&MatchID=1'`，
      200 表示該季已有資料、500 表示沒有。cup_id 就是屆數（官網標題「企業22年」= 22），
      外部系統還沒建新季前 `EXT_CUP_ID` 維持舊季，rosters 步驟照抓舊季即可。
- [x] 在 `seasons` 插一列新賽季：於 `sql/schema.sql` 的種子 `INSERT OR IGNORE INTO seasons`
      加一列（`cup_id`、名稱如 `2026-27`、起訖日 `2026-07-01`～`2027-06-30`）；下次
      `match_crawler` 執行時 `init_matches_table` 重跑 schema.sql，`INSERT OR IGNORE`
      會把新列帶進既有 DB 的 `seasons` 表（只補列，不改既有表結構），也可直接用
      sqlite 對 DB 插入。未插入時 `match_crawler` 會因找不到賽季而報錯。
      （2026-27 已於 2026-10-15 插入 cup_id=22，見下方 daily-crawl 維運紀錄；下季再把本項改回未勾。）
- [ ] `SEASON_YEAR_MAP` 與 `DEFAULT_YEAR`（constants.py）：**跟著 `EXT_CUP_ID` 一起改，不跟官網賽季**。
      這兩個常數只有 `stats_crawler` 用（外部系統的「MM/DD」補年份），改成新季的
      「11、12 月屬 X 年，其餘屬 X+1 年」要等外部系統建好新季、`EXT_CUP_ID` 切過去的同一次。
      提前改會讓每日 rosters 步驟把舊季 11、12 月算成下一年，去重鍵（match_date）失效而重複登錄。
      （2026-10-15：官網已換季但外部系統 CupID=22 仍回 500，兩者都維持 21／2025。）
- [ ] `match_crawler.py` 的 `--range-start`/`--range-end` argparse 預設值（目前 220–400 已涵蓋 2026-27 開季的 329+／281+，且 `--range-end` 不含端點；不在範圍內才需要改）：
      新季 game_id 起點不用再手動到官網確認，`schedule_crawler`（daily_update 第一步）會先把
      官網列表頁（含「下一年度」頁）的場次寫進 matches，查
      `SELECT gender, min(game_id), max(game_id) FROM matches WHERE cup_id=<新季> GROUP BY gender` 即可。
- [x] 球隊身分（有新隊伍/改名才需要動；2026-27 已於 2026-10-15 補：鳳凰／龍邁延續 4／5，三福／可利為新 team_id 11／12，本地 team_id 自此與官網隊伍頁編號脫鉤，下季再把本項改回未勾）：改 `sql/team_seed.sql`（teams 與 team_aliases），
      再套用：`python -c "from src.utils.db_config import get_connection; from src.utils.teams import apply_team_seed; c=get_connection(); apply_team_seed(c); c.commit()"`。
      過渡期相容：`EXT_TEAM_MAP`、`OPP_SHORT_TO_TEAM`、`TEAM_NAME_SHORT`（constants.py）與
      `TEAM_ALIAS`（match_crawler.py）仍被爬蟲使用，需同步修改，直到爬蟲改用 DB。
      `match_crawler` 寫入 matches 時以 `team_aliases`（`source <> 'ext_stats'`）解析 `home_team_id`/`away_team_id`；
      新隊名或新別名未進種子會 log warning 並寫 NULL，記得補 `sql/team_seed.sql`。

- [ ] `STANDINGS_NOTES`（src/export/site_data.py）：以 `(cup_id, gender)` 標記官網收錄不全的
      戰績（目前只有 2024-25 兩組）。新季正常不用加；若官網缺漏賽事再補一列。

## 季前作業

- [x] 跑 `python -m src.etl.crawler` 抓新季官網名單 → `data/raw/`。（2026-27 已於 2026-10-15 跑：10 隊 161 人；官網球員卡自本季起無生日欄，dob 全空屬預期，下季再把本項改回未勾）
- [x] 跑 `python -m src.etl.db_loader` 更新 players 身分層與 teams。（2026-10-15 已跑：players 151→222，dob 以 COALESCE 保留舊值；下季改回未勾）
- [ ] 到 Actions 頁確認 daily-crawl workflow 未因 60 天不活動被自動停用（被停用則按 Re-enable；GitHub 停用前會寄警告信）。
- [ ] 手動 `workflow_dispatch` 觸發一次 daily-crawl 驗證整條管道，
      確認新季第一批資料正確落庫（cup_id 應為新季編號）。

## 模型（可延後）

- [ ] 新季累積足夠場次後跑 `python -m src.models.train --trials 100` 重訓
      match_predictor_v2；產出後跑 `python -m pytest tests/test_prediction_artifact.py` 驗契約。

## daily-crawl 維運（自架 runner）

### 失敗時怎麼看

- Actions 頁與 `gh run view <id> --log-failed` 對自架 runner 的 run 常是空的；改抓整份日誌：
  `gh api repos/oldyi317/TVL-Analysis/actions/runs/<run_id>/logs > run.zip`，用 `python -I -c "import zipfile;zipfile.ZipFile('run.zip').extractall('run')"`
  解開（runner 機器沒裝 unzip），步驟輸出在 `run/0_update.txt`。
- runner 本機 `~/actions-runner/_diag/Worker_*.log` 只有程序起訖與 exit code，沒有步驟 stdout。
- 本機重現：**不要在 `/mnt/d/TVL-Analysis` 直接跑 `DRY_RUN=1 bash scripts/daily_update.sh`**，
  腳本結尾會 `git checkout -- data/db/tvl_database.db`，會把工作樹裡未 commit 的 .db 改動洗掉；
  先 `git clone /mnt/d/TVL-Analysis <暫存目錄>` 再在 clone 裡跑。

### 2026-09-29 ～ 2026-10-14 連續失敗紀錄（issue #23）

兩個獨立原因，runner 日誌（run 38520838587）與本機 `DRY_RUN=1` 重現的錯誤輸出尾段：

```text
# runner 2026-10-14 / 本機 2026-10-15 重現（stats_crawler --rosters）
  File ".../src/etl/stats_crawler.py", line 350, in fetch_match_list
    r.raise_for_status()
requests.exceptions.HTTPError: 404 Client Error: Not Found for url: http://114.35.229.141/Match.aspx?CupID=21
##[error]Process completed with exit code 1.

# 本機 2026-10-15 重現（match_crawler，官網當天剛上架新季賽程）
    "SELECT cup_id FROM seasons WHERE ? BETWEEN start_date AND end_date",
RuntimeError: 找不到涵蓋 2026-10-24 的賽季，請在 sql/schema.sql 的 seasons 種子與 DB 的 seasons 表新增該賽季。
```

1. 外部統計系統 `http://114.35.229.141` 所有 `.aspx` 頁面自 2026-09-29 起回 404（IIS 錯誤頁
   「檔案 '/Match.aspx' 不存在」，官網首頁連到的 `Default.aspx?CupID=21` 也一樣），但
   `_handler/*.ashx` 全部正常。`stats_crawler --rosters` 靠 `Match.aspx` 下拉選單列舉 MatchID，
   因此整步拋 `HTTPError 404`。修法：`fetch_match_list` 遇 404 改以 `Match.ashx` 從 1 逐號探測
   （查無該場回 HTTP 500），連續 120 個空號即停（2025-26 實測例行賽 1–108、季後賽 201–210，
   中間空 92 格），多花約 2 分鐘；若 `Match.aspx` 哪天回來，仍優先走下拉選單。
2. 官網 2026-10-15 上架 2026-27 賽程（`/game/330`、`/wgame/281` 起，首日 2026-10-24），
   `match_crawler` 的 `resolve_cup_id` 找不到涵蓋該日期的賽季而 `RuntimeError`。修法：
   seasons 種子加 `(22, '2026-27', '2026-07-01', '2027-06-30')`（官網標題「企業22年」）。

連帶兩件事：

- seasons 多了 22 之後 `seasons.json` 的 `current_cup_id` 變 22，首頁會指向 2026-27；
  web 的 styleguide 樣本改挑「最近有五支男隊的賽季」（`demoSeason`），不再直接用 current。
- **2026-27 男子組名單未定（開季後再核對）**：首批賽程出現四個 seed 沒有的隊名
  （三福產業、臺北鳳凰、新竹可利工程師、龍邁建設；官網單場頁 `/game/N` 誤植為「龍脈建設」，列表頁 `/schedule/115` 為正確的「龍邁建設」，補 team_seed 時兩個別名都要收），2025-26 的獅子王、臺北國北獅、桃園臺產
  沒出現；官網隊伍頁（`/team/N`）當時全部轉回首頁，查不到名單，無法判斷是改名還是新軍，
  使用者也說官網還在更新。因此**沒有**補 `sql/team_seed.sql`，這些場次在 DB 裡 `home/away_team_id`
  為 NULL（match_crawler 既有行為）；匯出器 `_group_matches` 會略過 team_id 為 NULL 的場次並
  log「隊伍未解析，略過匯出」，standings／box_scores 本來就略過。開季名單確定後補 seed
  （新隊給新 team_id；改名則把別名指到舊 team_id），下一次 daily-crawl 的 UPDATE 會自動刷新
  matches 的 team_id，不需手動回填。

### GITHUB_TOKEN push 與 pages.yml

依 GitHub 文件，runner 用 `GITHUB_TOKEN` push `.db` 後，GitHub 不會為該 push 建立其他 workflow 的 run
（防遞迴規則；`workflow_dispatch` 是例外），靜態站不會自動重建。`daily_update.sh` 因此在 push 後自己
`gh workflow run pages.yml --ref main`（workflow 以 `GH_TOKEN: ${{ github.token }}` 帶 token，
並需 `permissions: actions: write`）。實證方式：手動 `gh workflow run daily-crawl.yml` 後，
`gh run list --workflow=pages.yml --json event,headSha` 看該 push 的 sha 只有 `workflow_dispatch`、沒有 `push` 事件的 run。
實證結果（2026-10-15）：手動 `gh workflow run daily-crawl.yml` 成功
（https://github.com/oldyi317/TVL-Analysis/actions/runs/38665660363，比分新增 12 場、Match.ashx 探測 117 場、
登錄 3264 筆），runner 以 bot 身分 push `eca6f9b`；pages.yml 對該 sha **只有一個 `workflow_dispatch` 的 run**
（https://github.com/oldyi317/TVL-Analysis/actions/runs/38666534435，成功部署），沒有 `push` 事件的 run，
證實 GITHUB_TOKEN 的 push 不觸發 pages、腳本點火有效。
