# daily-crawl 維運

每日更新跑在 GitHub 雲端 runner（workflow 的 `cloud` job，`ubuntu-latest`，Python 版本與 CI 相同，照 CI 裝 `requirements.txt` 並用 pip 快取），整包預設、寫入；資料庫有變更才 commit、push，再啟動網站建置（pages.yml）。
準時觸發由比分服務（部署在 Cloudflare Workers，名稱 `tvl-score`）負責，不靠筆電開機；筆電 runner 只在備援期用。

## 觸發與手動執行

- 觸發來源：(1) 比分服務（下一節）每天台北 09:23 啟動整包；當天有賽事時，另在 15:30、18:30、21:15 各啟動一次精簡版（只抓當天場次的比分與單場數據，再重算賽前勝率）。都跑在雲端、寫入，有實質變更才 commit 並部署網站，發送服務讀到新賽果就發賽果摘要，所以比賽日的賽果當天進站。
  (2) workflow 自己的 schedule（cron 01:23 UTC）留作備援，實際常晚 4–7 小時，同樣在雲端整包、寫入。
  一天跑多次無害，`.db` 沒有實質變更就不 commit；執行中的 run 不會被取消，後來的排隊等它跑完；排隊的位置只有一個，再來新的會取代還在排隊的那一個。
- 筆電 timer `tvl-crawl-trigger` 已在 #294 停用，`setup_runner.sh` 不再安裝它。比分服務故障、一時修不好時，可以暫時跑 `bash scripts/setup_crawl_trigger.sh` 重裝當臨時觸發（筆電要開機），修好後 `systemctl --user disable --now tvl-crawl-trigger.timer` 停掉；這支腳本在 #296 與筆電 runner 一起退役。
- 手動觸發的輸入：`mode`（`full`／`slim`，預設 `full`）、`runner`（`cloud`／`laptop`，預設 `cloud`）、`dry_run`（預設否；為真時不 commit、不 push、不啟動網站建置）。
  例：`gh workflow run daily-crawl.yml -f mode=slim -f dry_run=true`，測分支上的改動加 `--ref <分支>`。
- 雲端取樣（#291）已結束：11/19～11/22 四次取樣（含週六、週日比賽日）全部成功，官網擋的是 UA，爬蟲的瀏覽器 UA 在雲端回 200，雲端 runner 進得去官網。

## 比分服務（`tvl-score`）

程式、判斷規則與設定值在 `score/`，設定值的表見 `score/README.md`。服務不存狀態、沒有網址（`workers_dev: false`），cron 每分鐘呼叫一次，由程式依台北時間決定要不要呼叫 GitHub 啟動這個 workflow（帶 `mode`；`DRY_RUN` 為真時另帶 `dry_run: true`）。

### 部署

不做自動部署，GitHub 不放 Cloudflare 金鑰；由維護者在筆電上手動部署，從已合進 main 的 commit 做。

1. 在 `score/` 底下 `npm ci`（Node 24，`score/` 有自己的 `package.json`）。之後的 wrangler 都用 `node_modules/.bin/wrangler`，版本由 `package.json` 鎖。
2. wrangler 登入 Cloudflare：`node_modules/.bin/wrangler login`；`node_modules/.bin/wrangler whoami` 印得出帳號就是登入有效。與發送服務同一個帳號。
3. `node_modules/.bin/wrangler deploy`。輸出要看到 `schedule: * * * * *` 與 `wrangler.json` 裡的各項設定值。
4. 第一次部署、或 token 要換時：維護者自己在另一個終端機跑 `bash scripts/score_token_wizard.sh`（見下一節）。secret 要排在第一次部署之後存，服務還沒部署時 wrangler 會問要不要先建同名服務。

`DRY_RUN` 要改，改 `score/wrangler.json` 開 PR，合併後重新部署；不要在 Cloudflare 後台改，下一次部署會被設定檔蓋回去。

### GitHub token

- secret 名稱 `GITHUB_TOKEN`，只存在 Cloudflare 上 `tvl-score` 的 secret 裡，不放 GitHub、不存在筆電上。
- 種類與權限：repo 擁有者 oldyi317 的 fine-grained token，Repository access 只選 `TVL-Analysis`，Repository permissions 只開 Actions: Read and write（啟動 workflow 的端點要求的權限，依據見 PR #300）。
- 產生與存放：`bash scripts/score_token_wizard.sh` 會開 GitHub 產生 token 的頁面（網址已預填名稱、擁有者、90 天到期與 Actions 讀寫；repo 要手動選），維護者把 token 貼進腳本（輸入不顯示）後，腳本用它啟動一次「精簡、不寫入」的 run 驗證權限，通過才以 `wrangler secret put GITHUB_TOKEN` 存進 `tvl-score`。token 不寫檔、不印出，也不要貼進 agent 的對話。
  帶 Administration 權限的 token（例如筆電 runner 那一把）腳本會拒收；GitHub 回 403 時，腳本會印出 token 屬於哪個帳號與這個端點要求的權限。
- 到期與更換：GitHub 會在到期前寄信給擁有者。收到信就重跑 `bash scripts/score_token_wizard.sh` 產生新 token 並覆蓋 secret（不必重新部署），確認之後的 09:23 有 run，再到 GitHub 的 Fine-grained tokens 頁刪掉舊的。目前這把的到期日記在 #294。
- 過期或被撤銷的徵兆：09:23 沒有新的 `workflow_dispatch` run，比分服務的日誌出現 `GitHub workflow dispatch 回應 401`。

### 部署後驗證

| 看什麼 | 怎麼看 | 預期 |
| --- | --- | --- |
| secret 在 | `node_modules/.bin/wrangler secret list` | 看得到名稱 `GITHUB_TOKEN`，看不到值是正常的 |
| 每分鐘有呼叫 | `node_modules/.bin/wrangler tail`（看完 Ctrl+C） | 每分鐘一筆定時呼叫，outcome 都是 `ok`；觸發失敗的那分鐘是 `exception` |
| 準時啟動 | `gh run list --workflow=daily-crawl.yml --event workflow_dispatch --limit 5 --json createdAt,displayTitle,url` | 每天台北 09:23 後幾分鐘內有一筆；比賽日 15:30、18:30、21:15 後也各有一筆 |

### 暫時改時間點驗證精簡版

非比賽日要驗精簡版時，用 `--var` 暫時覆寫，不改設定檔：

```bash
node_modules/.bin/wrangler deploy --var SLIM_TIMES:HH:MM --var FEED_URL:https://vstat.tw/push/none.json
```

`HH:MM` 填幾分鐘後的台北時間；賽事檔指到不存在的網址時讀檔失敗，照規則當作有賽事。看到那一分鐘的 run 之後，立刻不帶 `--var` 再 `node_modules/.bin/wrangler deploy` 一次，設定就回到 `wrangler.json`（沒加 `--keep-vars` 時，部署會以設定檔的值為準）。

## 備援期間改用筆電 runner

筆電 runner 在 #296 退役前保留作備援。命令：`gh workflow run daily-crawl.yml -f runner=laptop`（可加 `-f mode=slim`、`-f dry_run=true`），job 跑在筆電的 `[self-hosted, tvl]` runner，筆電要開機、runner 要在線。

- 該用：雲端 run 因為官網擋下而失敗時（官網回非 200，含 Cloudflare 的 403 封鎖頁；比分爬蟲遇到非 200 會以失敗結束、不寫入），當天手動改用筆電補跑。
- 不該用：其他失敗（程式錯誤、依賴裝不起來）先修程式，換 runner 只會原樣再失敗一次。

## 筆電 runner 安裝與運作（ephemeral；僅備援期用）

- 安裝或換機器：在 WSL 內從 repo 根目錄跑 `bash scripts/setup_runner.sh`（需 gh 已登入、sudo、WSL 已啟用 systemd）。
  取註冊 token 需要 repo admin，個人帳號 repo 的 collaborator 給不了 admin，所以用 owner（oldyi317）的 fine-grained PAT：
  只選 `TVL-Analysis`、Repository permissions 只開 Administration: Read and write，存成 `~/.config/tvl-runner/token`（chmod 600）。
  迴圈每輪重讀這個檔，PAT 到期換新只要覆寫檔案，不用重啟；PAT 只傳給 `gh`，不進 job 環境，但 job 與迴圈同一個使用者，讀得到這個檔。
  腳本會移除舊的 svc.sh 常駐服務與持久註冊，把 `scripts/runner_loop.sh` 複製到 `~/actions-runner/`，
  並安裝 systemd service `tvl-runner`（enable 後重啟，隨 WSL 開機自啟；WSL 開機靠 Windows 工作排程器，見腳本步驟 5）。
- 運作：`tvl-runner` 跑迴圈「`gh api` 取註冊 token → `config.sh --ephemeral --replace` 註冊（名稱 `tvl-wsl`、label `tvl`）
  → `run.sh` 接一個 job → job 結束 runner 自動註銷 → 回到開頭」。每輪先清掉 `_work` 與殘留的本機註冊檔；但 job 仍以同一個使用者執行，`$HOME`（venv、gh 登入）不會重置。
  取 token、註冊失敗，或 `run.sh` 不到 60 秒就結束（它對 listener 異常也回 0）時退避重試（30 秒起倍增，上限 15 分鐘）。
  service 的 PATH 固定為 `/usr/local/bin:/usr/bin:/bin`，job 也繼承這個 PATH；daily_update 的 python 走 venv 絕對路徑，不受影響。workflow 的 `runs-on: [self-hosted, tvl]` 不變。
- job 守門：公開 repo 的 fork PR 能改 workflow，把 job 指到 `tvl` label。迴圈以 `ACTIONS_RUNNER_HOOK_JOB_STARTED` 掛上
  `~/.config/tvl-runner/runner_job_gate.sh`（`setup_runner.sh` 步驟 4 從 `scripts/runner_job_gate.sh` 安裝），在 job 任何步驟之前執行：
  只放行本 repo 的 `schedule` 與 `workflow_dispatch`，其餘事件直接砍掉 `Runner.Worker`，job 以失敗收場，迴圈隨後退避重試。
  找不到守門腳本時迴圈不取 token、不註冊。守門要重跑 `setup_runner.sh` 才生效；之後要讓其他事件的 workflow 跑在這台 runner，得先改守門腳本。
- 改了 `runner_loop.sh` 要重跑 `setup_runner.sh`，service 跑的是 runner 目錄裡的副本；重跑會停掉 service，正在跑的 job 會被中斷，挑沒有 daily-crawl 在跑的時候做。
- 查狀態：`systemctl status tvl-runner`、`journalctl -u tvl-runner -f`；GitHub 端
  `gh api repos/oldyi317/TVL-Analysis/actions/runners --jq '.runners[] | {id, name, status}'`，每跑完一個 job，`id` 會換成新的。

## 失敗時怎麼看

- **先看 run log 哪一步失敗，再看程式**：`gh run list --workflow=daily-crawl.yml --limit 10 --json databaseId,createdAt,event,conclusion,url`，
  `gh run view <run_id> --log-failed` 看失敗步驟。依序判斷：是不是官網回非 200 或封鎖頁（比分爬蟲遇非 200 會失敗、不寫入；是的話當天改用筆電補跑，見上）、
  `pip install -r requirements.txt` 是否裝失敗（依賴問題）、其餘是程式錯誤（先修程式）。
- **該有的 run 沒出現**（09:23 或比賽日三個時間點後幾分鐘內沒有 `workflow_dispatch` run）：問題在比分服務。在 `score/` 底下 `node_modules/.bin/wrangler tail` 看那一分鐘的 outcome，或到 Cloudflare 後台 tvl-score 的 Logs 找錯誤訊息；`回應 401` 是 token 過期或被撤銷（見「GitHub token」），其他狀態碼先看 GitHub 狀態頁。當天先手動 `gh workflow run daily-crawl.yml` 補跑。
- 平日排程 run 的建立時間比 cron 晚 4.5–7 小時是 GitHub 端延遲（#196），不是故障。
- 以下是「改用 laptop 時才需要」的筆電 runner 排查：runner 在使用者筆電的 WSL，關機時 job 排隊 24 小時後 cancelled（annotation「exceeded the maximum
  execution time while awaiting a runner」）；跑到一半 WSL 被關則是「self-hosted runner lost communication」。先查 runner 在不在線再看程式。
- 取消或失敗原因：`gh api repos/oldyi317/TVL-Analysis/check-runs/<job_id>/annotations`；對照 runner 機器的
  `journalctl --list-boots` 與 job 的 started／completed 時間。筆電睡眠不算新 boot，`--list-boots` 看不出來，
  listener 的 `_diag` 日誌會跨越睡眠期持續。runner 迴圈日誌：`journalctl -u tvl-runner`。
- Actions 頁與 `gh run view <id> --log-failed` 對自架 runner（laptop job）的 run 常是空的；改抓整份日誌：
  `gh api repos/oldyi317/TVL-Analysis/actions/runs/<run_id>/logs > run.zip`，用 `python -I -c "import zipfile;zipfile.ZipFile('run.zip').extractall('run')"`
  解開（runner 機器沒裝 unzip），步驟輸出在 `run/0_update.txt`。
- runner 本機 `~/actions-runner/_diag/Worker_*.log` 只有程序起訖與 exit code，沒有步驟 stdout。
- 本機重現：**不要在 `/mnt/d/TVL-Analysis` 直接跑 `DRY_RUN=1 bash scripts/daily_update.sh`**，
  腳本結尾會 `git checkout -- data/db/tvl_database.db`，會把工作樹裡未 commit 的 .db 改動洗掉；
  先 `git clone /mnt/d/TVL-Analysis <暫存目錄>` 再在 clone 裡跑。

## 2026-09-29 ～ 2026-10-14 連續失敗紀錄（issue #23）

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

## GITHUB_TOKEN push 與 pages.yml

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
