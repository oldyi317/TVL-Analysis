# daily-crawl 維運（自架 runner）

## runner 安裝與運作（ephemeral）

- 安裝或換機器：在 WSL 內從 repo 根目錄跑 `bash scripts/setup_runner.sh`（需 gh 已登入、sudo、WSL 已啟用 systemd）。
  取註冊 token 需要 repo admin，個人帳號 repo 的 collaborator 給不了 admin，所以用 owner（oldyi317）的 fine-grained PAT：
  只選 `TVL-Analysis`、Repository permissions 只開 Administration: Read and write，存成 `~/.config/tvl-runner/token`（chmod 600）。
  迴圈每輪重讀這個檔，PAT 到期換新只要覆寫檔案，不用重啟；PAT 只傳給 `gh`，不進 job 環境，但 job 與迴圈同一個使用者，讀得到這個檔。
  腳本會移除舊的 svc.sh 常駐服務與持久註冊，把 `scripts/runner_loop.sh` 複製到 `~/actions-runner/`，
  並安裝 systemd service `tvl-runner`（enable 後重啟，隨 WSL 開機自啟；WSL 開機靠 Windows 工作排程器，見腳本步驟 6）。
- 每日觸發：`setup_runner.sh` 步驟 5 會跑 `scripts/setup_crawl_trigger.sh`（也可單獨跑，不需 sudo、需 linger），裝 user 層級 timer `tvl-crawl-trigger`：
  每天台北 09:23 `gh workflow run daily-crawl.yml`，`Persistent=true` 讓未開機錯過的那次在開機後補跑（開機當下網路可能未通，觸發失敗每 2 分鐘重試、最多 5 次）。workflow 自己的 schedule 實際常晚 4–7 小時（#196），只當備援；
  一天跑兩次無害，`.db` 沒有實質變更就不 commit。查看：`systemctl --user list-timers tvl-crawl-trigger.timer`、`journalctl --user -u tvl-crawl-trigger`。
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

- **先看 runner 在不在線，再看程式**：runner 在使用者筆電的 WSL，關機時 job 排隊 24 小時後 cancelled（annotation「exceeded the maximum
  execution time while awaiting a runner」）；跑到一半 WSL 被關則是「self-hosted runner lost communication」。週末筆電通常不開，
  週六 run 取消、週日 run 拖到週一開機才被接走，屬預期（#196）。
- 取消或失敗原因：`gh api repos/oldyi317/TVL-Analysis/check-runs/<job_id>/annotations`；對照 runner 機器的
  `journalctl --list-boots` 與 job 的 started／completed 時間。筆電睡眠不算新 boot，`--list-boots` 看不出來，
  listener 的 `_diag` 日誌會跨越睡眠期持續。runner 迴圈日誌：`journalctl -u tvl-runner`；本機觸發 timer：`journalctl --user -u tvl-crawl-trigger`。
- 平日排程 run 的建立時間比 cron 晚 4.5–7 小時是 GitHub 端延遲（#196），不是 runner；主要觸發已改由本機 timer。
- Actions 頁與 `gh run view <id> --log-failed` 對自架 runner 的 run 常是空的；改抓整份日誌：
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
