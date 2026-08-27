# Phase 3 設計：賽季限定鍵 + 每日自動化

2026-08-27 brainstorming 定案。目標：開季（2026 年 10 月中）前讓資料管道無人值守——每天自動爬新資料、有變更才 commit/push、觸發 Streamlit Cloud 重佈；並在自動化啟用前根除 week_label 跨賽季碰撞地雷。

## 背景與前提查證

- 連通性實測（2026-08-27，丟棄式 workflow，已刪）：GitHub 雲端 runner 連統計系統
  `114.35.229.141` 完全暢通，但官網 `tvl.ctvba.org.tw` 被 Cloudflare JS challenge
  擋（403 + `cf-mitigated: challenge`，補全瀏覽器 headers 無效）。不繞 Cloudflare。
- 定案：**self-hosted runner**（使用者機器、WSL 內、台灣家用 IP），全管道可跑。
- 使用者機器習慣：白天開機、晚上關機 → 排程訂台灣早上，靠 GitHub job 排隊
  （最多 24 小時）等 runner 上線後補跑，不做喚醒機制。
- 碰撞現況（探勘確認）：`roster_registrations` UNIQUE 為
  `(player_id, team_id, gender, week_label)`，全 DB 無任何賽季欄位；week_label
  直接是 `matches.round_name` 字串，下季同名週次會經 ON CONFLICT DO UPDATE
  覆寫本季登錄。

## 範圍

1. 賽季限定鍵 schema 修正（含 name→pid 對映的 gender 區分）。
2. 每日自動化管道（script + workflow + self-hosted runner）。

**硬約束：第 1 項 merge 且正式 DB 遷移完成後，第 2 項的 workflow 才准啟用。**

### 不做（non-goals）

- 不繞 Cloudflare、不改用混合式雲端架構。
- 換季常數更新維持手動（寫 checklist，不做自動偵測新賽季）。
- 同名同性別球員的碰撞不處理（罕見，記入已知限制）。
- `matches` 表不加 cup_id（`match_date` 天然不跨季；`resolve_week_label` 的
  `week_start_date` 查詢已有 200 天日期隔離）。
- 官網名單 CSV（`crawler` + `db_loader`）不進每日排程（季前作業）。

## 1. Schema：cup_id 進 roster_registrations

- `sql/schema.sql`：`roster_registrations` 加 `cup_id INTEGER NOT NULL`，
  UNIQUE 改為 `(player_id, team_id, gender, cup_id, week_label)`。
- SQLite 改 UNIQUE 需重建表。獨立遷移腳本 `src/etl/migrate_add_cup_id.py`：
  - 防重跑保護（偵測 cup_id 欄位已存在即跳出），模式沿用 `migrate_to_phase2.py`。
  - 遷移前用既有 `backup_db.backup_database` 備份。
  - 新表 → 複製全部資料（既有 2,034 筆補 `cup_id = 21`，**保留原
    registration_id**，`player_match_stats.registration_id` 外鍵不斷）→ drop 舊表
    → rename。過程中 `PRAGMA foreign_keys` 關閉，結束後驗證筆數守恆與孤兒 = 0。

## 2. 寫入與查詢路徑

寫入面（`src/etl/stats_crawler.py`）：

- `upsert_roster_registration`：INSERT 帶 cup_id，ON CONFLICT 鍵加 cup_id。
- `resolve_registration_for_stats`：查詢條件加 cup_id。
- `crawl_all_rosters` 與 `main()` 統計路徑傳入 `CUP_ID`。

查詢面（`src/app/helpers.py` 等）：

- **逐一盤點**所有以 `week_label` 當 join/filter 鍵的查詢，一律加
  `cup_id = EXT_CUP_ID` 限定（app 只呈現當季）。Phase 2 曾漏盤點 box_score
  選單，實作計畫需列成明確清單逐項勾稽。

同名碰撞：

- `build_name_to_pid` 改以 `(正規化姓名, gender)` 為鍵；兩條插入路徑
  （`crawl_all_rosters`、`main()` 統計路徑）同步改，與 `db_loader` 的
  `(name, gender)` 自然鍵對齊。

順帶修正：`helpers.py:135` 硬編 `2026` 改用 `DEFAULT_YEAR`。

## 3. 每日自動化管道

`scripts/daily_update.sh`（邏輯收在 script，可本機直跑、可測試）：

1. `python -m src.etl.match_crawler`
2. `python -m src.etl.stats_crawler --rosters`
3. `python -m src.etl.stats_crawler --incremental`
4. `git status --porcelain data/db/tvl_database.db` 有變更才
   commit（訊息 `chore: 每日自動更新 YYYY-MM-DD`）並 push。
   無變更則靜默結束（淡季每日 no-op，零噪音）。

`.github/workflows/daily-crawl.yml`（薄層）：

- `on: schedule`（cron `0 1 * * *`，= 台灣 09:00）+ `workflow_dispatch`。
- `runs-on: self-hosted`；`concurrency` 防重疊執行。
- 步驟：checkout → 以 runner 機器上的 venv 跑 `scripts/daily_update.sh`。
- 失敗通知靠 GitHub 內建 workflow failure email，不自建。

Runner 一次性安裝（精靈帶使用者走）：

- WSL 內裝 actions-runner，註冊 token 以 `gh api` 代辦，掛 systemd service。
- Windows 登入時自動啟動 WSL 的排程工作一條（runner 常駐進程使 WSL 不被閒置回收）。

**自動 commit/push 授權**：本 spec 明文授權 `daily_update.sh` 的自動
commit/push（僅限 `data/db/tvl_database.db` 的資料更新 commit），作為
「commit 前先問過使用者」規範的既定例外。

## 4. 測試

- `test_schema_v2` UNIQUE 測試更新：同四元組、不同 cup_id 可並存；
  同五元組衝突走 DO UPDATE。
- 遷移測試：舊 schema fixture → 跑遷移 → registration_id 不變、cup_id 全 21、
  筆數守恆、防重跑第二次為 no-op。
- `build_name_to_pid` gender 區分測試（同名不同性別各自對到正確 player_id）。
- 查詢面：既有 tab 查詢測試補「存在他季同名週次資料時不重複計算」的案例。
- `daily_update.sh` 本機 dry-run 驗證；workflow 以 `workflow_dispatch`
  手動觸發實測一輪（含無變更 no-op 與有變更 commit 兩情境）。

## 5. 維運文件

- `docs/ops/season-switch.md`：換季手動 checklist——`EXT_CUP_ID`、
  `SEASON_YEAR_MAP`/`DEFAULT_YEAR`、`match_crawler` game_id 區間預設、
  隊伍對照表（`EXT_TEAM_MAP`/`OPP_SHORT_TO_TEAM`/`TEAM_NAME_SHORT`/`TEAM_ALIAS`）、
  季前作業（`crawler` + `db_loader`）。
- `CLAUDE.md`：加一行指向 checklist；「week_label 跨賽季碰撞」地雷條目改寫為
  已修狀態（改為「換季照 checklist 手動更新常數」）。

## 已知限制

- 同名同性別球員仍會被 `(name, gender)` 鍵合併為同一人。
- 機器連續 24 小時以上未開機會錯過該日排程（GitHub 排隊逾時），下次排程自然補上
  （爬蟲為增量補缺設計，資料不遺失，只是晚更新）。
- 換季常數未更新前，爬蟲對新賽季是 no-op 或抓不到（`EXT_CUP_ID` 仍指舊季），
  屬預期行為，開季照 checklist 更新即可。
