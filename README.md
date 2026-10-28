# TVL 企業排球聯賽數據分析

台灣企業排球聯賽（TVL）男子組與女子組的賽事比分、賽季報名名單、週登錄與單場數據，每日自動爬取寫入 SQLite 資料庫，經 Python 匯出器轉為 JSON 後，以 SvelteKit 靜態站部署於 GitHub Pages，並提供未開打賽事的賽前勝率。

> 🔗 **線上網站**：<https://oldyi317.github.io/TVL-Analysis/>
>
> **免責聲明**：本專案為球迷自製的非官方專案，與中華民國排球協會、企業排球聯賽及各參賽球隊無關。站內所有數據與賽前勝率僅供看球參考，實際紀錄請以聯賽官方公布為準。

## 網站內容

網址結構為 `/<賽季>/<組別>/…`（例如 `/2026-27/men/standings`），頁首可隨時切換賽季（2024-25、2025-26、2026-27）與組別（男子組、女子組）。

| 頁面 | 內容 |
|------|------|
| **首頁** | 最新完賽與下一場賽事卡（含未開打賽事的賽前勝率）、當季戰績摘要 |
| **賽事** | 賽事列表、月份跳轉、逐局比分、雙方球員單場數據、未開打賽事的賽前勝率 |
| **戰績** | 依官方規則排序（勝場 → 積分 → 勝負局商數 → 得失分商數，僅計例行賽），支援手機版展開明細 |
| **球隊** | 當季戰績、賽事清單、最新週登錄（尚無週登錄時顯示賽季報名名單）、跨季曾用名與當季隊徽 |
| **球員** | 公開欄位（姓名、身高、體重、主要位置、背號）、跨季數據、攻擊／攔網／發球三大項統計、進階數據（同組別同主要位置 PR 值、同位置分佈、逐場趨勢） |

位置一律採用兩字簡稱並附英文代碼：**主攻（OH）**、**攔中（MB）**、**副攻（OP）**、**舉球（S）**、**自由（L）**。

## 資料流／架構

```text
TVL 官網 tvl.ctvba.org.tw ──┐  schedule_crawler / crawler + db_loader / match_crawler / match_stats_writer
                            ├──▶ SQLite (data/db/tvl_database.db) ──▶ src.export.site_data ──▶ web/src/lib/data/*.json
外部統計系統（cup ≤21 歷史） ┘  stats_crawler（已自每日更新停用）             │                        │
                                                                 src.models.predict        SvelteKit build ──▶ GitHub Pages
```

1. **ETL 爬蟲與資料庫（SQLite）**：爬蟲從官網抓取賽事列表、賽季報名名單、各局比分、週登錄與單場數據，經清洗規則（只標記與警告異常，不插補、不竄改原始數據）後寫入 `data/db/tvl_database.db`。
2. **賽前勝率推論**：爬蟲完成後，以 `src/models/match_predictor_v2.pkl` 計算所有未開打賽事的主隊賽前勝率，寫入 `match_predictions` 表（ADR 0003）。
3. **JSON 匯出與靜態站建置**：`src/export/site_data.py` 僅使用 Python 標準庫讀取 SQLite，匯出結構化 JSON 至 `web/src/lib/data/`；SvelteKit 前端在建置時驗證 JSON Schema、產生 TypeScript 型別並輸出純靜態網站（ADR 0001）。
4. **檔案入庫策略**：`data/db/tvl_database.db` 與 `src/models/match_predictor_v2.pkl` 刻意 commit 進 git，使雲端建置環境無須安裝爬蟲或機器學習套件即可直接匯出 JSON 並建置靜態站。

### 主要資料表

| 資料表 | 說明 |
|--------|------|
| `seasons` | 賽季定義（`cup_id`、賽季名稱與起訖日） |
| `teams` / `team_aliases` / `team_season_names` | 球隊身分（複合主鍵 `team_id + gender`）、隊名別名解析、各季當季隊名（ADR 0002、ADR 0005） |
| `players` | 球員身分層（僅存姓名、組別、身高、體重等公開欄位，不含背號、位置與生日，見 ADR 0006） |
| `season_rosters` | 賽季報名名單：球員 × 賽季 × 球隊的背號與位置（ADR 0004） |
| `roster_registrations` | 週登錄：球員 × 賽季 × 週次 × 球隊的背號與位置（`source` 為 `match_page` 或 `backfill`） |
| `player_match_stats` | 球員單場數據（攻擊、攔網、發球、接發球、防守、舉球），關聯至 `registration_id` |
| `matches` | 賽事基本資料與各局比分（`game_id + gender` 唯一；所有局比分皆為 `NULL` 代表未開打賽事，見 ADR 0003） |
| `match_predictions` | 未開打賽事的主隊賽前勝率，依 `model_version` 分版，完賽後自動凍結不再重算（ADR 0003） |

## 目錄結構

```text
TVL-Analysis/
├── src/
│   ├── etl/
│   │   ├── schedule_crawler.py    # 官網賽事列表 → matches（未開打賽事日期、時間、場館、對戰）
│   │   ├── crawler.py             # 官網球隊頁 → data/raw/ 賽季報名名單 CSV
│   │   ├── db_loader.py           # 名單 CSV → players 身分層 + season_rosters 賽季報名名單
│   │   ├── match_crawler.py       # 官網賽事頁 → matches 各局比分
│   │   ├── official_box_score.py  # 官網賽事頁單場數據 HTML 解析與驗證
│   │   ├── match_stats_writer.py  # 本季（cup ≥22）有比分且無統計的場次 → roster_registrations + player_match_stats
│   │   ├── stats_crawler.py       # 歷史工具：外部統計系統週登錄與單場數據回填（僅 cup ≤21，ADR 0009）
│   │   └── cleaner.py             # 資料清洗與異常檢查（只標記異常，不插補）
│   ├── export/
│   │   ├── site_data.py           # 靜態站 JSON 匯出器（僅用 Python 標準庫）
│   │   ├── standings.py           # 戰績排序與積分計算
│   │   ├── player_seasons.py      # 球員跨季匯總與主要位置判定
│   │   └── player_advanced.py     # 球員進階數據（PR 值、同位置分佈、逐場趨勢）
│   ├── models/
│   │   ├── features.py            # 球隊層級賽前滾動特徵與真實勝負標籤
│   │   ├── train.py               # TimeSeriesSplit + Optuna + XGBoost 重訓腳本
│   │   ├── predict.py             # 未開打賽事的主隊賽前勝率推論 → match_predictions
│   │   └── match_predictor_v2.pkl # 已訓練之賽果預測模型 artifact（進 git）
│   └── utils/
│       ├── constants.py           # 共用常數（連線設定、賽季常數、headers）
│       ├── db_config.py           # SQLite 連線管理（啟用 foreign_keys）
│       ├── match_status.py        # 賽事完賽／未開打判定條件
│       ├── seasons.py             # 賽季查詢輔助函式
│       ├── teams.py               # 隊名別名解析與球隊身分查詢
│       └── logger.py              # 日誌設定
├── web/                           # SvelteKit 靜態站（含 schema/ JSON Schema 與 ECharts 圖表元件）
├── sql/
│   ├── schema.sql                 # SQLite Schema 唯一來源（冪等 CREATE TABLE IF NOT EXISTS）
│   └── team_seed.sql              # 球隊身分、隊名別名、當季隊名與隊色種子資料
├── scripts/
│   ├── daily_update.sh            # 每日增量更新腳本（排程與本機演練共用）
│   ├── export_snapshot.sh         # 匯出器 JSON 快照比對工具
│   ├── setup_runner.sh            # 自架 GitHub Actions runner 安裝腳本
│   └── runner_loop.sh             # 自架 ephemeral runner 常駐迴圈
├── .github/workflows/
│   ├── daily-crawl.yml            # 每日排程爬蟲（自架 runner）
│   ├── pages.yml                  # 靜態站建置與 GitHub Pages 部署（雲端 runner）
│   └── ci.yml                     # PR 測試守門（pytest、svelte-check、Vitest）
├── data/db/tvl_database.db        # SQLite 資料庫（進 git）
├── tests/                         # Python 單元與合約測試（pytest）
├── docs/
│   ├── adr/                       # 架構決策紀錄（ADR 0001–0009）
│   ├── ops/season-switch.md       # 換季操作檢查表
│   └── design-system.md           # 網站視覺與設計規範
├── CONTEXT.md                     # 領域詞彙表（組別、賽季、週登錄、主要位置等命名規範）
├── AGENTS.md                      # AI Agent 開發慣例與地雷清單
└── requirements.txt               # Python 依賴（爬蟲、模型重訓、測試共用）
```

## 本機開發

### 環境需求與安裝

本專案使用 **Python 3.14** 與 **Node.js 24**。所有檔案行尾一律使用 **LF**（WSL `/mnt/d` 環境下請留意避免工具轉為 CRLF）。

```bash
# 1. 安裝 Python 依賴（爬蟲、模型重訓、pytest）
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. 安裝前端依賴
cd web && npm ci && cd ..
```

### 常用指令

```bash
# 靜態站（匯出器從 repo 根目錄跑）
python -m src.export.site_data                   # 匯出 JSON 到 web/src/lib/data/
cd web && npm run dev                            # 以下 npm 指令都在 web/ 下執行
npm run build                                    # 建置到 web/build/
npm run check                                    # 型別檢查
npm run test:unit                                # 單元測試
npm run test:e2e                                 # Playwright smoke（需先 build）

# 爬蟲（皆從 repo 根目錄以 module 方式執行）
python -m src.etl.crawler                        # 官網球員名單 → data/raw/
python -m src.etl.match_crawler                  # 官網各局比分 → matches 表
python -m src.etl.match_stats_writer             # 本季（cup≥22）有比分且無統計的場次 → player_match_stats（官網單場數據，spec #75；先跑 match_crawler）
python -m src.etl.stats_crawler --rosters --cup-id 20   # 歷史工具：外部統計系統週登錄回填，僅 cup ≤21（ADR 0009；cup 22 起拒絕執行）
python -m src.etl.stats_crawler --incremental --cup-id 20   # 歷史工具：外部統計系統單場數據回填，僅 cup ≤21（先跑 match_crawler）
python -m src.etl.db_loader                      # roster CSV → players 身分層 + season_rosters 賽季報名名單
python -m src.etl.db_loader --csv assets/tvl_20_player_list/tvl20_rosters.csv --cup-id 20   # 歷史名單回填（圖檔轉錄 CSV，指定賽季）
bash scripts/daily_update.sh                     # 每日增量更新（GitHub Actions 排程同款；DRY_RUN=1 可演練）
python -m src.models.train                       # 重訓賽果預測模型 → match_predictor_v2.pkl
python -m src.models.predict                     # 未開打賽事的主隊賽前勝率 → match_predictions 表（daily_update 內含）
```

## 每日自動更新與部署（GitHub Actions）

專案透過三個 GitHub Actions workflow 協作（見 `.github/workflows/`）：

1. **每日增量爬蟲（`daily-crawl.yml`）**：
   - 每日台灣時間 09:23（`01:23 UTC`）排程觸發，亦可手動 `workflow_dispatch`。
   - 因 TVL 官網有 Cloudflare 防護，雲端 runner 無法連線，故運行於自架 runner（`[self-hosted, tvl]`）。
   - 執行 `scripts/daily_update.sh`：依序跑官網賽事列表（`schedule_crawler`）、賽季報名名單（`crawler` + `db_loader`）、各局比分（`match_crawler`）、官網單場數據與週登錄（`match_stats_writer`）以及未開打賽事賽前勝率（`src.models.predict`）。
   - 腳本透過 SQLite `iterdump()` 計算資料庫邏輯內容 SHA-256 雜湊（忽略 `sqlite_sequence` 自增序號雜訊）；確認有實質資料變更才 commit 並 push `data/db/tvl_database.db`，隨後主動觸發 `pages.yml` 部署靜態站。
2. **靜態站建置與部署（`pages.yml`）**：
   - 當 `main` 分支有 push 或由 `daily_update.sh` 觸發時，在 GitHub 雲端 runner（`ubuntu-latest`）執行。
   - 不安裝 `requirements.txt`，直接以 Python 3.14 標準庫執行匯出器（`src.export.site_data`）從 repo 內的 `.db` 產出 JSON，接著在 `web/` 下安裝前端依賴並建置靜態站（含 Ajv JSON Schema 驗證與型別產生），最後將 `web/build` 部署至 GitHub Pages。
3. **PR 持續整合檢查（`ci.yml`）**：
   - 於 Pull Request 時在雲端 runner 執行 Python 測試（pytest）以及前端 JSON 匯出、型別檢查與單元測試。

## 賽果預測模型簡述

未開打賽事的**賽前勝率**由 `src/models/` 下的機器學習管線產生（ADR 0003）：

- **模型與訓練（`train.py`）**：採用 **XGBoost** 分類器（`match_predictor_v2.pkl`），搭配 5 折時間序列交叉驗證（`TimeSeriesSplit`）與 **Optuna** 進行超參數搜尋。
- **賽前滾動特徵（`features.py`）**：以球隊（`team_id + gender`）為單位，計算賽前已知的近 3 場與近 5 場滾動平均指標（先 `shift(1)` 避免資料洩漏，排除黃金決勝局，跨季連續不重置）：
  - 攻擊成功率（`ASR_roll3`、`ASR_roll5`）
  - 接發球到位率（`GP_pct_roll3`、`GP_pct_roll5`）
  - 防守到位率（`DIG_pct_roll3`、`DIG_pct_roll5`）
  - 每局攔網得分（`BLK_per_set_roll3`、`BLK_per_set_roll5`）
  - 發球得分率（`ACE_pct_roll3`、`ACE_pct_roll5`）
  - 當前連勝／連敗場數（`win_streak`）
- **雙隊勝率正規化與凍結規則（`predict.py`）**：模型分別推估主、客隊各自的勝率 $p_{\text{home}}$ 與 $p_{\text{away}}$，再正規化為主隊賽前勝率 $p_{\text{home}} / (p_{\text{home}} + p_{\text{away}})$，使兩隊勝率總和為 100%。兩隊皆須具備至少 1 場歷史單場數據才會計算；賽事一旦開打完賽（任一局比分非 `NULL`），該場的賽前勝率即凍結保留，不再覆寫。

## 資料來源與限制聲明

- **資料來源演進（ADR 0009）**：
  - **2026-27 賽季（cup 22）起**：完全以 [TVL 企業排球聯賽官網](https://tvl.ctvba.org.tw) 為唯一來源（包含賽事列表、各局比分、賽季報名名單、週登錄與單場數據）。由於官網會將歷史場次的背號重綁至現行名單，單場數據僅於比賽當日有比分且尚無統計時寫入一次，寫入後不再覆寫。
  - **2024-25、2025-26 賽季（cup 20、21）**：單場數據與週登錄來自舊版外部技術統計系統，既有歷史資料原樣保留於資料庫中；外部統計系統爬蟲（`stats_crawler`）已自每日更新停用。
- **資料口徑與已知限制**：
  - **出賽局數（`sets_played`）**：官網單場數據未提供球員個人出賽局數，因此 2026-27 賽季起 `sets_played` 為 `NULL`（代表未知而非 0），網站不以整場局數替代個人局數，個人「每局攔網」等以個人局數為分母的指標在該季顯示為無數值（僅賽果預測模型的球隊層級特徵會退回該場實際局數）；2024-25 與 2025-26 賽季則有完整個人出賽局數。
  - **未上場球員紀錄**：官網單場數據會列出整場各項統計皆為 `0-0` 的球員，匯入時仍會建立週登錄，但不產生 `player_match_stats` 單場數據列；因此跨賽季的「出賽場數」口徑不同，不宜直接跨季比較。
  - **小樣本與缺值原則**：遵循「只標記異常、不插補、不竄改原始數據」原則。分母為 0 時比率型指標為無數值（非 `0%`）；進階數據的 PR 值與每局攔網設有樣本門檻（已知局數需達 5 局、未知局數需達 2 場），不足時顯示「—」，避免小樣本偏差。
  - **開季賽前勝率**：新賽季開打初期，賽前勝率的滾動特徵會延續上一賽季末的近況推估；若有新成立或無歷史單場數據的球隊則顯示資料不足。

## 授權

- **程式碼**：採用 MIT License 授權，詳見 [LICENSE](LICENSE)。
- **資料與商標**：比賽與球員公開資料整理自 [TVL 企業排球聯賽官網](https://tvl.ctvba.org.tw)；站內所載之隊徽與球隊名稱商標均屬中華民國排球協會、企業排球聯賽及各球團所有，本專案僅作賽事資訊識別之用。
