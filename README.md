# TVL 企業排球聯賽數據分析

台灣企業排球聯賽（TVL）男子組、女子組的賽果、週登錄名單與逐場技術統計，每日自動爬取進 SQLite，
匯出成 JSON 後以 SvelteKit 靜態站呈現在 GitHub Pages，並附 XGBoost 賽前勝率。

> 🔗 **網站**：<https://oldyi317.github.io/TVL-Analysis/>
>
> 球迷自製的非官方專案，與聯賽及各球隊無關，數據僅供參考，請以官方公布為準。

## 網站內容

網址結構為 `/<賽季>/<組別>/…`，頁首可切換賽季與男女組。

| 頁面 | 內容 |
|------|------|
| 首頁 | 最新比分卡、戰績摘要 |
| 賽事 | 賽程列表、月份跳轉、逐局比分、單場技術統計、未開打賽事的賽前勝率 |
| 戰績 | 依官方規則排序，手機精簡版可展開 |
| 球隊 | 當季戰績、賽事清單、當週登錄名單 |
| 球員 | 基本資料、攻擊／攔網／發球三大項、進階數據（PR 值、同位置分佈、逐場趨勢） |

## 資料流

```
官網 tvl.ctvba.org.tw ──┐  schedule_crawler / match_crawler / match_stats_writer / crawler
                        ├──▶ SQLite data/db/tvl_database.db ──▶ src.export.site_data ──▶ web/src/lib/data/*.json
外部統計系統（cup ≤21 歷史）┘  stats_crawler（已停用）                   │                        │
                                                       src.models.predict        SvelteKit build ──▶ GitHub Pages
```

- **每日更新**（`.github/workflows/daily-crawl.yml`）：台灣時間 09:23 在自架 runner 跑
  `scripts/daily_update.sh`，依序抓賽事列表、賽季報名名單、各局比分、週登錄、逐場技術統計，
  算賽前勝率，`.db` 有實質變更才 commit/push，再觸發 pages.yml。官網有 Cloudflare，雲端 runner
  進不去，所以爬蟲只能在自架 runner 跑。
- **靜態站**（`.github/workflows/pages.yml`）：push 到 `main` 時在雲端 runner 匯出 JSON、建置、部署。
  匯出器只用 Python 標準庫，直接讀 repo 內的 `.db`，不裝任何 pip 套件。
- `data/db/tvl_database.db` 與 `src/models/match_predictor_v2.pkl` 因此是刻意 commit 進 git 的。

## 專案結構

```
TVL-Analysis/
├── src/
│   ├── etl/
│   │   ├── schedule_crawler.py   # 官網賽程列表 → matches（未開打場次的基本資料）
│   │   ├── match_crawler.py      # 官網單場頁 → matches 各局比分
│   │   ├── crawler.py            # 官網球隊頁 → data/raw/ 賽季報名名單 CSV
│   │   ├── db_loader.py          # 名單 CSV → players 身分層 + season_rosters
│   │   ├── stats_crawler.py      # 外部統計系統 → 週登錄與單場數據（歷史補抓工具，cup ≤21；ADR 0009）
│   │   ├── cleaner.py            # 清洗規則（只標記異常，不插補）
│   │   └── backup_db.py          # 遷移前備份 .db
│   ├── export/
│   │   ├── site_data.py          # 靜態站 JSON 匯出器
│   │   └── player_advanced.py    # 球員進階數據純函式
│   ├── models/
│   │   ├── features.py           # 球隊層級特徵（賽前可知的滾動指標）與勝負標籤
│   │   ├── train.py              # Optuna 調參 + XGBoost 重訓 → match_predictor_v2.pkl
│   │   └── predict.py            # 未開打賽事的主隊賽前勝率 → match_predictions
│   └── utils/
│       ├── constants.py          # 共用常數（外部系統設定、賽季常數、headers）
│       ├── db_config.py          # SQLite 連線（開 foreign_keys）
│       ├── seasons.py / teams.py # 賽季、球隊身分解析
│       └── logger.py
├── web/                          # SvelteKit 靜態站（schema/ 為 JSON schema 唯一來源）
├── scripts/
│   ├── daily_update.sh           # 每日增量更新（DRY_RUN=1 可演練）
│   ├── setup_runner.sh           # 自架 runner 安裝精靈（ephemeral + systemd）
│   └── runner_loop.sh            # runner 迴圈腳本
├── sql/
│   ├── schema.sql                # Schema 唯一來源（冪等，不清空資料）
│   └── team_seed.sql             # 球隊身分、別名、隊色種子
├── data/db/tvl_database.db       # 進 git
├── tests/                        # pytest（CI 不跑，本機要跑）
├── docs/
│   ├── adr/                      # 架構決策紀錄
│   ├── ops/season-switch.md      # 換季 checklist
│   └── design-system.md          # 網站設計規範
├── CONTEXT.md                    # 詞彙表（組別、賽季、週登錄、主要位置……）
└── requirements.txt
```

## 資料庫

| 資料表 | 說明 |
|---|---|
| `seasons` | 賽季（`cup_id` 即屆數，含名稱與起訖日） |
| `teams` / `team_aliases` / `team_season_names` | 球隊身分（複合主鍵 `team_id + gender`）、官網各種隊名寫法的對照、歷史賽季隊名（ADR 0005） |
| `players` | 球員身分層（姓名、性別、身高、體重；不含背號、位置、隊伍） |
| `season_rosters` | 賽季報名名單：球員 × 賽季 × 隊伍的背號與位置（ADR 0004） |
| `roster_registrations` | 週登錄：球員 × 週次 × 隊伍的背號與位置；`source` 為 `match_page`（實際出賽名單）或 `backfill` |
| `player_match_stats` | 逐場技術統計（攻擊、攔網、發球、接發、防守、舉球），掛在 `registration_id` 下 |
| `matches` | 賽事（`game_id + gender` 唯一）：日期、時間、場館、輪次、兩隊、各局比分 |
| `match_predictions` | 主隊賽前勝率，依 `model_version` 分版；開打後凍結（ADR 0003） |

## 本機開發

### 環境

需要 Python 3.14 與 Node 24（與 `.devcontainer` 和 pages.yml 相同）。

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # 爬蟲、模型與 pytest 全在這一份
cd web && npm ci && cd ..
```

### 爬蟲（皆從 repo 根目錄以 module 執行）

```bash
python -m src.etl.schedule_crawler               # 賽程列表 → matches
python -m src.etl.match_crawler                  # 各局比分 → matches（match_stats_writer 依賴此表最新）
python -m src.etl.official_box_score --dry-run --game-id 281 --gender F  # 官網單場數據解析與閘門檢查（不讀寫 DB）
python -m src.etl.match_stats_writer             # 本季有比分且無統計的場次 → player_match_stats（官網單場數據，spec #75）
python -m src.etl.match_stats_writer --force-partial 281 --gender F   # 有未解析列的場次仍寫入其餘球員（人工決定）
python -m src.etl.crawler                        # 賽季報名名單 → data/raw/
python -m src.etl.db_loader                      # 名單 CSV → players + season_rosters

# 歷史工具（外部統計系統，僅 cup ≤21；已自每日更新停用，cup 22 起拒絕執行，見 ADR 0009）
python -m src.etl.stats_crawler --rosters --cup-id 20        # 週登錄 → roster_registrations
python -m src.etl.stats_crawler --incremental --cup-id 20    # 單場數據（增量；不加 --incremental 為全量補缺）
python -m src.etl.db_loader --csv assets/tvl_20_player_list/tvl20_rosters.csv --cup-id 20

# 整套每日流程演練（不 commit）
DRY_RUN=1 bash scripts/daily_update.sh
```

所有爬蟲都是「補缺不清表」：schema 冪等、`db_loader` 以自然鍵 upsert、`match_stats_writer`
只寫兩隊皆無統計的場次且寫入後不覆寫、歷史工具 `stats_crawler` 以
`match_date + is_golden_set` 逐球員去重。遇到週登錄名單外的球員會插入身分層並建 `source='backfill'`
的登錄（背號、位置皆 NULL）。

### 模型

```bash
python -m src.models.train      # Optuna 調參重訓 → src/models/match_predictor_v2.pkl
python -m src.models.predict    # 未開打賽事 → match_predictions（daily_update.sh 內含）
```

artifact 必須含 `version` 與 `feature_cols`，`tests/test_prediction_artifact.py` 鎖住合約，
重訓一律走 `train`，不要手動 pickle。兩隊各需至少一場有技術統計的歷史賽事，否則該場不寫列。

### 本機跑網站

```bash
python -m src.export.site_data   # 1. 匯出 JSON 到 web/src/lib/data/（不進 git）
cd web
npm run dev                      # 2. dev server（先驗證 JSON 符合 web/schema 並產生型別）
npm run build                    # 3. 建置到 web/build/
```

重構匯出器時的行為鎖：改前 `bash scripts/export_snapshot.sh save`，改後 `bash scripts/export_snapshot.sh diff`（逐檔 cmp，有差異以非 0 結束；基準放 `.snapshots/export`，不進 git）。

### 測試

```bash
pytest                           # Python：爬蟲解析、匯出器、模型合約
cd web
npm run check                    # 型別檢查
npm run test:unit                # Vitest
npm run test:e2e                 # Playwright 手機 smoke（會先 build；第一次先 npx playwright install chromium）
```

## 營運

- **自架 runner**：`bash scripts/setup_runner.sh` 在 WSL 內安裝 ephemeral runner 並以 systemd 常駐，
  每跑完一個 job 自動註銷重註冊。機器未開機時 job 排隊等 runner 上線。
- **換季**：新賽季開打前照 `docs/ops/season-switch.md` 逐項執行（`seasons` 列等是手動更新，
  未更新前每日排程照跑但抓不到新季資料）。
- **行尾一律 LF**（`.gitattributes` 強制）：repo 在 WSL 的 `/mnt/d` 上，Windows 工具容易轉成 CRLF。

## 位置代號

依 `CONTEXT.md` 詞彙表，一律用兩字簡稱，空間足夠時附英文代碼。

| 簡稱 | 代碼 |
|---|---|
| 主攻 | OH (Outside Hitter) |
| 攔中 | MB (Middle Blocker) |
| 副攻 | OP (Opposite) |
| 舉球 | S (Setter) |
| 自由 | L (Libero) |

## 資料品質原則

- 只標記與警告異常，不插補、不竄改原始數據
- 缺失值保留為 `NULL`/`None`；比率型指標分母為 0 時沒有數值，不是 0%
- 比率型指標分母 < 10 時前端顯示 N/A，避免小樣本偏差
- 日期統一為 `YYYY-MM-DD`，數值欄位去除單位字串

## 技術堆疊

- **爬蟲**：requests、BeautifulSoup
- **資料處理**：pandas、NumPy、SQLite
- **模型**：XGBoost、scikit-learn、Optuna
- **前端**：SvelteKit（adapter-static）、ECharts、Vitest、Playwright
- **部署**：GitHub Actions（自架 runner 爬蟲 + 雲端 runner 建置）、GitHub Pages

## 更多文件

- `CONTEXT.md`：領域詞彙表，UI 文案與程式命名的依據
- `docs/adr/`：架構決策紀錄（靜態站取代 Streamlit、賽前勝率由爬蟲計算、賽季報名名單獨立成表……）
- `docs/design-system.md`、`web/DESIGN.md`：網站設計規範與 token
- `CLAUDE.md`：給 AI agent 的開發慣例與地雷清單

## 授權與資料來源

- **程式碼**：MIT 授權，見 [LICENSE](LICENSE)。
- **資料**：整理自 [TVL 企業排球聯賽官網](https://tvl.ctvba.org.tw) 與其技術統計系統。
- **商標**：隊徽與球隊名稱之商標屬各球隊與聯賽所有，本專案僅為識別之用。

## 開發者

**小易** — 數據工程與網站開發
