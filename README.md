# TVL 企業排球聯賽數據分析儀表板

針對台灣企業排球聯賽（TVL）男女組賽事的數據爬取、清洗、儲存與視覺化分析系統。

## 功能特色

- **資料爬取**：自動抓取 TVL 官網的球隊名單、球員資料與逐場技術統計（支援增量更新）
- **資料清洗與載入**：標準化位置名稱、日期格式與數值欄位，寫入 SQLite 資料庫
- **進階統計指標**：攻擊效率、同位置 PR 值、綜合防守到位率等 Proxy Metrics
- **互動式儀表板**：Streamlit + Plotly 打造的七分頁視覺化分析介面
- **賽事預測**：基於 XGBoost 的比賽結果預測模型，搭配 SHAP 特徵解釋
- **AI 戰報**：透過 PCAI MLIS（OpenAI 相容 API）自動產生每周結構化中文戰報

## 儀表板分頁
> 🔗 **Live Demo**: [點此查看互動儀表板](https://tvl-analysis-jggmoeky3gnjrdbcc4kzrc.streamlit.app/)

| 分頁 | 功能 |
|------|------|
| 球員個人深度 | KPI 卡片、雷達圖、逐場趨勢（依位置自動調整） |
| 聯盟 PR 值與分佈 | 百分位排名、象限散佈圖（16 種指標可選） |
| 逐場趨勢 | 熱力資料表、對戰對手績效分佈 |
| 單場 Box Score | 雙方並列 Box Score、局比分、Top-10 排行 |
| 賽果預測 | ML 滑桿模擬器、SHAP 戰術診斷圖 |
| 每周戰報 | 視覺化比賽卡片、MLIS AI 自動撰寫戰報 |
| 系統設定 | 設定 MLIS endpoint/model/API key，測試連線 |

sidebar 最上層可選擇賽季（預設最新），下方所有分頁的資料皆依所選賽季過濾。

## 專案結構

```
TVL-Analysis/
├── src/
│   ├── app/
│   │   ├── main.py            # Streamlit 入口（路由 + sidebar，含賽季選擇器）
│   │   ├── helpers.py         # 共用函式（DB 查詢、指標計算、外部 API）
│   │   ├── llm_client.py      # MLIS（OpenAI 相容 API）呼叫層
│   │   ├── settings_store.py  # app_settings key-value 表存取
│   │   └── tabs/               # 七個分頁模組
│   │       ├── player_deep.py
│   │       ├── league_pr.py
│   │       ├── match_trend.py
│   │       ├── box_score.py
│   │       ├── prediction.py
│   │       ├── weekly_report_tab.py
│   │       └── settings_tab.py
│   ├── etl/                   # ETL 模組
│   │   ├── crawler.py         # 球員名單爬蟲
│   │   ├── stats_crawler.py   # 技術統計爬蟲（支援 --incremental）
│   │   ├── cleaner.py         # 資料清洗
│   │   ├── db_loader.py       # 資料庫載入
│   │   └── weekly_report.py   # 週報資料彙整
│   ├── models/                # ML 模型
│   └── utils/
│       ├── constants.py       # 共用常數（外部系統設定、隊伍對照表）
│       ├── db_config.py       # SQLite 連線管理
│       └── logger.py          # 統一日誌設定
├── data/
│   ├── raw/                   # 原始爬取資料 (CSV)
│   ├── processed/             # 清洗後資料
│   └── db/                    # SQLite 資料庫
├── notebooks/                 # 探索性分析與模型開發
├── sql/
│   └── schema.sql             # 資料庫 Schema（含效能索引）
└── requirements.txt
```

## 資料庫 Schema

| 資料表 | 說明 |
|---|---|
| `teams` | 球隊資料（複合主鍵：team_id + gender） |
| `players` | 球員基本資料（背號、位置、身高、體重等） |
| `player_match_stats` | 逐場技術統計（攻擊、攔網、發球、接發、防守、舉球） |

## 資料庫升級（v2 schema）

v2 schema 為 `players`/`player_match_stats`/`matches` 加入 `season` 賽季欄位，upsert 唯一鍵皆含 season，換季時不會覆蓋舊賽季資料。

- `DATABASE_URL`：連線目標，未設定時 fallback 至本地 `data/db/tvl_database.db`（SQLite）；指向 PostgreSQL 時使用 `postgresql+psycopg://...` 格式。
- `SEASON`：目前賽季字串（如 `2025-26`），未設定時預設 `2025-26`，ETL 寫入資料時以此標記賽季。
- `MLIS_BASE_URL` / `MLIS_API_KEY` / `MLIS_MODEL`：PCAI MLIS 的 OpenAI 相容 endpoint 設定，供「每周戰報」分頁產生 AI 戰報。可在「系統設定」分頁的 UI 設定（存於 Postgres/SQLite 的 `app_settings` 表，優先於環境變數），或直接設定環境變數；兩者皆未設定時戰報頁會顯示引導訊息。

若本地仍是舊版 `data/db/tvl_database.db`（無 `season` 欄位），直接使用會被 `init_db()` 擋下並提示錯誤，需先執行一次性升級遷移（目標不可與來源檔案相同）：

```bash
# 遷移至新的本地 SQLite 檔案
DATABASE_URL=sqlite:///data/db/tvl_v2.db python -m src.etl.migrate_to_postgres

# 或直接指向新的 PostgreSQL
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/tvl python -m src.etl.migrate_to_postgres
```

遷移完成後，將 `DATABASE_URL` 指向新資料庫（或以新檔案取代舊檔案）即可。

**部署順序**：在全新資料庫上，務必先跑過一次 `init_db`（透過 ETL 腳本或上述 migrate 腳本）建好 schema，才啟動 dashboard；dashboard 本身不會呼叫 `init_db`。唯一例外是 `app_settings` 表——dashboard 啟動與存設定時會自動補建（`CREATE TABLE IF NOT EXISTS`），即使忘了先跑 `init_db` 也不會因此崩潰，但其他表（`teams`/`players`/`player_match_stats`/`matches`）仍需 ETL 先建好。

## 安裝與使用

### 環境建置

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 執行爬蟲

```bash
# 爬取球員名單
python -m src.etl.crawler

# 爬取技術統計（全量）
python -m src.etl.stats_crawler

# 爬取技術統計（增量，僅新增缺少的比賽）
python -m src.etl.stats_crawler --incremental
```

### 載入資料庫

```bash
python -m src.etl.db_loader
```

### 啟動儀表板

```bash
streamlit run src/app/main.py
```

### AI 戰報（選用）

在環境變數或「系統設定」分頁中設定 `MLIS_BASE_URL`、`MLIS_API_KEY` 與 `MLIS_MODEL`（PCAI MLIS 叢集資訊），即可在儀表板「每周戰報」分頁使用 AI 自動產生戰報。

若 endpoint 憑證由內部自簽 CA 簽發導致連線測試失敗（`unable to get local issuer certificate`），可在「系統設定」分頁取消勾選「驗證 TLS 憑證」，或設定環境變數 `MLIS_CA_BUNDLE` 指向該 CA 憑證檔（正式環境建議採用後者）。

## 位置代號對照

| 中文 | 英文縮寫 |
|---|---|
| 主攻手 | OH (Outside Hitter) |
| 中間手 | MB (Middle Blocker) |
| 副攻手 | OP (Opposite) |
| 舉球員 | S (Setter) |
| 自由球員 | L (Libero) |

## 技術堆疊

- **爬蟲**：requests, BeautifulSoup
- **資料處理**：Pandas, NumPy
- **儀表板**：Streamlit
- **視覺化**：Plotly, Matplotlib
- **資料庫**：SQLite
- **ML**：XGBoost, scikit-learn, SHAP
- **AI 戰報**：PCAI MLIS（OpenAI 相容 API）

## 資料品質原則

- 缺失值保留為 `NA`/`None`，不做插補
- 日期統一為 `YYYY-MM-DD` 格式
- 數值欄位去除單位字串，轉為 Float/Integer
- 比率型指標分母 < 10 時顯示 N/A（避免小樣本偏差）

## 開發者

**小易** — 數據工程與儀表板開發
