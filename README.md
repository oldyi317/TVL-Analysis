# TVL 企業排球聯賽數據分析儀表板

針對台灣企業排球聯賽（TVL）男女組賽事的數據爬取、清洗、儲存與視覺化分析系統。

## 功能特色

- **資料爬取**：自動抓取 TVL 官網的球隊名單、球員資料與逐場技術統計（支援增量更新）
- **資料清洗與載入**：標準化位置名稱、日期格式與數值欄位，寫入 SQLite 資料庫
- **進階統計指標**：攻擊效率、同位置 PR 值、綜合防守到位率等 Proxy Metrics
- **互動式儀表板**：Streamlit + Plotly 打造的五分頁視覺化分析介面
- **賽事預測**：基於 XGBoost 的比賽結果預測模型，搭配 SHAP 特徵解釋

## 儀表板分頁
> 🔗 **Live Demo**: [點此查看互動儀表板](https://tvl-analysis-jggmoeky3gnjrdbcc4kzrc.streamlit.app/)

| 分頁 | 功能 |
|------|------|
| 球員個人深度 | KPI 卡片、雷達圖、逐場趨勢（依位置自動調整） |
| 聯盟 PR 值與分佈 | 百分位排名、象限散佈圖（16 種指標可選） |
| 逐場趨勢 | 熱力資料表、對戰對手績效分佈 |
| 單場 Box Score | 雙方並列 Box Score、局比分、Top-10 排行 |
| 賽果預測 | ML 滑桿模擬器、SHAP 戰術診斷圖 |

## 專案結構

```
TVL-Analysis/
├── src/
│   ├── app/
│   │   ├── main.py            # Streamlit 入口（路由 + sidebar）
│   │   ├── helpers.py         # 共用函式（DB 查詢、指標計算、外部 API）
│   │   └── tabs/              # 五個分頁模組
│   │       ├── player_deep.py
│   │       ├── league_pr.py
│   │       ├── match_trend.py
│   │       ├── box_score.py
│   │       └── prediction.py
│   ├── etl/                   # ETL 模組
│   │   ├── crawler.py         # 球員名單爬蟲
│   │   ├── stats_crawler.py   # 技術統計爬蟲（支援 --incremental）
│   │   ├── cleaner.py         # 資料清洗
│   │   └── db_loader.py       # 資料庫載入
│   ├── models/                # ML 模型
│   └── utils/
│       ├── constants.py       # 共用常數（外部系統設定、隊伍對照表）
│       ├── db_config.py       # SQLite 連線管理
│       └── logger.py          # 統一日誌設定
├── data/
│   ├── raw/                   # 原始爬取資料 (CSV)
│   ├── processed/             # 清洗後資料
│   └── db/                    # SQLite 資料庫
├── notebooks/                 # 探索性分析（01_eda.ipynb）
├── sql/
│   └── schema.sql             # 資料庫 Schema（含效能索引）
└── requirements.txt
```

## 資料庫 Schema

| 資料表 | 說明 |
|---|---|
| `teams` | 球隊資料（複合主鍵：team_id + gender） |
| `players` | 球員身分層（姓名、性別、生日、身高、體重；不含背號/位置/隊伍） |
| `roster_registrations` | 球員 × 週次 × 隊伍的登錄（背號、位置；`source` 標記 `match_page`（實際出賽名單）或 `backfill`（以舊快照回推或該場無出賽記錄，非當週實際登錄）） |
| `player_match_stats` | 逐場技術統計（攻擊、攔網、發球、接發、防守、舉球），掛在 `registration_id` 下 |

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

# 爬取各局比分 → matches 表（stats_crawler 依賴此表最新，需先跑）
python -m src.etl.match_crawler

# 爬取技術統計（全量：掃全部場次，僅補齊缺少的紀錄，不清空既有資料）
python -m src.etl.stats_crawler

# 爬取技術統計（增量，僅新增缺少的比賽）
python -m src.etl.stats_crawler --incremental

# 爬取出賽名單 → roster_registrations
python -m src.etl.stats_crawler --rosters
```

### 載入資料庫

```bash
python -m src.etl.db_loader
```

`db_loader` 只負責建立/更新 `players` 身分層（姓名、性別等自然鍵 upsert），
背號、位置、隊伍等每週會變動的資訊改由 `roster_registrations` 承載。

### 重訓賽果預測模型

```bash
python -m src.models.train
```

以真實局數標籤重訓 XGBoost 賽果預測模型，產出 `src/models/match_predictor_v2.pkl`。

### 啟動儀表板

```bash
streamlit run src/app/main.py
```

### 靜態站（web/）本機開發三步

靜態站以 SvelteKit 建置，部署在 GitHub Pages（push 到 `main` 由 `.github/workflows/pages.yml` 自動匯出、建置、部署）。本機開發：

```bash
# 1. 匯出 JSON（從 repo 根目錄；輸出到 web/src/lib/data/，不進 git）
#    產出 seasons、teams、standings、matches、players、player_advanced、rosters、box_scores 八份，
#    球員 JSON 內含各賽季合計，進階數據 JSON 含每季比率、同位置 PR 值與分佈，
#    單場數據以賽事編號為鍵；統計列對不回賽事會印警告並排除
python3 -m src.export.site_data

# 2. dev server（會先驗證 JSON 符合 web/schema 並產生 TypeScript 型別）
cd web && npm ci && npm run dev

# 3. 建置靜態產物到 web/build/
npm run build
```

其他：`npm run check` 跑型別檢查；`npm run test:e2e` 以手機視窗對建置產物跑 Playwright smoke（第一次先 `npx playwright install chromium`）；匯出器測試 `pytest tests/test_export_*.py`。

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

## 資料品質原則

- 缺失值保留為 `NA`/`None`，不做插補
- 日期統一為 `YYYY-MM-DD` 格式
- 數值欄位去除單位字串，轉為 Float/Integer
- 比率型指標分母 < 10 時顯示 N/A（避免小樣本偏差）

## 開發者

**小易** — 數據工程與儀表板開發
