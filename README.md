# TVL 企業排球聯賽數據分析儀表板

針對台灣企業排球聯賽（TVL）男女組賽事的數據爬取、清洗、儲存與視覺化分析系統。

## 功能特色

- **資料爬取**：自動抓取 TVL 官網的球隊名單、球員資料與逐場技術統計
- **資料清洗與載入**：標準化位置名稱、日期格式與數值欄位，寫入 SQLite 資料庫
- **進階統計指標**：攻擊效率、同位置 PR 值比較等 Proxy Metrics
- **互動式儀表板**：Streamlit + Plotly 打造的視覺化分析介面，支援個人深度分析與聯盟分佈比較
- **賽事預測**：基於機器學習的比賽結果預測模型

## 專案結構

```
TVL-Analysis/
├── src/
│   ├── etl/                # ETL 模組
│   │   ├── crawler.py      # 球員名單爬蟲
│   │   ├── stats_crawler.py# 技術統計爬蟲
│   │   ├── cleaner.py      # 資料清洗
│   │   ├── db_loader.py    # 資料庫載入
│   │   └── weekly_report.py# 週報產出
│   ├── app/
│   │   └── main.py         # Streamlit 儀表板主程式
│   ├── models/             # ML 模型
│   └── utils/              # 共用工具（DB 設定、Logger）
├── data/
│   ├── raw/                # 原始爬取資料 (CSV)
│   ├── processed/          # 清洗後資料
│   └── db/                 # SQLite 資料庫
├── notebooks/              # 探索性分析與模型開發 Notebook
├── sql/
│   └── schema.sql          # 資料庫 Schema 定義
├── requirements.txt
└── CLAUDE.md
```

## 資料庫 Schema

| 資料表 | 說明 |
|---|---|
| `teams` | 球隊資料（複合主鍵：team_id + gender） |
| `players` | 球員基本資料（背號、位置、身高、體重等） |
| `player_match_stats` | 逐場技術統計（攻擊、攔網、發球、接發、防守、舉球） |

## 安裝與使用

### 環境建置

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 初始化資料庫

```bash
sqlite3 data/db/tvl_database.db < sql/schema.sql
```

### 執行爬蟲

```bash
# 爬取球員名單
python -m src.etl.crawler

# 爬取技術統計
python -m src.etl.stats_crawler
```

### 啟動儀表板

```bash
streamlit run src/app/main.py
```

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
- **ML**：scikit-learn

## 資料品質原則

- 缺失值保留為 `NA`/`None`，不做插補
- 日期統一為 `YYYY-MM-DD` 格式
- 數值欄位去除單位字串，轉為 Float/Integer

