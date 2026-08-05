# TVL Dashboard 搬遷 HPE PCAI 設計

日期：2026-08-05
狀態：待使用者審核

## 1. 目標與範圍

把 TVL-Analysis 的 Streamlit dashboard 以**長期正式運行**等級搬上 HPE PCAI：

- 透過 AI Essentials 的 **Import Framework** 以 Helm chart 部署
- 資料庫由 SQLite 遷移到 **PostgreSQL**（chart 內 subchart）
- ETL 排程交給 PCAI 內建 **Airflow**（每日一次）
- 接上平台 **SSO**（oauth2-proxy，免改 app 程式碼）
- AI 戰報由 Gemini 改接 **PCAI MLIS 的 Qwen endpoint**，並可在 dashboard UI 直接設定 endpoint 與 API key
- schema 加入**賽季維度**，官網新賽季覆蓋資料時，歷年資料完整保存在 DB
- 修掉會影響正式運行的程式碼問題（破壞性 ETL、快取無 TTL、`st.stop()` 誤用、設定分散、依賴未鎖版）

**不在範圍內（YAGNI）**：

- dashboard 多副本水平擴展（單副本即可，Streamlit session 為 in-process）
- ML 模型重訓 pipeline（`match_predictor.pkl` 照現狀烘進 image）
- notebooks 改造（僅開發用，不進容器）
- 跨賽季追蹤同一球員（各賽季名單獨立，見 §7）
- 裸 `except Exception` 全面清理、設定頁角色權限（之後再做）

## 2. 環境前提

- PCAI 叢集**可連外網**：image 推 Docker Hub；爬蟲直連 `tvl.ctvba.org.tw` 與 `114.35.229.141`
- MLIS 上已有（或將部署）Qwen model deployment，提供 OpenAI 相容 endpoint 與 API token
  - 註：實際 endpoint URL 與 token 於實作時到叢集上驗證，不寫死於程式碼

## 3. 架構總覽

```
PCAI (AI Essentials)
├─ Import Framework → tvl-dashboard Helm chart (tar.gz)
│   ├─ Deployment: Streamlit app（1 replica）
│   ├─ PostgreSQL subchart（Bitnami；volumePermissions.enabled=true；storageClass 留空用預設）
│   ├─ templates/ezua/virtualService.yaml → istio-system/ezaf-gateway 路由
│   │    （prefix: / 全轉發，涵蓋 /_stcore/stream WebSocket）
│   ├─ templates/ezua/kyverno.yaml → 打 hpe-ezua/type: vendor-service 標籤（平台監控）
│   └─ AuthorizationPolicy（action: CUSTOM, provider: oauth2-proxy）→ 平台 SSO
├─ 內建 Airflow → tvl_etl DAG（每日台北時間凌晨）
│   crawler → db_loader → stats_crawler --incremental → match_crawler → pg_dump 備份
│   （各 task 用 KubernetesPodOperator 跑 dashboard 同一個 image）
└─ MLIS → Qwen deployment（OpenAI 相容 API）← dashboard 週報戰報呼叫
```

參考範例：[HPEEzmeral/byoa-tutorials](https://github.com/HPEEzmeral/byoa-tutorials)、[ai-solution-eng/frameworks](https://github.com/ai-solution-eng/frameworks)（已移植的 PostgreSQL chart）、[GuopingJia/pcai-helm-examples](https://github.com/GuopingJia/pcai-helm-examples)。

## 4. 程式碼改造

### 4a. 設定收斂與外部化

- 刪除 6 個檔案各自的 `try/except ModuleNotFoundError` fallback 區塊（`etl/crawler.py`、`etl/cleaner.py`、`etl/db_loader.py`、`etl/stats_crawler.py`、`etl/match_crawler.py`、`app/helpers.py` 等），統一從 `src/utils/constants.py` 匯入
- 重複的隊名對照表（`TEAM_NAME_SHORT`、`EXT_TEAM_MAP`、`TEAM_ALIAS`）合併到 constants，單一真實來源
- 環境變數可覆寫（皆有預設值，本地行為不變）：`EXT_BASE`（外部系統位址）、`SEASON`（當前賽季）、`LOG_LEVEL`、`DATABASE_URL`

### 4b. DB 層改 SQLAlchemy Core

- `src/utils/db_config.py` 改為由 `DATABASE_URL` 建立 SQLAlchemy engine；未設定時 fallback 現有 SQLite 路徑（本地開發零改變）
- `app/helpers.py` 廢除自己的路徑推導，改用 db_config
- ETL 寫入改參數化 SQL；upsert 用 SQLAlchemy 的方言相容寫法（SQLite/Postgres 皆可跑）
- `sql/schema.sql` 提供 Postgres 版；`pandas.read_sql` 接 engine 直接相容

### 4c. 修破壞性 ETL（Airflow 排程的前提）

- `sql/schema.sql` 移除 `DROP TABLE`，改 `CREATE TABLE IF NOT EXISTS`
- `db_loader.py` 與 `stats_crawler.py` 全量模式改 upsert，不再砍表重建
- 任何 ETL 重跑皆冪等：跑兩次結果相同、不丟資料

### 4d. Streamlit 修正

- 6 個 tab 內共 8 處 `st.stop()` 改為 `return`（`box_score.py:71,92`、`league_pr.py:38,114,128`、`match_trend.py:28`、`player_deep.py:80`、`weekly_report_tab.py:322,351`）
- `helpers.py` 兩處 `@st.cache_data` 加 `ttl=3600`（配合每日 ETL）
- `_purge_mpl_font_cache` 與字型初始化包進 `@st.cache_resource`，避免每次互動做磁碟 I/O
- `load_dotenv` 改為選用（檔案不存在即跳過）

### 4e. 依賴鎖定

- `requirements.txt` 全部釘版本，以目前能正常載入 `match_predictor.pkl` 的版本為準（實測驗證）
- 移除 `google-genai`，加入 `openai`、`sqlalchemy`、`psycopg[binary]`

## 5. AI 戰報改接 MLIS

- 移除 Gemini 三模型 fallback 迴圈與 `time.sleep(30)` 阻塞重試
- 改用 `openai` client：`base_url` 指向 MLIS endpoint、`api_key` 為 MLIS token、`model` 為 Qwen 部署名稱
- 設定讀取順序：**DB 的 app_settings（UI 設定）→ 環境變數 → 皆無則戰報頁顯示引導訊息**
- 簡單重試（1–2 次、短間隔）即可，內部 endpoint 無外部 rate limit 問題

## 6. 模型設定 UI

- dashboard 新增「系統設定」頁：endpoint base URL、model 名稱、API key 三個欄位 + 「測試連線」按鈕（實際打一次 endpoint 驗證）
- 設定存 Postgres `app_settings`（key-value 表）；API key 顯示時遮罩
- 整個 dashboard 已在平台 SSO 之後，設定頁天然受保護，不另做角色權限

## 7. 跨賽季資料保存

- **schema 加 `season` 欄位**：`players`、`matches`、`player_match_stats` 皆加入，所有 upsert unique key 包含 season——新賽季寫入新 season 的列，永不觸碰舊賽季資料
- **各賽季名單獨立**：同一人在不同賽季是不同筆 player 資料，零歧義（不處理改名/轉隊對應）
- 當前賽季由 `SEASON` 設定驅動（取代寫死的 `SEASON_YEAR_MAP`），換季只改設定
- dashboard sidebar 最上層加**賽季選擇器**，預設最新賽季，其餘篩選與分頁邏輯不變
- 既有資料（170 場、3,807 筆逐場數據）由遷移 script 標上目前賽季

## 8. 容器化

- **Dockerfile**：`python:3.11-slim` 多階段建置；`apt-get install fonts-noto-cjk`（取代 Streamlit Cloud 的 `packages.txt`）；non-root user；`MPLCONFIGDIR` 指向可寫目錄；`TZ=Asia/Taipei`
- **`.streamlit/config.toml`**：`headless=true`、`address=0.0.0.0`、XSRF 保持開啟、關閉 usage stats
- **`.dockerignore`**：排除 `notebooks/`、`data/`、`.git/`
- 健康探針：Streamlit 內建 `/_stcore/health`（liveness + readiness）
- 同一個 image 供 dashboard 與 Airflow ETL task 共用（ETL 進入點 `python -m src.etl.xxx`）
- image 推 Docker Hub，chart values 參數化 image repository/tag

## 9. Helm chart

- 以 byoa-tutorials 範例為骨架：Deployment / Service / ConfigMap / Secret
- `values.yaml` 含 `ezua.virtualService.endpoint: tvl.${DOMAIN_NAME}`、`ezua.virtualService.istioGateway: istio-system/ezaf-gateway`、resource requests/limits
- `templates/ezua/`：VirtualService + Kyverno ClusterPolicy（照官方模板）
- AuthorizationPolicy 接 oauth2-proxy（SSO）
- PostgreSQL Bitnami subchart：`volumePermissions.enabled: true`、storageClass 留空、密碼由 Secret 管理
- 打包 tar.gz，從 Import Framework UI 匯入（分類：Analytics）

## 10. Airflow DAG

- 單一 `tvl_etl` DAG，**每日一次**（台北時間凌晨，DAG 明確設 `Asia/Taipei` 時區）
- 順序：`crawler` → `db_loader` → `stats_crawler --incremental` → `match_crawler` → `pg_dump 備份到 PVC`
- 各 task 用 KubernetesPodOperator 跑 dashboard image，`DATABASE_URL` 指向 chart 內 Postgres service（`svc.namespace.svc.cluster.local`）
- 失敗通知用 Airflow 內建機制，不另建告警

## 11. 資料遷移（一次性）

- script 讀現有 `data/db/tvl_database.db`，經 SQLAlchemy 寫入 Postgres，標上目前賽季
- 部署後執行一次，之後由 Airflow 增量維護

## 12. 測試與驗證

- **ETL 冪等性測試**（本次最關鍵的行為改變）：pytest 驗證重跑兩次結果相同、不丟資料；upsert 不觸碰其他 season 的列
- SQLite 與 Postgres 雙方言驗證（本地 SQLite 跑測試、docker compose 起 Postgres 跑同一套）
- 本地 `docker compose`（app + Postgres）驗證容器化與 `DATABASE_URL` 切換，通過後再上 PCAI
- 不追求全面覆蓋，只覆蓋本次改動的部分

## 13. 實作順序（高層次）

1. 程式碼改造（§4）：設定收斂 → DB 層 → ETL 冪等 → Streamlit 修正 → 依賴鎖定
2. 賽季維度（§7）與遷移 script（§11）
3. MLIS 整合與設定 UI（§5、§6）
4. 容器化（§8）+ docker compose 本地驗證
5. Helm chart（§9）→ Import Framework 匯入
6. Airflow DAG（§10）→ 端到端驗證
