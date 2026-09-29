# TVL 儀表板遷移 HPE PCAI 設計書（Phase 1）

日期：2026-08-28
狀態：**已取消（2026-09-29）**——決定不遷移至 PCAI，維持 Streamlit Cloud 部署。
本文件僅留作決策紀錄；已完成的實作封存於本地 `pcai-migration` 分支，未 merge。
範圍：方案三第一期——SQLite 平移 + MinIO 儲存 + Airflow ETL + Helm/Harbor 部署。
Phase 2（EzPresto/Superset）與 Qdrant 不在本期，僅附綱要。

## 1. 背景與目標

現況：ETL 爬蟲 → SQLite（`data/db/tvl_database.db`，912 KB）→ Streamlit 儀表板 +
XGBoost 賽果預測（`src/models/match_predictor_v2.pkl`，438 KB）。部署於 Streamlit
Cloud，每日更新靠 GitHub Actions self-hosted runner（在使用者 WSL 上，因官網有
Cloudflare 擋雲端 runner）爬完後把 `.db` commit 回 repo 觸發重部署——「git 當儲存、
push 當部署」。

目標：正式搬家到公司 HPE PCAI（AI Essentials 1.12.x），內部使用即可。過渡期
Streamlit Cloud 與 GitHub Actions 雙軌保留，穩定後退役。

## 2. 已確認的決策

| 面向 | 決定 |
|---|---|
| 目的 | 正式搬家，PCAI 為主要部署目標 |
| 受眾 | 內部使用；PCAI 版上線穩定後 Streamlit Cloud 退役 |
| ETL | 搬進 PCAI，用 Airflow DAG（KubernetesPodOperator）排程 |
| ML 重訓 | Kubeflow 手動觸發，產出 `.pkl` 上傳 MinIO；app 只讀 |
| 資料層 | 維持 SQLite，MinIO 取代 git 作為 artifact 儲存（方案三第一期） |
| 前後端 | 不拆。Streamlit 是伺服器端渲染單體，一個 Deployment 即可 |
| 平台可用服務 | Airflow、Qdrant、MinIO（Import Framework 裝的）、EzPresto、Kubeflow |
| 部署方式 | Docker image → Harbor → Helm chart（PCAI import framework 規格）→ Import Frameworks 精靈 |

## 3. 動工前驗證（checklist）

1. **［go/no-go］叢集出口能否爬到資料來源**：在叢集 notebook 或臨時 pod 實測
   `https://tvl.ctvba.org.tw`（Cloudflare）與 `http://114.35.229.141`（明文 HTTP）。
   若 Cloudflare 擋叢集出口 → 啟動 §12 備援（爬蟲留 runner、改推 MinIO 對外
   endpoint），其餘設計不變。
2. **MinIO tile**：確認實際 in-cluster endpoint（預期
   `http://minio.minio.svc.cluster.local:9000`，release/namespace 不同則異）、
   在 console 建 access key 與 bucket `tvl`。
3. **Airflow DAG git repo**：全叢集共用一個 DAG repo 設定（Airflow tile → 三點 →
   Configure 的 `dags.git:` 段），確認目前指向哪個 repo、DAG 檔要放進哪裡、誰有權限 push。
4. **Harbor**：確認自己的專案與帳號可用；建 project 級 robot account（只勾 Pull，
   token 只顯示一次；登入名格式 `robot$<project>+<name>`，含 `$` 引用時加單引號）。

## 4. 目標架構

```
官網 tvl.ctvba.org.tw ─┐
統計系統 114.35.229.141 ─┤（每日爬蟲）
                        ▼
        Airflow DAG（KubernetesPodOperator，跑 tvl-analysis image）
          MinIO 拉 .db → 三支爬蟲 → iterdump hash 比對 → 有變更才上傳
                        ▼
        MinIO bucket tvl/（db/tvl_database.db、models/match_predictor_v2.pkl）
                        ▲                         ▼
   Kubeflow（手動重訓：拉 .db →         Streamlit app Pod（啟動下載 +
   src.models.train → 上傳 .pkl）        每小時 ETag 檢查，變更即重拉並清快取）
                                                  ▼
              瀏覽器 ← Istio ezaf-gateway ← VirtualService
                       （tvl-dashboard.<DOMAIN>，SSO via oauth2-proxy）
```

- 單一 image `tvl-analysis` 供 app / ETL / Kubeflow 三種用途，靠 command 區分。
- 寫入 MinIO 的只有 Airflow DAG（模型檔則只有 Kubeflow），無併發寫入問題。
- 查詢邏輯（SQLite raw SQL）完全不動。

## 5. 端點與設定的歸屬

原則：程式碼不寫死 endpoint，一律環境變數；環境變數唯一來源是 chart 的
`values.yaml`（部署後在 tile 三點選單 → Configure 線上改，存檔即重佈署）。

| 環境變數 | 意義 | 未設時 |
|---|---|---|
| `TVL_DB_PATH` | SQLite 檔本地路徑 | fallback repo 內 `data/db/tvl_database.db` |
| `TVL_MODEL_PATH` | 模型檔本地路徑 | fallback repo 內 `src/models/match_predictor_v2.pkl` |
| `TVL_S3_ENDPOINT` | MinIO endpoint | **artifact_store 整個不作用**（本機/Streamlit Cloud 開關） |
| `TVL_S3_BUCKET` | bucket 名 | — |
| `TVL_S3_ACCESS_KEY` / `TVL_S3_SECRET_KEY` | 憑證，來自 K8s Secret（`secretKeyRef`） | — |

爬蟲目標 URL 維持在 `src/utils/constants.py`（資料來源非基礎設施）。
儀表板對外網址：`values.yaml` 的 `ezua.virtualService.endpoint:
"tvl-dashboard.${DOMAIN_NAME}"`（`${DOMAIN_NAME}` 由平台注入）。

## 6. 程式碼改動（既有檔案，5 處）

1. `src/utils/db_config.py`：`DB_PATH` 先讀 `TVL_DB_PATH`，fallback 現路徑。
2. `src/app/helpers.py`：刪除自複製的 `DB_PATH`（helpers.py:94-95），改 import
   `db_config`；`MODEL_PATH` 改讀 `TVL_MODEL_PATH`。
3. `src/app/tabs/prediction.py:26`：裸 `sqlite3.connect` 改走 `db_config.get_connection()`。
4. `src/app/main.py`：啟動時呼叫 artifact 同步；加 TTL 一小時的 ETag 檢查，變更則
   重拉 `.db`/`.pkl` 並清 `st.cache_data` / `st.cache_resource`。
5. `requirements.txt`：加 `boto3`（pin 死；`tests/test_requirements_pinned.py` 同步更新）。

## 7. 新增程式碼

- `src/utils/artifact_store.py`：boto3 小封裝。讀 §5 環境變數；
  `download_if_changed(key, local_path)`（ETag 比對）、`upload(local_path, key)`。
  `TVL_S3_ENDPOINT` 未設時所有函式為 no-op。
- `src/etl/pcai_daily_update.py`：容器版每日更新（等價 `daily_update.sh` 去掉 git）：
  MinIO 拉 `.db` → 依序跑 `match_crawler`、`stats_crawler --rosters`、
  `stats_crawler --incremental` → iterdump SHA-256 比對（濾 `sqlite_sequence`，
  自 bash 移植為 Python）→ 有變更才上傳。支援 `DRY_RUN=1`。

## 8. 新增部署檔案

### 8.1 `Dockerfile`（repo 根目錄）

- base：`python:3.11-slim-bookworm`
- `apt-get install fonts-noto-cjk`（接手 `packages.txt` 的角色；`main.py:45-66` 靠
  walk `/usr/share/fonts` 找 Noto CJK）
- `pip install -r requirements.txt`；COPY `src/ sql/`
- `WORKDIR /app`（滿足 `main.py:11-13` 的 sys.path 推導）
- `ENV HOME=/tmp`（`main.py:18-27` 需要可寫的 matplotlib cache）
- 預設 CMD：`streamlit run src/app/main.py --server.address 0.0.0.0
  --server.enableCORS false --server.enableXsrfProtection false`
- 子網域路由走 `/`，**不需** `--server.baseUrlPath`。

### 8.2 `deploy/helm/tvl-dashboard/`（照 AIE 1.12 import framework 規格）

- `values.yaml`：
  - `image: {registry: harbor.ai-application.scsvc.com, repository: <project>/tvl-analysis, tag: ..., pullPolicy: IfNotPresent}`
  - `service: {type: ClusterIP, port: 80, targetPort: 8501}`（container port 命名 `http`）
  - `env:` 區塊（§5 的變數）
  - `imagePullSecrets: [{name: tvl-harbor-pull}]`（Secret 手動建，token 不進 chart）
  - `ezua.virtualService: {endpoint: "tvl-dashboard.${DOMAIN_NAME}", istioGateway: "istio-system/ezaf-gateway"}`
  - `ezua.authorizationPolicy: {namespace: "istio-system", providerName: "oauth2-proxy", matchLabels: {istio: ingressgateway}}`
  - `ingress.enabled: false`（PCAI 用 VirtualService，不用一般 ingress）
- `templates/`：deployment、service、`ezua/virtualService.yaml`（destination host
  必須等於 Service 名 `<svc>.{{ .Release.Namespace }}.svc.cluster.local`）、
  `ezua/authpolicy.yaml`、`_hpe-ezua.tpl`：

  ```yaml
  {{- define "hpe-ezua.labels" -}}
  hpe-ezua/app: {{ .Release.Name }}
  hpe-ezua/type: vendor-service
  {{- end }}
  ```

  Deployment 的 metadata.labels 與 pod template labels **兩處**都要 include，
  否則 tile 顯示 Unknown。
- probes：Streamlit 健康端點 `/_stcore/health`；readiness
  `initialDelaySeconds: 30`（shap/numba 冷啟動 >10 秒，`main.py:31-35`）。
- SSO 無縫登入需請管理員在 Keycloak（Unified Analytics realm → Clients → UA）加
  redirect URL `https://tvl-dashboard.<DOMAIN>/oauth2/callback`。

### 8.3 `deploy/airflow/tvl_daily_dag.py`

- KubernetesPodOperator：image 同 8.1、command `python -m src.etl.pcai_daily_update`、
  env 照 §5、憑證 `secretKeyRef`。
- 排程 `23 1 * * *`（Airflow 預設 UTC = 台灣 09:23，同現制錯開整點）。
- 注意：Airflow executor 資源上限固定（CPU 1 / mem 2Gi），爬蟲在獨立 pod 跑不受限。
- 此檔最終要放進叢集設定的 DAG git repo（驗證項 3）。

## 9. 一次性手動步驟

1. `kubectl create secret docker-registry tvl-harbor-pull ...`（robot account）。
2. `kubectl create secret generic tvl-minio-cred ...`（access/secret key）。
3. MinIO console 建 bucket `tvl`；初始上傳現行 `.db` 與 `.pkl`（小腳本或 console 拖放）。
4. Windows Docker Desktop 裝 AIE Root CA（Chrome 匯出最上層 Root CA → 裝入
   「受信任的根憑證授權單位」→ 完全重啟 Docker Desktop）後
   `docker login harbor.ai-application.scsvc.com` → build → tag → push。
5. `helm package` 出 `.tgz` → Tools & Frameworks → Import Frameworks 精靈
   （icon ≤2MB；Category 選 Analytics）。後續更新版本走 tile 三點 → Update → Upgrade。

## 10. 測試策略

- 既有 `pytest` 每步改動後跑（含 pin 檢查）。
- 新增單元測試：`artifact_store`（env 未設時 no-op；ETag 邏輯，mock boto3，不加新測試依賴）、
  `pcai_daily_update` 的 hash 移植對照 bash 版行為。
- **本機 Docker E2E（push Harbor 前守門）**：Docker Desktop 起本機 MinIO 容器 →
  build image → 上傳 `.db`/`.pkl` → `docker run` 帶完整 env → 驗證五個 tab 與中文
  字型 → 容器內跑 `DRY_RUN=1` 更新。把「程式碼對不對」與「平台通不通」拆開。
- 平台 smoke test：tile 狀態非 Unknown、頁面開得起來且不轉圈（websocket 過
  Istio）、SSO 生效、一小時內拿到 DAG 更新的資料。

## 11. 上線步驟

1. §3 四項前置驗證（第 1 項 go/no-go）
2. 程式碼改動 + 測試（§6、§7）
3. 本機 Docker E2E（§10）
4. 裝 CA → push image 到 Harbor
5. §9 的 secrets / bucket / 初始上傳
6. chart 打包 → Import → 平台 smoke test
7. DAG 進 DAG repo → 手動 trigger → 驗證 MinIO 更新與 app 取得新資料
8. Kubeflow 重訓演練一次（拉 `.db` → train → 上傳 `.pkl` → app 拿到）

## 12. 風險與備援

| 風險 | 對策 |
|---|---|
| Cloudflare 擋叢集出口（頭號） | 備援：爬蟲留在現有 self-hosted runner，改推 MinIO 對外 endpoint `https://minio-api.<DOMAIN>`（需驗證 runner 所在網路連得到公司內網）；其餘設計不變 |
| DAG repo 全叢集共用 | 驗證項 3 先確認歸屬與權限；DAG 對所有已認證使用者可見，檔名/task id 加 `tvl_` 前綴避免衝突 |
| websocket 被 gateway 擋 | smoke test 第一檢查點；Istio HTTP route 預設支援 upgrade，異常時查 VirtualService 設定與 controller log（`kubectl logs -n ezapp-system -l control-plane=controller-manager`） |
| tile 顯示 Unknown | `hpe-ezua` labels 兩處 include 缺一不可；檢查 `EzAppConfig.Spec.Values` 的 `ezua.virtualService` 語法 |
| Secret 輪替 | robot token / MinIO key 更新後重啟 pod 即可，皆不進 git |

## 13. 過渡與退役

- 雙軌期 2–4 週：兩邊每日各自爬（官網每日被爬兩次，可接受）；每週抽查前一日
  兩邊 `.db` 的 iterdump hash 一致性。
- 退役：停用 `.github/workflows/daily-crawl.yml` → 移除 self-hosted runner →
  下架 Streamlit Cloud。
- repo 內 `.db`/`.pkl` **保留**作本機開發 seed，但不再每日自動 commit；
  屆時同步改寫 `CLAUDE.md` 地雷章節與 README。

## 14. Phase 2 綱要（另立 spec，不在本期）

- ETL 加一步 Parquet 匯出到 MinIO → 管理員在 Data Engineering 建 Hive connector
  資料源（免 Metastore，子目錄=表）→ EzPresto / 內建 Superset 可查同一份資料；
  app 本體不動。
- EzPresto 連線要點：Python `prestodb` client、port 443、BasicAuthentication
  （密碼 = refresh token，自 `https://token-service.<DOMAIN>/refresh-token-download`
  無痕視窗取得）；資料源名稱不可含底線；只有管理員可建連線。
- Qdrant：現有功能無語意檢索需求，留給未來 RAG 類功能。
- 模型推論獨立成 KServe/MLIS 服務：目前互動式 what-if + SHAP 在 app 內即可，暫不做。

## 15. 參考資料

- AIE 1.12 文件 API 直取法：`https://support.hpe.com/hpesc/public/api/document/a00aie112hen_us?docLocale=en_US&page=<path>`（docDisplay 是 SPA，curl 抓不到；`/toc` 回 JSON 目錄）
- Preparing the Framework Helm Chart：`...docDisplay?docId=a00aie112hen_us&page=ManageClusters/prepare-helm-charts-for-import-app.html`
- Importing Frameworks Using UI：`...&page=ManageClusters/importing-applications.html`
- Managing Imported Tools and Frameworks：`...&page=ManageClusters/managing-applications.html`
- Airflow DAGs Git Repository：`...&page=Airflow/airflow-dags-git-repos.html`
- 本地文件：`official-docs/HPE_a00aie112hen_us_ Data Engineering.pdf`
  （EzPresto p.10–113、Python client p.106–110、S3 免金鑰 p.6–7、Airflow p.136–156）
- Notion：Prepare and deploy Helm chart using Harbor registry
  （`https://app.notion.com/p/3a08440a38278038a7a6e1b5fb18b396`）、
  Connect to Harbor and push images
  （`https://app.notion.com/p/3a08440a3827807e8902c23c7b7c1241`）
- PCAI 適配 chart 範例：`https://github.com/ai-solution-eng/frameworks`
  （minio/、qdrant/）、`https://github.com/GuopingJia/pcai-helm-examples`
- Harbor：`harbor.ai-application.scsvc.com`；Istio gateway：`istio-system/ezaf-gateway`
