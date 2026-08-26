# TVL-Analysis

台灣企業排球聯賽（TVL）數據分析：ETL 爬蟲 → SQLite → Streamlit 儀表板 + ML 賽果預測。
公開部署於 Streamlit Cloud，繁體中文介面。

## 常用指令

```bash
# 啟動 app（從 repo 根目錄）
streamlit run src/app/main.py

# 爬蟲（皆從 repo 根目錄以 module 方式執行）
python -m src.etl.crawler                        # 官網球員名單 → data/raw/
python -m src.etl.stats_crawler --incremental    # 逐場技術統計（增量）
python -m src.etl.match_crawler                  # 官網各局比分 → matches 表
python -m src.etl.db_loader                      # roster CSV → teams/players
```

## 地雷（改動前必讀）

- **`schema.sql` 已冪等，不會清空資料**：全部改用 `CREATE TABLE IF NOT EXISTS`，
  沒有 `DROP TABLE`。`db_loader` 對 `teams`/`players` 是以自然鍵 upsert，會保留
  既有 `player_id`；`stats_crawler` 全量與增量模式皆為「補缺不清表」，去重鍵是
  `match_date + is_golden_set`（逐球員判斷）。
- **`.db` 與 `.pkl` 是刻意 commit 進 git 的**：Streamlit Cloud 靠 repo 內的
  `data/db/tvl_database.db` 與 `src/models/match_predictor.pkl` 拿到資料與模型，
  不要把它們加進 `.gitignore`。
- **行尾一律 LF**：repo 在 WSL 的 `/mnt/d` 上，Windows 工具容易把檔案轉成 CRLF，
  會製造上萬行假 diff。commit 前用 `git diff --stat -w` 確認沒有純行尾雜訊。
- **資料來源有兩個**：官網 `tvl.ctvba.org.tw`（名單、比分）與外部統計系統
  `http://114.35.229.141`（明文 HTTP、hardcoded IP，見 `src/utils/constants.py`）。
  後者不可從程式碼推斷格式，改爬蟲前先實際抓一頁看回應。
- **每週登錄名單會變動**：企業排球每週登錄球員可能不同，`stats_crawler` 遇到
  名單外的球員會直接插入 `players`（缺背號與位置），下游計算必須對缺值防禦。

## 慣例

- import 一律走 `from src.utils...` 絕對路徑，從 repo 根目錄執行。
- 共用常數（隊伍對照表、headers、DB 路徑）只放 `src/utils/constants.py` 與
  `src/utils/db_config.py`，不要在個別檔案裡複製一份。
- DB 連線用 `src/utils/db_config.get_connection()`（有開 `PRAGMA foreign_keys`），
  不要裸用 `sqlite3.connect()`。
- Schema DDL 的唯一來源是 `sql/schema.sql`，爬蟲內不要重寫 CREATE TABLE。
- UI 文案、註解、commit message 用繁體中文。
- 部署依賴：`requirements.txt`（pip）與 `packages.txt`（Streamlit Cloud 的 apt
  清單，目前只有中文字型）。改依賴要同時考慮 Streamlit Cloud 重建。

## 資料品質原則

只標記與警告異常，不插補、不竄改原始數據（cleaner 的既有原則，維持它）。
