# TVL-Analysis

台灣企業排球聯賽（TVL）數據分析：ETL 爬蟲 → SQLite → Python 匯出器 → SvelteKit 靜態站（GitHub Pages）+ ML 賽果預測。繁體中文介面。

## 常用指令

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
python -m src.etl.stats_crawler --incremental    # 逐場技術統計（增量；依賴 matches 表最新，需先跑 match_crawler）
python -m src.etl.stats_crawler --rosters        # 出賽名單→roster_registrations（先跑 match_crawler）
python -m src.etl.db_loader                      # roster CSV → players 身分層
bash scripts/daily_update.sh                     # 每日增量更新（GitHub Actions 排程同款；DRY_RUN=1 可演練）
python -m src.models.train                       # 重訓賽果預測模型 → match_predictor_v2.pkl
python -m src.models.predict                     # 未開打賽事的主隊賽前勝率 → match_predictions 表（daily_update 內含）
```

## 地雷（改動前必讀）

- **`schema.sql` 已冪等，不會清空資料**：全部改用 `CREATE TABLE IF NOT EXISTS`，
  沒有 `DROP TABLE`。`db_loader` 對 `teams`/`players` 是以自然鍵 upsert，會保留
  既有 `player_id`；`stats_crawler` 全量與增量模式皆為「補缺不清表」，去重鍵是
  `match_date + is_golden_set`（逐球員判斷）。
- **`.db` 與 `.pkl` 是刻意 commit 進 git 的**：pages.yml 在雲端 runner 只
  checkout repo 就跑匯出器建置，靠 repo 內的 `data/db/tvl_database.db`；
  `src/models/match_predictor_v2.pkl` 供第三期賽前勝率匯出使用。
  不要把它們加進 `.gitignore`。artifact 需含 `version`/`feature_cols`，
  `tests/test_prediction_artifact.py` 鎖住合約（CI 不跑 pytest，本機要跑），重訓務必走 `python -m src.models.train`，不要手動 pickle。
- **行尾一律 LF**：repo 在 WSL 的 `/mnt/d` 上，Windows 工具容易把檔案轉成 CRLF，
  會製造上萬行假 diff。commit 前用 `git diff --stat -w` 確認沒有純行尾雜訊。
- **資料來源有兩個**：官網 `tvl.ctvba.org.tw`（名單、比分）與外部統計系統
  `http://114.35.229.141`（明文 HTTP、hardcoded IP，見 `src/utils/constants.py`）。
  後者不可從程式碼推斷格式，改爬蟲前先實際抓一頁看回應。
- **每週登錄名單會變動**：企業排球每週登錄球員可能不同，`stats_crawler` 遇到
  名單外的球員現在是「插入身分層 `players` (name, gender) + 建
  `source='backfill'` 登錄（背號/位置皆 NULL）」，下游計算仍須對缺值防禦。
- **換季前必跑 `docs/ops/season-switch.md` checklist**：cup_id 已進
  roster_registrations 的 UNIQUE（Phase 3），跨季同名週次不再互相覆寫；但
  `EXT_CUP_ID` 等常數是手動更新的，未更新前爬蟲抓不到新季資料。

## 慣例

- import 一律走 `from src.utils...` 絕對路徑，從 repo 根目錄執行。
- 共用常數（隊伍對照表、headers、DB 路徑）只放 `src/utils/constants.py` 與
  `src/utils/db_config.py`，不要在個別檔案裡複製一份。
- DB 連線用 `src/utils/db_config.get_connection()`（有開 `PRAGMA foreign_keys`），
  不要裸用 `sqlite3.connect()`。
- Schema DDL 的唯一來源是 `sql/schema.sql`，爬蟲內不要重寫 CREATE TABLE。
- UI 文案、註解、commit message 用繁體中文。
- Python 依賴：`requirements.txt` 只服務自架 runner 的爬蟲與模型重訓；pages.yml
  的匯出器只用標準庫、不裝 pip 套件，改匯出器時不要引入第三方套件。前端依賴在
  `web/package.json`，改動要考慮 pages.yml 的 npm ci。

## 資料品質原則

只標記與警告異常，不插補、不竄改原始數據（cleaner 的既有原則，維持它）。

## Agent skills

### Issue tracker

Issue 記在 GitHub Issues（github.com/oldyi317/TVL-Analysis，以 `gh` CLI 操作）。見 `docs/agents/issue-tracker.md`。

### Triage labels

沿用預設五個 label：needs-triage、needs-info、ready-for-agent、ready-for-human、wontfix。見 `docs/agents/triage-labels.md`。

### Domain docs

單一 context：根目錄 `CONTEXT.md` 加 `docs/adr/`，由 /domain-modeling 於需要時建立。見 `docs/agents/domain.md`。
