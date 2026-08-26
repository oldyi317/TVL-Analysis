# TVL Phase 2 Schema 重建模 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `players` 拆成「身分層」（`players`：姓名、性別、生日、身高體重）與「週次登錄層」（`roster_registrations`：球員×週次×隊伍×背號×位置），`player_match_stats` 改掛在登錄記錄上，讓每週登錄名單變動（企業排球每週可能不同）成為資料模型本身能表達的事實，而不是被迫塞進一張只能存一份「現時快照」的 `players` 表。
**Architecture:** 新增 `roster_registrations` 作為 `players`（身分）與 `player_match_stats`（事實）之間的橋接維度表；資料來源是外部統計系統 `http://114.35.229.141` 的逐場出賽名單頁 `_handler/Match.ashx`（給背號/姓名/位置），透過 `matches.round_name` 反查週次標籤，讓每筆登錄落在正確的週次。既有 3,807 筆 `player_match_stats` 連同全部 tab、helpers、main.py 側欄一併遷移到新結構。
**Tech Stack:** 沿用 Phase 1 之後的技術棧（Python、SQLite、BeautifulSoup4、pandas、Streamlit、pytest），不新增套件。
**Spec:** roadmap 記憶檔 `tvl-optimization-roadmap` + 本 session 決策：Phase 2 採 B 案變體（逐場出賽名單重建）。

## 前提

本計畫假設 `2026-08-26-phase1-foundation.md` 已完整執行完畢：
- 每周戰報功能（`weekly_report.py`、`weekly_report_tab.py`、`google-genai`）已刪除。
- 6 處 import fallback 已消除，所有模組一律 `from src...` 絕對 import。
- `requirements.txt` 已釘版本，`requirements-dev.txt` 已存在並含 `pytest`。
- `sql/schema.sql` 已改為全面 `CREATE TABLE IF NOT EXISTS`（冪等，不清空既有資料），`db_loader.py` 的 `insert_teams`/`insert_players` 已改名為 `upsert_teams`/`upsert_players`。
- `tests/` 目錄與 `tests/conftest.py` 的 `tmp_db_path` fixture已存在。

若上述任一項尚未完成，先執行 Phase 1 計畫，不要跳著做。

**Phase 1 終審遺留事項（本 phase 執行時需知）：**
- Phase 1 終審修正後，`stats_crawler` 全量/增量模式皆為「補缺不清表」，去重鍵 `(match_date, is_golden_set)` 逐球員判斷（`filter_new_records()`）；本 phase 改爬蟲時沿用此語意。
- `upsert_players` 的自然鍵含 `jersey_number`：球員換背號會被視為新身分而分裂出第二筆 `players` 列（終審 Important #3，裁決帶到本 phase 處理）——本 phase 的身分層重建正是根治點，遷移前先檢查是否已有分裂列需合併。
- NULL/NaN `jersey_number` 的 upsert 行為目前靠 sqlite3 C 層 NaN→NULL 轉換保證正確、無回歸測試守門；若本 phase 改動 pandas dtype（如 nullable Int64/pd.NA），需補測試。

## Global Constraints

- **行尾一律 LF**：每個 task 收尾前跑 `git diff --stat -w` 確認無純行尾雜訊。
- **繁體中文**：UI 文案、commit message 一律繁體中文。
- **DB 連線一律用 `src.utils.db_config.get_connection()`**，不要裸用 `sqlite3.connect()`（一次性遷移腳本因需要對同一個檔案做結構性 DDL 操作，可用 `get_connection()` 拿到連線後照常操作，不需要另開連線方式）。
- **DDL 唯一來源是 `sql/schema.sql`**：新表 `roster_registrations` 的定義也只能寫在這裡，任何 `.py` 都不得重寫 `CREATE TABLE`。
- **requirements.txt 只放 Streamlit Cloud 執行期依賴**：本 phase 不新增任何第三方套件，不用動 `requirements.txt`。
- **資料品質原則：只標記不插補**：歷史週次若查無法對應到真實出賽名單頁的資料，一律用 `source = 'backfill'` 明確標記為「用當下快照回推、非該週真實登錄」，禁止用插值或猜測產生看似逐週變化的假資料。
- **`.db` 與 `.pkl` 不得進 `.gitignore`**：遷移後的 `data/db/tvl_database.db` 仍要 commit 進 git。
- **commit 前需徵求使用者同意**：每個 task 走到 commit 步驟時，先把 commit message 草案貼給使用者確認，同意後才 `git commit`。
- **遷移前必須備份**：任何會修改 `data/db/tvl_database.db` 結構或內容的步驟，執行前一律先複製一份帶時間戳的備份檔，且不得刪除備份（备份保留給使用者事後比對）。
- **不過度工程化**：函式小而專注、early return 優先。

---

## 資料模型與外部資料來源（已於規劃 session 實查，供所有 task 參照，不得再自行假設格式）

### 目標 schema（`sql/schema.sql` 最終形態，Task 1 會實際寫入這份定義）

```sql
CREATE TABLE IF NOT EXISTS players (
    player_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT,
    gender     TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    dob        DATE,
    height_cm  REAL,
    weight_kg  REAL
);

CREATE TABLE IF NOT EXISTS roster_registrations (
    registration_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id        INTEGER NOT NULL,
    team_id          INTEGER NOT NULL,
    gender           TEXT    NOT NULL CHECK (gender IN ('M', 'F')),
    week_label       TEXT    NOT NULL,
    week_start_date  DATE,
    jersey_number    INTEGER,
    position         TEXT,
    source           TEXT    NOT NULL CHECK (source IN ('match_page', 'backfill')),
    FOREIGN KEY (player_id) REFERENCES players (player_id),
    FOREIGN KEY (team_id, gender) REFERENCES teams (team_id, gender),
    UNIQUE (player_id, team_id, gender, week_label)
);

CREATE TABLE IF NOT EXISTS player_match_stats (
    stat_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    registration_id   INTEGER NOT NULL,
    match_date        DATE,
    opponent          TEXT,
    sets_played       INTEGER,
    attack_total      INTEGER,
    attack_points     INTEGER,
    block_points      INTEGER,
    serve_total       INTEGER,
    serve_points      INTEGER,
    receive_total     INTEGER,
    receive_excellent INTEGER,
    dig_total         INTEGER,
    dig_excellent     INTEGER,
    set_total         INTEGER,
    set_excellent     INTEGER,
    total_points      INTEGER,
    is_golden_set     INTEGER NOT NULL DEFAULT 0 CHECK (is_golden_set IN (0, 1)),
    FOREIGN KEY (registration_id) REFERENCES roster_registrations (registration_id)
);
```

`teams` 與 `matches` 兩表結構不變（沿用 Phase 1 之後的 `CREATE TABLE IF NOT EXISTS` 版本）。

**`week_label` 的定義（重要，不可用別的邏輯重新發明）**：`week_label` 的值就是 `matches.round_name` 的字面值（例如 `"例行賽 Week 6"`、`"挑戰賽 Week 18"`、`"總決賽 Week 19"`），透過 `match_date` 反查取得，**不是**從 `_handler/Match.ashx` 頁面自己的標題文字重建。原因見下方「外部資料來源」小節的實查結果。

`week_start_date` = `SELECT MIN(match_date) FROM matches WHERE round_name = <week_label>`，用來讓「該球隊最新一週的登錄名單」這類查詢可以直接 `ORDER BY week_start_date DESC` 取得，不用對 `week_label` 字串做語意排序。

### 外部資料來源：`_handler/Match.ashx`（本 session 已用 curl 實際抓取驗證，非猜測）

**用途**：這是官方統計系統上「單場比賽的完整出賽名單 + box score」頁面，也是本 phase 唯一能取得「某球員在某一週背號、位置」的資料源（`_handler/Player.ashx`，也就是 `stats_crawler.py` 現有的逐場數據來源，只給數值統計，完全不含位置/背號欄位）。

**如何列舉全部 MatchID（重要地雷）**：`MatchID` **不是**從 1 開始連續編號 —— 實測 `CupID=21` 時 `MatchID=1..108` 有效、`109..205` 全部回傳「並未將物件參考設定為物件的執行個體」錯誤頁、但 `206..210`（總決賽）又有效。**絕對不能用「連續掃描直到連續 N 次空頁就停止」的邏輯**（`match_crawler.py` 對官網 `game_id` 用的那招在這裡行不通）。正確做法是先呼叫：

```
GET http://114.35.229.141/Match.aspx?CupID=21
```

解析 `<select id='divSelect'>` 底下的 `<option value='MatchID'>{賽別}-{性別} {編號}：{主隊} vs {客隊} (MM月DD日)</option>`，`value` 屬性就是精確的 `MatchID` 清單（實測 CupID=21 共 118 個 `<option>`）。這與 `src/app/helpers.py` 現有的 `fetch_match_index()` 走的是同一個 `Match.aspx` 端點，可直接參考其寫法（但 `fetch_match_index()` 目前的 regex 沒有擷取賽別/性別前綴，本 phase 的新函式要自己寫，不要改動 `fetch_match_index()` 本身，因為 `box_score.py` 還在用它）。

**單場出賽名單頁的實際 HTML 結構**（`GET http://114.35.229.141/_handler/Match.ashx?CupID=21&MatchID=1`，2026-08-26 實測，節錄）：

```html
<h3><img src='_images/Sex_2.png' height='32' />女子組 第1週(成功大學體育館) 編號：1 (11月1日 13:00)  歷時 02:22    <a id='21_1' href='P2/21_1.pdf'>P2</a></h3>
...
<h3>新北中纖：邱雅慧、簡佳慧</h3>
<div class='TableFormat_1'><table ...>
<tr><td class='head' colspan='3'>新北中纖</td>...<td class='head' colspan='2'>攻擊(Attack)</td>...</tr>
<tr><td class='head'>N<SUP>o</SUP></td><td class='head' colspan='2'>球員</td><td class='head'>得</td><td class='head'>總</td>...</tr>
<tr><td class='largeFont_1'>2</td><td><a href='Player.aspx?CupID=21&PlayerID=124'>張瓈文</a></td><td>對角</td><td class='lightBackground'>2</td><td>5</td><td class='lightBackground'>0</td><td class='lightBackground'>1</td><td>4</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td class='lightBackground'>3</td></tr>
...
<tr><td class='head' colspan='3'>全隊合計</td>...</tr>
</table></div>
<h3>義力營造：林雍順、康金塗、莊明叡</h3>
<div class='TableFormat_1'><table>...同上結構...</table></div>
```

關鍵事實（都已用實際球員資料交叉驗證，見下方「位置代碼對照」）：

1. 整份文件的第一個 `<h3>` 是比賽標題（性別、週次或賽別、場館、比賽編號、日期時間）；之後每個 `<h3>` 是「隊名：教練名單」，緊接著的 `<table>`（用 `team_h3.find_next("table")` 取得）才是該隊的出賽名單表。
2. 每個球員一列的欄位順序（扣掉表頭 2 列與尾端「全隊合計」列）固定是 **15 欄**：`[0]背號 [1]姓名(含<a>連結) [2]位置(中文) [3]攻擊得 [4]攻擊總 [5]攔網得 [6]發球得 [7]發球總 [8]接發好 [9]接發總 [10]防守好 [11]防守總 [12]舉球好 [13]舉球總 [14]總得分`。
3. **這個欄位順序跟 `stats_crawler.py` 現有的 `fetch_player_stats()`（`_handler/Player.ashx`）完全不同**：`Player.ashx` 是「總在前、得/好在後」（`attack_total, attack_points, ..., receive_total, receive_excellent, ...`），`Match.ashx` 是「得/好在前、總在後，且攻擊是得在前總在後」（`attack_points, attack_total, block_points, serve_points, serve_total, receive_excellent, receive_total, dig_excellent, dig_total, set_excellent, set_total`）。**寫 parser 時絕對不能複製貼上 `fetch_player_stats()` 的欄位映射假設**，這是本次規劃 session 中最容易誤踩的坑，已用實際數值反推驗證過（詳見下一點）。
4. `Match.ashx` **完全沒有「局數 (sets_played)」欄位**。因此 Phase 2 的 `player_match_stats.sets_played` 仍然只能來自既有的 `Player.ashx`（`stats_crawler.fetch_player_stats()`），`Match.ashx` 只負責背號/位置/週次歸屬，兩者互補、不是互相取代。
5. 位置中文與現有 `constants.POSITION_MAP`（官網用語：主攻手/中間手/副攻手/舉球員/自由球員）**用語不同**。已用資料庫既有球員資料交叉比對驗證出對照（例如「新北中纖 #2 張瓈文」在 `Match.ashx` 顯示位置「對角」，資料庫既有 `players.position = 'OP'`；「新北中纖 #17 杜家馨」顯示「攔中」，資料庫是 `'MB'`）：

   | Match.ashx 中文 | 對照代碼 | 驗證用球員（姓名／背號／隊伍） |
   |---|---|---|
   | 對角 | OP | 張瓈文 #2 新北中纖 |
   | 長攻 | OH | 劉映彤 #6 新北中纖 |
   | 攔中 | MB | 杜家馨 #17 新北中纖 |
   | 舉球 | S  | 陳妘臻 #9 新北中纖 |
   | 自由 | L  | 范張予馨 #16 新北中纖 |

6. `MatchID` 與官網 `match_crawler.py` 用的 `game_id`（`/game/{id}`、`/wgame/{id}`）是**完全不同的編號空間**，不能互相假設對應。兩者的橋接鍵是 `match_date`（見下方 week_label 反查邏輯）。
7. 黃金決勝局（golden set）在 `Match.aspx` 的下拉選單中是**獨立的 `MatchID`**，標籤帶 `-1` 後綴（例如 `"總決賽-女 115-1：高雄台電 vs 臺北鯨華 (03月22日)"` 對應另一個 `MatchID`）。本 phase 的出賽名單爬蟲**不需要特別處理黃金局**：黃金局的 `MatchID` 一樣會被 `fetch_match_list()` 列出、一樣能正常呼叫 `fetch_match_roster()` 取得该局出賽名單，只是通常球員較少（板凳輪替）。因為 `roster_registrations` 的粒度是「球員×週次×隊伍」（不分是否黃金局），同一球員同一週出現在黃金局與正規局都會 upsert 到同一列，不會重複也不會遺漏。

**待確認（若執行 Task 3 時外部系統網路不通，用以下替代驗證步驟）**：本規劃 session 用 `curl --max-time 12` 對 `http://114.35.229.141` 實測成功（HTTP 200，回傳上述真實 HTML）。若執行時無法連線，替代驗證：改用 `tests/fixtures/match_ashx_sample.html`（Task 3 會建立，內容就是本次實測抓到的真實回應）先把 parser 邏輯開發完、單元測試打綠，再排程之後網路恢復時才真的執行 `crawl_all_rosters()` 對外抓取。**不要**因為連不上就用猜測的 HTML 結構寫 parser。

**Week Label 反查的驗證基礎**：本 session 已用 SQL 驗證 `player_match_stats` 現有全部 3,807 筆資料涵蓋的 `match_date`，每一個都能在 `matches` 表找到唯一一筆 `round_name`（無一對多、無查無資料的情形，唯一的 6 筆 `round_name IS NULL` 都落在 2025-03-22~03-24，是上一季的殘留資料，日期早於 `player_match_stats` 最早的 `2025-11-01`，不影響本次遷移）。因此「用 `match_date` 反查 `matches.round_name` 當作 `week_label`」這個設計對現有資料是可靠的。

---

## Task 1：新 schema.sql 定案 + 備份腳本 + 冪等性測試

**Files:**
- Modify: `sql/schema.sql`（拆分 `players`、新增 `roster_registrations`、`player_match_stats` FK 改指向 `roster_registrations`）
- Create: `src/etl/backup_db.py`
- Create: `tests/test_schema_v2.py`

**Interfaces:**
- Produces：
  ```python
  def backup_database(db_path: Path | None = None) -> Path:
      """複製 data/db/tvl_database.db 到帶時間戳的備份檔，回傳備份檔路徑。"""
  ```
- Consumes：`src.utils.db_config.DB_PATH`。

**步驟：**

- [ ] **Step 1:** 編輯 `sql/schema.sql`，把 `players` 表定義（原本含 `team_id, jersey_number, position`）改為身分層 only，並新增 `roster_registrations`，`player_match_stats` 的 FK 從 `player_id` 改成 `registration_id`：
   ```sql
   -- TVL 資料庫 Schema（可重複執行，冪等：僅 CREATE TABLE IF NOT EXISTS，不清空既有資料）
   -- players = 球員身分層（跨週不變的屬性）；roster_registrations = 球員週次登錄層
   -- （球員×週次×隊伍×背號×位置，因企業排球每週登錄名單可能不同）
   -- 注意：男女組的 team_id 可能重複，因此 teams 使用複合主鍵 (team_id, gender)

   CREATE TABLE IF NOT EXISTS teams (
       team_id   INTEGER NOT NULL,
       team_name TEXT    NOT NULL,
       gender    TEXT    NOT NULL CHECK (gender IN ('M', 'F')),
       PRIMARY KEY (team_id, gender)
   );

   CREATE TABLE IF NOT EXISTS players (
       player_id  INTEGER PRIMARY KEY AUTOINCREMENT,
       name       TEXT,
       gender     TEXT NOT NULL CHECK (gender IN ('M', 'F')),
       dob        DATE,
       height_cm  REAL,
       weight_kg  REAL
   );

   CREATE TABLE IF NOT EXISTS roster_registrations (
       registration_id  INTEGER PRIMARY KEY AUTOINCREMENT,
       player_id        INTEGER NOT NULL,
       team_id          INTEGER NOT NULL,
       gender           TEXT    NOT NULL CHECK (gender IN ('M', 'F')),
       week_label       TEXT    NOT NULL,
       week_start_date  DATE,
       jersey_number    INTEGER,
       position         TEXT,
       source           TEXT    NOT NULL CHECK (source IN ('match_page', 'backfill')),
       FOREIGN KEY (player_id) REFERENCES players (player_id),
       FOREIGN KEY (team_id, gender) REFERENCES teams (team_id, gender),
       UNIQUE (player_id, team_id, gender, week_label)
   );

   CREATE TABLE IF NOT EXISTS player_match_stats (
       stat_id           INTEGER PRIMARY KEY AUTOINCREMENT,
       registration_id   INTEGER NOT NULL,
       match_date        DATE,
       opponent          TEXT,
       sets_played       INTEGER,
       attack_total      INTEGER,
       attack_points     INTEGER,
       block_points      INTEGER,
       serve_total       INTEGER,
       serve_points      INTEGER,
       receive_total     INTEGER,
       receive_excellent INTEGER,
       dig_total         INTEGER,
       dig_excellent     INTEGER,
       set_total         INTEGER,
       set_excellent     INTEGER,
       total_points      INTEGER,
       is_golden_set     INTEGER NOT NULL DEFAULT 0 CHECK (is_golden_set IN (0, 1)),
       FOREIGN KEY (registration_id) REFERENCES roster_registrations (registration_id)
   );

   CREATE TABLE IF NOT EXISTS matches (
       match_id        INTEGER PRIMARY KEY AUTOINCREMENT,
       game_id         INTEGER NOT NULL,
       gender          TEXT NOT NULL CHECK (gender IN ('M', 'F')),
       match_date      DATE NOT NULL,
       venue           TEXT,
       round_name      TEXT,
       game_label      TEXT,
       is_golden_set   INTEGER NOT NULL DEFAULT 0,
       home_team       TEXT NOT NULL,
       away_team       TEXT NOT NULL,
       home_set1       INTEGER,
       home_set2       INTEGER,
       home_set3       INTEGER,
       home_set4       INTEGER,
       home_set5       INTEGER,
       home_total      INTEGER,
       away_set1       INTEGER,
       away_set2       INTEGER,
       away_set3       INTEGER,
       away_set4       INTEGER,
       away_set5       INTEGER,
       away_total      INTEGER,
       home_sets_won   INTEGER,
       away_sets_won   INTEGER,
       UNIQUE (game_id, gender)
   );

   -- 效能索引
   CREATE INDEX IF NOT EXISTS idx_pms_registration_id ON player_match_stats(registration_id);
   CREATE INDEX IF NOT EXISTS idx_pms_match_date       ON player_match_stats(match_date);
   CREATE INDEX IF NOT EXISTS idx_roster_player        ON roster_registrations(player_id);
   CREATE INDEX IF NOT EXISTS idx_roster_team_gender   ON roster_registrations(team_id, gender);
   CREATE INDEX IF NOT EXISTS idx_roster_week          ON roster_registrations(week_label);
   CREATE INDEX IF NOT EXISTS idx_matches_date         ON matches(match_date);
   ```

   > 這是**目標形態**：套用到現有資料庫之前，現有的 `players`/`player_match_stats` 兩表已經存在且欄位不同（`players` 還有 `team_id/jersey_number/position`，`player_match_stats` 還是 `player_id` 而非 `registration_id`）。**不要在這一步就對正式 DB 執行這份新 schema**——`CREATE TABLE IF NOT EXISTS` 遇到同名但欄位不同的既有表會直接略過、不會報錯也不會更新結構，這正是 Task 4 遷移腳本要處理的事（先 rename 舊表，再套用這份新 schema 建全新表，再搬資料）。本 task 只負責把「目標形態」寫定，並驗證這份 DDL 本身語法正確、在全新空白 DB 上可以正確建表。

- [ ] **Step 2:** 用一個全新的記憶體 DB 驗證新 schema 語法正確、且是真正冪等（可重複執行）：
   ```bash
   python3 -c "
   import sqlite3
   sql = open('sql/schema.sql', encoding='utf-8').read()
   conn = sqlite3.connect(':memory:')
   conn.execute('PRAGMA foreign_keys = ON')
   conn.executescript(sql)
   conn.executescript(sql)
   tables = {r[0] for r in conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")}
   assert tables == {'teams', 'players', 'roster_registrations', 'player_match_stats', 'matches'}, tables
   print('schema v2 OK:', sorted(tables))
   "
   ```
   **預期輸出**：`schema v2 OK: ['matches', 'player_match_stats', 'players', 'roster_registrations', 'teams']`

- [ ] **Step 3:** 用一筆假資料驗證 FK 鏈與 `UNIQUE (player_id, team_id, gender, week_label)` 約束確實生效：
   ```bash
   python3 -c "
   import sqlite3
   sql = open('sql/schema.sql', encoding='utf-8').read()
   conn = sqlite3.connect(':memory:')
   conn.execute('PRAGMA foreign_keys = ON')
   conn.executescript(sql)

   conn.execute(\"INSERT INTO teams (team_id, team_name, gender) VALUES (1, '測試隊', 'F')\")
   conn.execute(\"INSERT INTO players (name, gender) VALUES ('測試球員', 'F')\")
   pid = conn.execute(\"SELECT player_id FROM players\").fetchone()[0]
   conn.execute(
       \"INSERT INTO roster_registrations (player_id, team_id, gender, week_label, jersey_number, position, source) VALUES (?, 1, 'F', '例行賽 Week 1', 5, 'OH', 'match_page')\",
       (pid,),
   )
   rid = conn.execute(\"SELECT registration_id FROM roster_registrations\").fetchone()[0]
   conn.execute(
       \"INSERT INTO player_match_stats (registration_id, match_date, total_points) VALUES (?, '2026-01-01', 10)\",
       (rid,),
   )

   # 重複 (player_id, team_id, gender, week_label) 應該違反 UNIQUE
   try:
       conn.execute(
           \"INSERT INTO roster_registrations (player_id, team_id, gender, week_label, jersey_number, position, source) VALUES (?, 1, 'F', '例行賽 Week 1', 6, 'MB', 'match_page')\",
           (pid,),
       )
       raise SystemExit('應該要違反 UNIQUE 約束卻沒有！')
   except sqlite3.IntegrityError as e:
       print('UNIQUE 約束正常擋下重複列：', e)

   # 無效 registration_id 應該違反 FK
   try:
       conn.execute(
           \"INSERT INTO player_match_stats (registration_id, match_date, total_points) VALUES (99999, '2026-01-01', 1)\"
       )
       raise SystemExit('應該要違反 FK 約束卻沒有！')
   except sqlite3.IntegrityError as e:
       print('FK 約束正常擋下無效 registration_id：', e)
   "
   ```
   **預期輸出**：兩行分別印出 `UNIQUE 約束正常擋下重複列：...` 與 `FK 約束正常擋下無效 registration_id：...`，程式正常結束（沒有 `SystemExit`）。

- [ ] **Step 4:** 新增 `src/etl/backup_db.py`：
   ```python
   """
   資料庫備份工具
   在任何會改動 data/db/tvl_database.db 結構或內容的遷移前，先備份成帶時間戳的檔案。
   """

   import shutil
   from datetime import datetime
   from pathlib import Path

   from src.utils.db_config import DB_PATH
   from src.utils.logger import get_logger

   logger = get_logger(__name__)


   def backup_database(db_path: Path | None = None) -> Path:
       """複製 DB 檔到 <db_path>.bak-<timestamp>，回傳備份檔路徑。"""
       source = db_path or DB_PATH
       if not source.exists():
           raise FileNotFoundError(f"找不到要備份的資料庫：{source}")

       timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
       backup_path = source.with_name(f"{source.name}.bak-{timestamp}")
       shutil.copy2(source, backup_path)
       logger.info("已備份資料庫：%s -> %s", source, backup_path)
       return backup_path


   def main():
       path = backup_database()
       print(f"備份完成：{path}")


   if __name__ == "__main__":
       main()
   ```

- [ ] **Step 5:** 新增 `tests/test_schema_v2.py`：
   ```python
   import sqlite3
   from pathlib import Path

   import pytest

   from src.etl.backup_db import backup_database

   SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


   @pytest.fixture
   def conn():
       connection = sqlite3.connect(":memory:")
       connection.execute("PRAGMA foreign_keys = ON")
       connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
       yield connection
       connection.close()


   def test_v2_tables_exist(conn):
       tables = {
           row[0] for row in
           conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
       }
       assert tables == {"teams", "players", "roster_registrations", "player_match_stats", "matches"}


   def test_players_has_no_team_columns(conn):
       cols = {row[1] for row in conn.execute("PRAGMA table_info(players)")}
       assert "team_id" not in cols
       assert "jersey_number" not in cols
       assert "position" not in cols
       assert cols == {"player_id", "name", "gender", "dob", "height_cm", "weight_kg"}


   def test_roster_registrations_unique_constraint(conn):
       conn.execute("INSERT INTO teams (team_id, team_name, gender) VALUES (1, '測試隊', 'F')")
       conn.execute("INSERT INTO players (name, gender) VALUES ('測試球員', 'F')")
       pid = conn.execute("SELECT player_id FROM players").fetchone()[0]
       conn.execute(
           """INSERT INTO roster_registrations
              (player_id, team_id, gender, week_label, jersey_number, position, source)
              VALUES (?, 1, 'F', '例行賽 Week 1', 5, 'OH', 'match_page')""",
           (pid,),
       )
       with pytest.raises(sqlite3.IntegrityError):
           conn.execute(
               """INSERT INTO roster_registrations
                  (player_id, team_id, gender, week_label, jersey_number, position, source)
                  VALUES (?, 1, 'F', '例行賽 Week 1', 6, 'MB', 'match_page')""",
               (pid,),
           )


   def test_player_match_stats_fk_to_registration(conn):
       with pytest.raises(sqlite3.IntegrityError):
           conn.execute(
               "INSERT INTO player_match_stats (registration_id, match_date, total_points) VALUES (99999, '2026-01-01', 1)"
           )


   def test_backup_database_creates_timestamped_copy(tmp_path):
       source = tmp_path / "fake.db"
       source.write_bytes(b"fake sqlite content")
       backup_path = backup_database(db_path=source)
       assert backup_path.exists()
       assert backup_path.name.startswith("fake.db.bak-")
       assert backup_path.read_bytes() == b"fake sqlite content"
   ```

- [ ] **Step 6:** 跑測試：
   ```bash
   python -m pytest tests/test_schema_v2.py -v
   ```
   **預期輸出**：5 個測試 PASSED。

   **未實測**：需執行者在有依賴的 venv 中實跑確認。

- [ ] **Step 7:** 確認行尾乾淨：`git diff --stat -w`

- [ ] **Step 8:** **[STOP：等待使用者同意 commit]**：
   ```
   feat: schema.sql 拆分 players 為身分層 + 新增 roster_registrations

   players 收斂為姓名/性別/生日/身高體重等跨週不變屬性；新增
   roster_registrations 承載球員×週次×隊伍×背號×位置；
   player_match_stats 改 FK 到 roster_registrations。此為目標
   schema 定義，尚未套用遷移（見後續 task）。
   ```

---

## Task 2：constants.py 新增位置代碼對照表

**Files:**
- Modify: `src/utils/constants.py`（新增 `MATCH_POSITION_MAP`）
- Create: `tests/test_constants_match_position.py`

**Interfaces:**
- Produces：`src.utils.constants.MATCH_POSITION_MAP: dict[str, str]`
- Consumes：無（純資料常數）。

**步驟：**

- [ ] **Step 1:** 編輯 `src/utils/constants.py`，在既有 `POSITION_MAP` 定義之後新增：
   ```python
   # ── 外部統計系統（Match.ashx）位置用語 → 內部代碼 ─────────────
   # 用語與官網 POSITION_MAP 不同（官網：主攻手/中間手/副攻手/舉球員/自由球員），
   # 已用資料庫既有球員資料交叉驗證（見 Phase 2 計畫文件）。
   MATCH_POSITION_MAP = {
       "對角": "OP",
       "長攻": "OH",
       "攔中": "MB",
       "舉球": "S",
       "自由": "L",
   }
   ```

- [ ] **Step 2:** 新增 `tests/test_constants_match_position.py`：
   ```python
   from src.utils.constants import MATCH_POSITION_MAP, VALID_POSITIONS


   def test_match_position_map_covers_five_positions():
       assert set(MATCH_POSITION_MAP.keys()) == {"對角", "長攻", "攔中", "舉球", "自由"}


   def test_match_position_map_values_are_valid_codes():
       assert set(MATCH_POSITION_MAP.values()) == VALID_POSITIONS


   def test_known_player_position_mappings():
       # 已用 DB 既有資料驗證過的具體對照（見計畫文件的驗證表格）
       verified = {
           "對角": "OP",  # 張瓈文 #2 新北中纖
           "長攻": "OH",  # 劉映彤 #6 新北中纖
           "攔中": "MB",  # 杜家馨 #17 新北中纖
           "舉球": "S",   # 陳妘臻 #9 新北中纖
           "自由": "L",   # 范張予馨 #16 新北中纖
       }
       for raw, expected in verified.items():
           assert MATCH_POSITION_MAP[raw] == expected
   ```

- [ ] **Step 3:** 跑測試：
   ```bash
   python -m pytest tests/test_constants_match_position.py -v
   ```
   **預期輸出**：3 個測試 PASSED。

- [ ] **Step 4:** 確認行尾乾淨：`git diff --stat -w`

- [ ] **Step 5:** **[STOP：等待使用者同意 commit]**：
   ```
   feat: constants.py 新增外部系統位置用語對照表 MATCH_POSITION_MAP

   Match.ashx 出賽名單頁的位置用語（對角/長攻/攔中/舉球/自由）與
   官網 POSITION_MAP 不同，新增獨立對照表供 Phase 2 roster 爬蟲使用。
   ```

---

## Task 3：新增出賽名單爬蟲（`stats_crawler.py`）

**Files:**
- Modify: `src/etl/stats_crawler.py`（新增 `fetch_match_list`、`fetch_match_roster`、`resolve_week_label`、`upsert_roster_registration`、`crawl_all_rosters` 函式）
- Create: `tests/fixtures/match_ashx_sample.html`
- Create: `tests/test_roster_crawler.py`

**Interfaces:**
- Produces：
  ```python
  def fetch_match_list(cup_id: int = CUP_ID) -> list[dict]:
      """回傳 [{'match_id': int, 'label': str}, ...]，來源是 Match.aspx 的 <select id='divSelect'>。"""

  def fetch_match_roster(cup_id: int, match_id: int) -> list[dict] | None:
      """
      回傳該場出賽名單，每位球員一筆 dict：
      {'match_date': str, 'team_id': int, 'team_gender': str,
       'jersey_number': int | None, 'name': str, 'position': str | None}
      頁面無效（如錯誤頁）回傳 None。
      """

  def resolve_week_label(conn, match_date: str, title_text: str) -> tuple[str, str]:
      """回傳 (week_label, week_start_date)。優先用 matches.round_name 反查，查無則退化用標題文字。"""

  def upsert_roster_registration(conn, player_id: int, row: dict, week_label: str, week_start_date: str) -> None:
      """upsert 一筆 roster_registrations，source='match_page'。"""

  def crawl_all_rosters(conn, cup_id: int = CUP_ID) -> dict:
      """批次爬取全部場次的出賽名單並 upsert 進 roster_registrations，回傳統計 dict。"""
  ```
- Consumes：`src.utils.constants.MATCH_POSITION_MAP`（Task 2 新增）、`OPP_SHORT_TO_TEAM`、既有的 `normalize_name`、`safe_int`、`EXT_BASE`、`HEADERS`、`SEASON_YEAR_MAP`、`DEFAULT_YEAR`、`build_name_to_pid`（`stats_crawler.py` 既有函式，本 task 直接複用，不重寫）。

**步驟：**

- [ ] **Step 1:** 建立測試 fixture `tests/fixtures/match_ashx_sample.html`，內容就是本規劃 session 用 `curl` 實際抓到的真實回應（`CupID=21&MatchID=1`，女子組 第1週 新北中纖 vs 義力營造）：
   ```html
   <h3><img src='_images/Sex_2.png' height='32' />女子組 第1週(成功大學體育館) 編號：1 (11月1日 13:00)  歷時 02:22    <a id='21_1' href='P2/21_1.pdf'>P2</a></h3>
   <div class='TableFormat_1'><table cellpadding='0' cellspacing='0' align='center' width='100%'  >
   <tr><td><div class='MatchResult'><table cellpadding='0' cellspacing='0' align='center' width='100%'  >
   <tr><td class='TeamName  lightBackground'><a href='Team.aspx?CupID=21&TeamID=8'><a href='Team.aspx?CupID=21&TeamID=8' style='font-size:28px; vertical-align:central;'><img src='_images\TeamLogo\21_8.png' height='28' />&nbsp;新北中纖</a></a></td><td class='Score largeFont_3' style='color:red'>25</td><td class='Score largeFont_3' style='color:red'>25</td><td class='Score largeFont_3'>21</td><td class='Score largeFont_3'>19</td><td class='Score largeFont_3' style='color:red'>15</td><td class='Final largeFont_3' style='color:red'>105</td><td class='Final' style='color:red'>3</td></tr>
   <tr><td class='TeamName '><a href='Team.aspx?CupID=21&TeamID=9'><a href='Team.aspx?CupID=21&TeamID=9' style='font-size:28px; vertical-align:central;'><img src='_images\TeamLogo\21_9.png' height='28' />&nbsp;義力營造</a></a></td><td class='Score largeFont_3'>20</td><td class='Score largeFont_3'>15</td><td class='Score largeFont_3' style='color:red'>25</td><td class='Score largeFont_3' style='color:red'>25</td><td class='Score largeFont_3'>12</td><td class='Final largeFont_3'>97</td><td class='Final'>2</td></tr>
   </table>
   </div></td><td class='Top'><div class='TableFormat_1'><table cellpadding='0' cellspacing='0' align='center' width='100%'  >
   <tr><th>第一裁判</td></th><td>許耿豪</td></td></tr>
   <tr><th>第二裁判</td></th><td>劉鎰源</td></td></tr>
   </table>
   </div></td></tr>
   </table>
   </div>

   <h3>新北中纖：邱雅慧、簡佳慧</h3>
   <br/>
   <div class='TableFormat_1'><table cellpadding='0' cellspacing='0' align='center' width='100%'  >
   <tr><td class='head' colspan='3'>新北中纖</td></td><td class='head' colspan='2'>攻擊(Attack)</td></td><td class='head' colspan='1'>攔網(Block)</td></td><td class='head' colspan='2'>發球(Serve)</td></td><td class='head' colspan='2'>接發(Receive)</td></td><td class='head' colspan='2'>防守(Dig)</td></td><td class='head' colspan='2'>舉球(Set)</td></td><td class='head' rowspan='2'>總得分<br/>(Points)</td></td></tr>
   <tr><td class='head'>N<SUP>o</SUP></td></td><td class='head' colspan='2'>球員</td></td><td class='head'>得</td></td><td class='head'>總</td></td><td class='head'>得</td></td><td class='head'>得</td></td><td class='head'>總</td></td><td class='head'>好</td></td><td class='head'>總</td></td><td class='head'>好</td></td><td class='head'>總</td></td><td class='head'>好</td></td><td class='head'>總</td></td></tr>
   <tr><td class='largeFont_1'>2</td></td><td><a href='Player.aspx?CupID=21&PlayerID=124'>張瓈文</a></td><td>對角</td></td><td class='lightBackground'>2</td></td><td>5</td></td><td class='lightBackground'>0</td></td><td class='lightBackground'>1</td></td><td>4</td></td><td>0</td></td><td>0</td></td><td>0</td></td><td>0</td></td><td>0</td></td><td>0</td></td><td class='lightBackground'>3</td></td></tr>
   <tr><td class='largeFont_1'>6</td></td><td><a href='Player.aspx?CupID=21&PlayerID=119'>劉映彤</a></td><td>長攻</td></td><td class='lightBackground'>0</td></td><td>0</td></td><td class='lightBackground'>0</td></td><td class='lightBackground'>0</td></td><td>4</td></td><td>0</td></td><td>1</td></td><td>1</td></td><td>3</td></td><td>0</td></td><td>0</td></td><td class='lightBackground'>0</td></td></tr>
   <tr><td class='largeFont_1'>16</td></td><td><a href='Player.aspx?CupID=21&PlayerID=125'>范張予馨</a></td><td>自由</td></td><td class='lightBackground'>0</td></td><td>0</td></td><td class='lightBackground'>0</td></td><td class='lightBackground'>0</td></td><td>0</td></td><td>19</td></td><td>32</td></td><td>2</td></td><td>6</td></td><td>0</td></td><td>0</td></td><td class='lightBackground'>0</td></td></tr>
   <tr><td class='largeFont_1'>17</td></td><td><a href='Player.aspx?CupID=21&PlayerID=121'>杜家馨</a></td><td>攔中</td></td><td class='lightBackground'>3</td></td><td>6</td></td><td class='lightBackground'>0</td></td><td class='lightBackground'>0</td></td><td>9</td></td><td>0</td></td><td>0</td></td><td>0</td></td><td>1</td></td><td>0</td></td><td>0</td></td><td class='lightBackground'>3</td></td></tr>
   <tr><td class='largeFont_1'>9</td></td><td><a href='Player.aspx?CupID=21&PlayerID=114'>陳妘臻</a></td><td>舉球</td></td><td class='lightBackground'>0</td></td><td>2</td></td><td class='lightBackground'>0</td></td><td class='lightBackground'>0</td></td><td>4</td></td><td>0</td></td><td>0</td></td><td>0</td></td><td>0</td></td><td>2</td></td><td>12</td></td><td class='lightBackground'>0</td></td></tr>
   <tr><td class='head' colspan='3'>全隊合計</td></td><td>61</td></td><td>139</td></td><td>6</td></td><td>7</td></td><td>104</td></td><td>39</td></td><td>64</td></td><td>8</td></td><td>20</td></td><td>24</td></td><td>119</td></td><td>74</td></td></tr>
   </table>
   </div>
   <h3>義力營造：林雍順、康金塗、莊明叡</h3>
   <br/>
   <div class='TableFormat_1'><table cellpadding='0' cellspacing='0' align='center' width='100%'  >
   <tr><td class='head' colspan='3'>義力營造</td></td><td class='head' colspan='2'>攻擊(Attack)</td></td><td class='head' colspan='1'>攔網(Block)</td></td><td class='head' colspan='2'>發球(Serve)</td></td><td class='head' colspan='2'>接發(Receive)</td></td><td class='head' colspan='2'>防守(Dig)</td></td><td class='head' colspan='2'>舉球(Set)</td></td><td class='head' rowspan='2'>總得分<br/>(Points)</td></td></tr>
   <tr><td class='head'>N<SUP>o</SUP></td></td><td class='head' colspan='2'>球員</td></td><td class='head'>得</td></td><td class='head'>總</td></td><td class='head'>得</td></td><td class='head'>得</td></td><td class='head'>總</td></td><td class='head'>好</td></td><td class='head'>總</td></td><td class='head'>好</td></td><td class='head'>總</td></td><td class='head'>好</td></td><td class='head'>總</td></td></tr>
   <tr><td class='largeFont_1'>1</td></td><td><a href='Player.aspx?CupID=21&PlayerID=130'>江辰翊</a></td><td>舉球</td></td><td class='lightBackground'>0</td></td><td>0</td></td><td class='lightBackground'>0</td></td><td class='lightBackground'>0</td></td><td>5</td></td><td>0</td></td><td>0</td></td><td>0</td></td><td>0</td></td><td>2</td></td><td>19</td></td><td class='lightBackground'>0</td></td></tr>
   <tr><td class='largeFont_1'>3</td></td><td><a href='Player.aspx?CupID=21&PlayerID=133'>曾予柔</a></td><td>對角</td></td><td class='lightBackground'>3</td></td><td>10</td></td><td class='lightBackground'>0</td></td><td class='lightBackground'>1</td></td><td>10</td></td><td>0</td></td><td>1</td></td><td>5</td></td><td>5</td></td><td>0</td></td><td>0</td></td><td class='lightBackground'>4</td></td></tr>
   <tr><td class='largeFont_1'>6</td></td><td><a href='Player.aspx?CupID=21&PlayerID=135'>陳品宇</a></td><td>長攻</td></td><td class='lightBackground'>22</td></td><td>41</td></td><td class='lightBackground'>0</td></td><td class='lightBackground'>2</td></td><td>19</td></td><td>10</td></td><td>12</td></td><td>3</td></td><td>4</td></td><td>0</td></td><td>0</td></td><td class='lightBackground'>24</td></td></tr>
   <tr><td class='largeFont_1'>7</td></td><td><a href='Player.aspx?CupID=21&PlayerID=134'>吳紫華</a></td><td>攔中</td></td><td class='lightBackground'>9</td></td><td>26</td></td><td class='lightBackground'>0</td></td><td class='lightBackground'>1</td></td><td>10</td></td><td>1</td></td><td>1</td></td><td>1</td></td><td>3</td></td><td>0</td></td><td>0</td></td><td class='lightBackground'>10</td></td></tr>
   <tr><td class='largeFont_1'>16</td></td><td><a href='Player.aspx?CupID=21&PlayerID=129'>江昀庭</a></td><td>自由</td></td><td class='lightBackground'>1</td></td><td>1</td></td><td class='lightBackground'>0</td></td><td class='lightBackground'>0</td></td><td>0</td></td><td>12</td></td><td>16</td></td><td>2</td></td><td>2</td></td><td>0</td></td><td>1</td></td><td class='lightBackground'>1</td></td></tr>
   <tr><td class='head' colspan='3'>全隊合計</td></td><td>57</td></td><td>148</td></td><td>2</td></td><td>6</td></td><td>98</td></td><td>41</td></td><td>69</td></td><td>18</td></td><td>29</td></td><td>29</td></td><td>124</td></td><td>65</td></td></tr>
   </table>
   </div>
   ```
   （為了測試檔精簡，每隊只保留 5 位球員的列，已涵蓋 5 種位置各一位；真實頁面每隊約 12-14 位球員，parser 邏輯對兩者行為一致，不影響測試有效性。）

- [ ] **Step 2:** 在 `src/etl/stats_crawler.py` 現有 import 區塊之後（不動既有函式），新增以下常數與函式：
   ```python
   from src.utils.constants import (
       EXT_BASE, EXT_CUP_ID as CUP_ID, EXT_HEADERS as HEADERS,
       SEASON_YEAR_MAP, DEFAULT_YEAR, EXT_TEAM_MAP,
       MATCH_POSITION_MAP, OPP_SHORT_TO_TEAM,
   )
   ```
   （修改既有 import 陳述式，把 `MATCH_POSITION_MAP, OPP_SHORT_TO_TEAM` 加進去。）

   在檔案尾端（`if __name__ == "__main__":` 之前）新增：
   ```python
   # ── 出賽名單爬蟲（Phase 2：週次登錄資料來源） ──────────────────

   def fetch_match_list(cup_id: int = CUP_ID) -> list[dict]:
       """
       透過 Match.aspx 的下拉選單取得所有 MatchID 清單。
       MatchID 本身不連續，不可用逐一 range 掃描列舉，必須用此函式。
       """
       url = f"{EXT_BASE}/Match.aspx"
       r = requests.get(url, params={"CupID": cup_id}, headers=HEADERS, timeout=15)
       r.raise_for_status()
       r.encoding = "utf-8"
       soup = BeautifulSoup(r.text, "html.parser")
       sel = soup.find("select", id="divSelect")
       if not sel:
           return []
       result = []
       for opt in sel.find_all("option"):
           value = opt.get("value")
           if not value:
               continue
           result.append({"match_id": int(value), "label": opt.get_text(strip=True)})
       return result


   def _parse_match_title(title_text: str) -> tuple[str, str] | None:
       """從標題文字解析 (match_date, raw_round_text)。解析失敗回傳 None。"""
       date_m = re.search(r"(\d{1,2})月(\d{1,2})日", title_text)
       if not date_m:
           return None
       month, day = int(date_m.group(1)), int(date_m.group(2))
       year = SEASON_YEAR_MAP.get(month, DEFAULT_YEAR)
       match_date = f"{year}-{month:02d}-{day:02d}"

       round_m = re.search(r"(第\d+週|挑戰賽|總決賽|季後賽|明星賽)", title_text)
       raw_round_text = round_m.group(1) if round_m else "未知賽別"
       return match_date, raw_round_text


   def fetch_match_roster(cup_id: int, match_id: int) -> list[dict] | None:
       """
       抓取單場出賽名單，回傳每位球員一筆 dict：
       {'match_date', 'title_text', 'team_id', 'team_gender',
        'jersey_number', 'name', 'position'}
       頁面無效（如錯誤頁、無比賽資料）回傳 None。

       欄位順序注意（與 fetch_player_stats() 不同，不可混用）：
       每列扣除背號/姓名/位置後，共 12 個數值欄位，順序固定為
       [攻擊得, 攻擊總, 攔網得, 發球得, 發球總,
        接發好, 接發總, 防守好, 防守總, 舉球好, 舉球總, 總得分]
       """
       url = f"{EXT_BASE}/_handler/Match.ashx"
       r = requests.get(
           url, params={"CupID": cup_id, "MatchID": match_id},
           headers=HEADERS, timeout=15,
       )
       r.raise_for_status()
       r.encoding = "utf-8"
       soup = BeautifulSoup(r.text, "html.parser")

       title_h3 = soup.find("h3")
       if title_h3 is None:
           return None
       title_text = title_h3.get_text(" ", strip=True)
       if "組" not in title_text:
           return None  # 錯誤頁不含「組」字

       parsed = _parse_match_title(title_text)
       if parsed is None:
           logger.warning("[MatchID=%d] 無法解析日期，跳過：%s", match_id, title_text)
           return None
       match_date, _raw_round_text = parsed

       team_h3s = soup.find_all("h3")[1:]
       rows = []
       for team_h3 in team_h3s:
           team_text = team_h3.get_text(strip=True)
           if "：" not in team_text:
               continue
           team_name = team_text.split("：", 1)[0].strip()
           team_info = OPP_SHORT_TO_TEAM.get(team_name)
           if team_info is None:
               logger.warning("[MatchID=%d] 無法辨識隊名：%s，跳過該隊", match_id, team_name)
               continue
           team_id, team_gender = team_info

           table = team_h3.find_next("table")
           if table is None:
               continue

           for tr in table.find_all("tr")[2:]:
               cells = [td.get_text(strip=True) for td in tr.find_all("td")]
               if not cells or cells[0] in ("全隊合計", ""):
                   continue
               if len(cells) < 15:
                   continue

               position_raw = cells[2]
               rows.append({
                   "match_date": match_date,
                   "title_text": title_text,
                   "team_id": team_id,
                   "team_gender": team_gender,
                   "jersey_number": safe_int(cells[0]),
                   "name": cells[1],
                   "position": MATCH_POSITION_MAP.get(position_raw),
               })

       return rows


   def resolve_week_label(conn: sqlite3.Connection, match_date: str, title_text: str) -> tuple[str, str]:
       """
       回傳 (week_label, week_start_date)。
       優先用 matches.round_name 反查（權威來源）；查無則退化用標題文字，
       並記錄警告（此為已知限制，非本次遷移的資料涵蓋範圍）。
       """
       row = conn.execute(
           "SELECT round_name FROM matches WHERE match_date = ? LIMIT 1", (match_date,)
       ).fetchone()
       if row and row[0]:
           week_label = row[0]
           start_row = conn.execute(
               "SELECT MIN(match_date) FROM matches WHERE round_name = ?", (week_label,)
           ).fetchone()
           week_start_date = start_row[0] if start_row and start_row[0] else match_date
           return week_label, week_start_date

       logger.warning(
           "match_date=%s 在 matches 表查無 round_name，退化用標題文字：%s",
           match_date, title_text,
       )
       parsed = _parse_match_title(title_text)
       raw_round_text = parsed[1] if parsed else "未知賽別"
       return f"未比對-{raw_round_text}", match_date


   def upsert_roster_registration(
       conn: sqlite3.Connection, player_id: int, row: dict,
       week_label: str, week_start_date: str,
   ) -> None:
       """upsert 一筆 roster_registrations，source 固定為 'match_page'（真實出賽名單）。"""
       conn.execute(
           """
           INSERT INTO roster_registrations
               (player_id, team_id, gender, week_label, week_start_date, jersey_number, position, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'match_page')
           ON CONFLICT (player_id, team_id, gender, week_label) DO UPDATE SET
               jersey_number = excluded.jersey_number,
               position = excluded.position,
               week_start_date = excluded.week_start_date,
               source = 'match_page'
           """,
           (player_id, row["team_id"], row["team_gender"], week_label,
            week_start_date, row["jersey_number"], row["position"]),
       )


   def crawl_all_rosters(conn: sqlite3.Connection, cup_id: int = CUP_ID) -> dict:
       """批次爬取全部場次出賽名單並 upsert 進 roster_registrations。"""
       name_map = build_name_to_pid(conn)
       match_list = fetch_match_list(cup_id)
       stats = {"matches_scanned": 0, "matches_skipped": 0, "registrations_upserted": 0, "new_players": 0}

       for m in match_list:
           roster_rows = fetch_match_roster(cup_id, m["match_id"])
           if not roster_rows:
               stats["matches_skipped"] += 1
               continue
           stats["matches_scanned"] += 1

           match_date = roster_rows[0]["match_date"]
           title_text = roster_rows[0]["title_text"]
           week_label, week_start_date = resolve_week_label(conn, match_date, title_text)

           for row in roster_rows:
               norm = normalize_name(row["name"])
               player_id = name_map.get(norm)
               if player_id is None:
                   cursor = conn.execute(
                       "INSERT INTO players (name, gender) VALUES (?, ?)",
                       (row["name"], row["team_gender"]),
                   )
                   player_id = cursor.lastrowid
                   name_map[norm] = player_id
                   stats["new_players"] += 1

               upsert_roster_registration(conn, player_id, row, week_label, week_start_date)
               stats["registrations_upserted"] += 1

           conn.commit()
           time.sleep(0.5)

       return stats
   ```

- [ ] **Step 3:** 靜態語法檢查：
   ```bash
   python3 -c "import ast; ast.parse(open('src/etl/stats_crawler.py', encoding='utf-8').read())"
   ```
   **預期輸出**：無錯誤。

- [ ] **Step 4:** 新增 `tests/test_roster_crawler.py`，用 fixture HTML（不連網路）測試 parser 邏輯：
   ```python
   import sqlite3
   from pathlib import Path
   from unittest.mock import patch

   import pytest
   from bs4 import BeautifulSoup

   from src.etl.stats_crawler import (
       fetch_match_roster, resolve_week_label, upsert_roster_registration,
   )

   FIXTURE = Path(__file__).resolve().parent / "fixtures" / "match_ashx_sample.html"


   class _FakeResponse:
       def __init__(self, text: str):
           self.text = text
           self.encoding = "utf-8"

       def raise_for_status(self):
           pass


   def test_fetch_match_roster_parses_fixture():
       fixture_html = FIXTURE.read_text(encoding="utf-8")
       with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(fixture_html)):
           rows = fetch_match_roster(cup_id=21, match_id=1)

       assert rows is not None
       assert len(rows) == 10  # 兩隊各 5 位（fixture 精簡版）

       zhang = next(r for r in rows if r["name"] == "張瓈文")
       assert zhang["jersey_number"] == 2
       assert zhang["position"] == "OP"       # 對角 -> OP
       assert zhang["team_id"] == 5           # 新北中纖
       assert zhang["team_gender"] == "F"
       assert zhang["match_date"] == "2025-11-01"

       du = next(r for r in rows if r["name"] == "杜家馨")
       assert du["position"] == "MB"          # 攔中 -> MB

       libero = next(r for r in rows if r["name"] == "范張予馨")
       assert libero["position"] == "L"       # 自由 -> L


   def test_fetch_match_roster_column_order_not_confused_with_player_ashx():
       """
       張瓈文那一列原始 cells（扣背號/姓名/位置後）是 2,5,0,1,4,0,0,0,0,0,0,3。
       若誤用 Player.ashx 的「總在前得在後」順序，attack_total 會被誤讀成 2 而非 5。
       """
       fixture_html = FIXTURE.read_text(encoding="utf-8")
       with patch("src.etl.stats_crawler.requests.get", return_value=_FakeResponse(fixture_html)):
           rows = fetch_match_roster(cup_id=21, match_id=1)

       zhang = next(r for r in rows if r["name"] == "張瓈文")
       # fetch_match_roster 目前只回傳背號/姓名/位置（roster_registrations 不需要
       # 逐場數值統計），因此這裡驗證的是「沒有把數值欄位誤植進 dict」——
       # dict 中不該出現任何統計欄位鍵名。
       assert set(zhang.keys()) == {
           "match_date", "title_text", "team_id", "team_gender",
           "jersey_number", "name", "position",
       }


   def test_resolve_week_label_uses_matches_round_name(tmp_db_path):
       conn = sqlite3.connect(tmp_db_path)
       conn.execute("""
           CREATE TABLE matches (
               match_id INTEGER PRIMARY KEY, game_id INTEGER, gender TEXT,
               match_date DATE, round_name TEXT, home_team TEXT, away_team TEXT
           )
       """)
       conn.execute(
           "INSERT INTO matches (game_id, gender, match_date, round_name, home_team, away_team) "
           "VALUES (1, 'F', '2025-11-01', '例行賽 Week 1', '新北中纖', '義力營造')"
       )
       conn.commit()

       week_label, week_start = resolve_week_label(conn, "2025-11-01", "女子組 第1週(...) 編號：1")
       assert week_label == "例行賽 Week 1"
       assert week_start == "2025-11-01"
       conn.close()


   def test_resolve_week_label_falls_back_when_no_match_found(tmp_db_path):
       conn = sqlite3.connect(tmp_db_path)
       conn.execute("""
           CREATE TABLE matches (
               match_id INTEGER PRIMARY KEY, game_id INTEGER, gender TEXT,
               match_date DATE, round_name TEXT, home_team TEXT, away_team TEXT
           )
       """)
       conn.commit()

       week_label, week_start = resolve_week_label(conn, "2099-01-01", "女子組 第99週(...) 編號：1")
       assert week_label.startswith("未比對-")
       assert week_start == "2099-01-01"
       conn.close()


   def test_upsert_roster_registration_is_idempotent(tmp_db_path):
       from pathlib import Path as _P
       schema_sql = (_P(__file__).resolve().parents[1] / "sql" / "schema.sql").read_text(encoding="utf-8")
       conn = sqlite3.connect(tmp_db_path)
       conn.execute("PRAGMA foreign_keys = ON")
       conn.executescript(schema_sql)
       conn.execute("INSERT INTO teams (team_id, team_name, gender) VALUES (5, '新北中纖', 'F')")
       conn.execute("INSERT INTO players (name, gender) VALUES ('張瓈文', 'F')")
       pid = conn.execute("SELECT player_id FROM players").fetchone()[0]

       row = {"team_id": 5, "team_gender": "F", "jersey_number": 2, "position": "OP"}
       upsert_roster_registration(conn, pid, row, "例行賽 Week 1", "2025-11-01")
       upsert_roster_registration(conn, pid, row, "例行賽 Week 1", "2025-11-01")  # 重跑一次

       rows = conn.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()
       assert rows[0] == 1, "重跑 upsert 不應產生重複列"
       conn.close()
   ```

- [ ] **Step 5:** 跑測試：
   ```bash
   python -m pytest tests/test_roster_crawler.py -v
   ```
   **預期輸出**：5 個測試 PASSED。

   **未實測**：需執行者在有 `beautifulsoup4`/`requests` 依賴的環境實跑。`test_fetch_match_roster_parses_fixture` 與 `test_fetch_match_roster_column_order_not_confused_with_player_ashx` 兩個測試不連網路（用 `unittest.mock.patch` 取代 `requests.get`），可離線驗證。

- [ ] **Step 6:** （若目前有網路）用真實外部系統驗證一次 `fetch_match_list`，確認回傳筆數與本規劃 session 實測的量級一致：
   ```bash
   python -c "
   from src.etl.stats_crawler import fetch_match_list
   matches = fetch_match_list(cup_id=21)
   print('共', len(matches), '場')
   print(matches[:3])
   print(matches[-3:])
   "
   ```
   **預期輸出**：`共 118 場`（或相近數字，若球季有更新可能略增），且 `matches[-3:]` 應包含 `MatchID` 落在 200 以上的總決賽場次。

   **未實測**：需執行者在有網路與依賴的環境實跑；此步驟結果會隨球季進展而變動，僅供 sanity check，不寫進自動化測試斷言。

- [ ] **Step 7:** 確認行尾乾淨：`git diff --stat -w`

- [ ] **Step 8:** **[STOP：等待使用者同意 commit]**：
   ```
   feat: stats_crawler 新增出賽名單爬蟲（roster_registrations 資料來源）

   新增 fetch_match_list/fetch_match_roster/resolve_week_label/
   upsert_roster_registration/crawl_all_rosters，透過 Match.ashx
   出賽名單頁取得球員週次×背號×位置，週次一律反查 matches.round_name
   （不信任 Match.ashx 標題自己的週次文字，因總決賽/挑戰賽週號會撞號）。
   ```

---

## Task 4：一次性遷移腳本（3,807 筆歷史統計）

**Files:**
- Create: `src/etl/migrate_to_phase2.py`
- Create: `tests/test_migrate_to_phase2.py`

**Interfaces:**
- Produces：
  ```python
  def run_migration(conn: sqlite3.Connection, cup_id: int = CUP_ID) -> dict:
      """
      執行完整遷移：players 拆表 → 爬全量 roster_registrations →
      舊 player_match_stats 逐筆重接 FK → 驗證 → 清掉舊表。
      回傳統計 dict：{'players_migrated', 'registrations_created',
      'stats_migrated', 'stats_backfilled', 'orphans_found'}
      """
  ```
- Consumes：`src.etl.backup_db.backup_database`（Task 1）、`src.etl.stats_crawler.crawl_all_rosters`（Task 3）、`src.utils.db_config.get_connection`。

**設計說明**：本 task 是整個 Phase 2 風險最高的一步，步驟刻意拆得很細、每步都可獨立驗證，且全程在同一個 transaction 之外用「先建新表、跑完整驗證、確認無誤才刪舊表」的方式進行，任何一步失敗都可以從 Task 1 建立的備份檔復原。

**步驟：**

- [ ] **Step 1:** **[備份，任何操作之前]**：
   ```bash
   python -m src.etl.backup_db
   ```
   **預期輸出**：`備份完成：data/db/tvl_database.db.bak-<timestamp>`。記下這個檔名，遷移全程都不要刪除它。

- [ ] **Step 2:** 記錄遷移前的基準數字，供後續驗證比對：
   ```bash
   python3 -c "
   import sqlite3
   conn = sqlite3.connect('data/db/tvl_database.db')
   print('players:', conn.execute('SELECT COUNT(*) FROM players').fetchone()[0])
   print('player_match_stats:', conn.execute('SELECT COUNT(*) FROM player_match_stats').fetchone()[0])
   print('players missing jersey/position:', conn.execute('SELECT COUNT(*) FROM players WHERE jersey_number IS NULL OR position IS NULL').fetchone()[0])
   "
   ```
   **未實測（本規劃 session 已在早期實查階段跑過，數字為 `players: 151`、`player_match_stats: 3807`、`players missing jersey/position: 7`）**：執行者跑之前請重新確認一次，若數字與本文件不同（例如中間有新的爬蟲跑過），以執行當下的實際數字為準，並更新後續驗證步驟的期望值。

- [ ] **Step 3:** 建立 `src/etl/migrate_to_phase2.py`：
   ```python
   """
   Phase 2 一次性遷移腳本
   players 拆表為身分層 + roster_registrations，player_match_stats 改掛 registration_id。
   執行前一律先備份（見 backup_db.py），此腳本本身也會在開頭再備份一次以防萬一。
   """

   import sqlite3

   from src.etl.backup_db import backup_database
   from src.etl.stats_crawler import crawl_all_rosters
   from src.utils.constants import EXT_CUP_ID as CUP_ID
   from src.utils.db_config import PROJECT_ROOT, get_connection
   from src.utils.logger import get_logger

   logger = get_logger(__name__)
   SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


   def _rename_old_tables(conn: sqlite3.Connection) -> None:
       conn.execute("ALTER TABLE players RENAME TO players_old")
       conn.execute("ALTER TABLE player_match_stats RENAME TO player_match_stats_old")
       conn.commit()
       logger.info("已將舊表重新命名為 players_old / player_match_stats_old")


   def _create_new_tables(conn: sqlite3.Connection) -> None:
       conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
       conn.commit()
       logger.info("已依新版 schema.sql 建立 players / roster_registrations / player_match_stats")


   def _migrate_player_identities(conn: sqlite3.Connection) -> int:
       """把 players_old 的身分欄位（保留原 player_id）搬進新 players。"""
       rows = conn.execute(
           "SELECT player_id, name, gender, dob, height_cm, weight_kg FROM players_old"
       ).fetchall()
       conn.executemany(
           "INSERT INTO players (player_id, name, gender, dob, height_cm, weight_kg) VALUES (?, ?, ?, ?, ?, ?)",
           rows,
       )
       conn.commit()
       logger.info("已搬移 %d 筆球員身分資料（player_id 保持不變）", len(rows))
       return len(rows)


   def _backfill_registration(conn: sqlite3.Connection, old_player_row: tuple, week_label: str, week_start_date: str) -> int:
       """
       用舊 players_old 的快照（team_id/gender/jersey_number/position）建一筆
       source='backfill' 的登錄記錄。只在 crawl_all_rosters() 抓不到真實出賽
       名單時才會呼叫（例如爬蟲涵蓋範圍外的週次、或該球員該週未被系統記錄）。
       """
       player_id, team_id, gender, jersey_number, position = old_player_row
       cur = conn.execute(
           """
           INSERT INTO roster_registrations
               (player_id, team_id, gender, week_label, week_start_date, jersey_number, position, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'backfill')
           ON CONFLICT (player_id, team_id, gender, week_label) DO NOTHING
           """,
           (player_id, team_id, gender, week_label, week_start_date, jersey_number, position),
       )
       if cur.lastrowid and cur.rowcount:
           return cur.lastrowid
       # ON CONFLICT DO NOTHING 命中時要回頭查已存在的那筆
       row = conn.execute(
           """SELECT registration_id FROM roster_registrations
              WHERE player_id = ? AND team_id = ? AND gender = ? AND week_label = ?""",
           (player_id, team_id, gender, week_label),
       ).fetchone()
       return row[0]


   def _migrate_stats(conn: sqlite3.Connection, cup_id: int) -> dict:
       """逐筆把 player_match_stats_old 重新掛到 roster_registrations，回傳統計。"""
       stat_cols = [
           "match_date", "opponent", "sets_played", "attack_total", "attack_points",
           "block_points", "serve_total", "serve_points", "receive_total",
           "receive_excellent", "dig_total", "dig_excellent", "set_total",
           "set_excellent", "total_points", "is_golden_set",
       ]
       old_rows = conn.execute(
           f"SELECT stat_id, player_id, {', '.join(stat_cols)} FROM player_match_stats_old"
       ).fetchall()

       migrated, backfilled = 0, 0
       for old_row in old_rows:
           stat_id, player_id, *values = old_row
           match_date = values[0]

           player_snapshot = conn.execute(
               "SELECT team_id, gender, jersey_number, position FROM players_old WHERE player_id = ?",
               (player_id,),
           ).fetchone()
           if player_snapshot is None:
               logger.error("stat_id=%s 找不到對應 players_old.player_id=%s，跳過", stat_id, player_id)
               continue
           team_id, gender, jersey_number, position = player_snapshot

           week_row = conn.execute(
               "SELECT round_name FROM matches WHERE match_date = ? LIMIT 1", (match_date,)
           ).fetchone()
           week_label = week_row[0] if week_row and week_row[0] else f"未比對-{match_date}"

           reg_row = conn.execute(
               """SELECT registration_id FROM roster_registrations
                  WHERE player_id = ? AND team_id = ? AND gender = ? AND week_label = ?
                    AND source = 'match_page'""",
               (player_id, team_id, gender, week_label),
           ).fetchone()

           if reg_row:
               registration_id = reg_row[0]
           else:
               start_row = conn.execute(
                   "SELECT MIN(match_date) FROM matches WHERE round_name = ?", (week_label,)
               ).fetchone()
               week_start_date = start_row[0] if start_row and start_row[0] else match_date
               registration_id = _backfill_registration(
                   conn, (player_id, team_id, gender, jersey_number, position),
                   week_label, week_start_date,
               )
               backfilled += 1

           conn.execute(
               f"""INSERT INTO player_match_stats (registration_id, {', '.join(stat_cols)})
                   VALUES ({', '.join(['?'] * (len(stat_cols) + 1))})""",
               (registration_id, *values),
           )
           migrated += 1

       conn.commit()
       return {"stats_migrated": migrated, "stats_backfilled": backfilled}


   def _verify(conn: sqlite3.Connection, expected_stat_count: int) -> int:
       """驗證：筆數不減少、無孤兒 FK。回傳孤兒數（應為 0）。"""
       actual = conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0]
       assert actual == expected_stat_count, f"筆數不符：預期 {expected_stat_count}，實際 {actual}"

       orphans = conn.execute("""
           SELECT COUNT(*) FROM player_match_stats s
           WHERE NOT EXISTS (
               SELECT 1 FROM roster_registrations r WHERE r.registration_id = s.registration_id
           )
       """).fetchone()[0]
       return orphans


   def _drop_old_tables(conn: sqlite3.Connection) -> None:
       conn.execute("DROP TABLE players_old")
       conn.execute("DROP TABLE player_match_stats_old")
       conn.commit()
       logger.info("已清除 players_old / player_match_stats_old")


   def run_migration(conn: sqlite3.Connection, cup_id: int = CUP_ID) -> dict:
       expected_stat_count = conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0]

       _rename_old_tables(conn)
       _create_new_tables(conn)

       n_players = _migrate_player_identities(conn)

       crawl_stats = crawl_all_rosters(conn, cup_id=cup_id)
       logger.info("出賽名單爬蟲結果：%s", crawl_stats)

       stats_result = _migrate_stats(conn, cup_id)

       orphans = _verify(conn, expected_stat_count)
       if orphans > 0:
           raise RuntimeError(
               f"遷移驗證失敗：發現 {orphans} 筆孤兒 player_match_stats，未清除舊表，"
               "請檢查 _migrate_stats() 邏輯後重跑（新表可安全重建，因為舊表還在）。"
           )

       _drop_old_tables(conn)

       return {
           "players_migrated": n_players,
           "registrations_created": crawl_stats["registrations_upserted"],
           **stats_result,
           "orphans_found": orphans,
       }


   def main():
       backup_database()
       conn = get_connection()
       try:
           result = run_migration(conn)
           print("\n===== Phase 2 遷移完成 =====")
           for k, v in result.items():
               print(f"{k}: {v}")
       finally:
           conn.close()


   if __name__ == "__main__":
       main()
   ```

- [ ] **Step 4:** 靜態語法檢查：
   ```bash
   python3 -c "import ast; ast.parse(open('src/etl/migrate_to_phase2.py', encoding='utf-8').read())"
   ```

- [ ] **Step 5:** 新增 `tests/test_migrate_to_phase2.py`，用**合成的小型資料庫**（不是正式 3,807 筆資料）驗證遷移邏輯本身的正確性（筆數守恆、無孤兒 FK、source 標記正確），並用 `unittest.mock.patch` 讓 `crawl_all_rosters` 不真的連網路：
   ```python
   import sqlite3
   from pathlib import Path
   from unittest.mock import patch

   from src.etl.migrate_to_phase2 import run_migration

   SCHEMA_V1 = """
   CREATE TABLE teams (
       team_id INTEGER NOT NULL, team_name TEXT NOT NULL, gender TEXT NOT NULL,
       PRIMARY KEY (team_id, gender)
   );
   CREATE TABLE players (
       player_id INTEGER PRIMARY KEY AUTOINCREMENT, team_id INTEGER NOT NULL,
       gender TEXT NOT NULL, jersey_number INTEGER, name TEXT, position TEXT,
       dob DATE, height_cm REAL, weight_kg REAL
   );
   CREATE TABLE player_match_stats (
       stat_id INTEGER PRIMARY KEY AUTOINCREMENT, player_id INTEGER NOT NULL,
       match_date DATE, opponent TEXT, sets_played INTEGER, attack_total INTEGER,
       attack_points INTEGER, block_points INTEGER, serve_total INTEGER,
       serve_points INTEGER, receive_total INTEGER, receive_excellent INTEGER,
       dig_total INTEGER, dig_excellent INTEGER, set_total INTEGER,
       set_excellent INTEGER, total_points INTEGER, is_golden_set INTEGER DEFAULT 0
   );
   CREATE TABLE matches (
       match_id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL,
       gender TEXT NOT NULL, match_date DATE NOT NULL, round_name TEXT,
       home_team TEXT NOT NULL, away_team TEXT NOT NULL
   );
   """


   def _seed_v1_db(tmp_db_path: Path) -> sqlite3.Connection:
       conn = sqlite3.connect(tmp_db_path)
       conn.execute("PRAGMA foreign_keys = OFF")  # 遷移過程要 rename/drop 表，暫時關閉
       conn.executescript(SCHEMA_V1)
       conn.execute("INSERT INTO teams VALUES (5, '新北中纖', 'F')")
       conn.execute(
           "INSERT INTO players (team_id, gender, jersey_number, name, position) "
           "VALUES (5, 'F', 2, '張瓈文', 'OP')"
       )
       conn.execute(
           "INSERT INTO matches (game_id, gender, match_date, round_name, home_team, away_team) "
           "VALUES (1, 'F', '2025-11-01', '例行賽 Week 1', '新北中纖', '義力營造')"
       )
       conn.execute(
           "INSERT INTO player_match_stats (player_id, match_date, opponent, sets_played, total_points) "
           "VALUES (1, '2025-11-01', '義力營造', 5, 20)"
       )
       conn.commit()
       return conn


   def test_migration_preserves_stat_row_count(tmp_db_path):
       conn = _seed_v1_db(tmp_db_path)

       with patch(
           "src.etl.migrate_to_phase2.crawl_all_rosters",
           return_value={"matches_scanned": 0, "matches_skipped": 0, "registrations_upserted": 0, "new_players": 0},
       ):
           result = run_migration(conn, cup_id=21)

       assert result["stats_migrated"] == 1
       assert result["orphans_found"] == 0

       final_count = conn.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0]
       assert final_count == 1
       conn.close()


   def test_migration_backfills_when_no_match_page_registration(tmp_db_path):
       conn = _seed_v1_db(tmp_db_path)

       with patch(
           "src.etl.migrate_to_phase2.crawl_all_rosters",
           return_value={"matches_scanned": 0, "matches_skipped": 0, "registrations_upserted": 0, "new_players": 0},
       ):
           result = run_migration(conn, cup_id=21)

       assert result["stats_backfilled"] == 1
       source = conn.execute("SELECT source FROM roster_registrations").fetchone()[0]
       assert source == "backfill"
       conn.close()


   def test_migration_preserves_player_id(tmp_db_path):
       conn = _seed_v1_db(tmp_db_path)

       with patch(
           "src.etl.migrate_to_phase2.crawl_all_rosters",
           return_value={"matches_scanned": 0, "matches_skipped": 0, "registrations_upserted": 0, "new_players": 0},
       ):
           run_migration(conn, cup_id=21)

       row = conn.execute("SELECT player_id, name FROM players").fetchone()
       assert row == (1, "張瓈文")
       conn.close()


   def test_migration_drops_old_tables(tmp_db_path):
       conn = _seed_v1_db(tmp_db_path)

       with patch(
           "src.etl.migrate_to_phase2.crawl_all_rosters",
           return_value={"matches_scanned": 0, "matches_skipped": 0, "registrations_upserted": 0, "new_players": 0},
       ):
           run_migration(conn, cup_id=21)

       tables = {
           row[0] for row in
           conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
       }
       assert "players_old" not in tables
       assert "player_match_stats_old" not in tables
       conn.close()
   ```

- [ ] **Step 6:** 跑測試：
   ```bash
   python -m pytest tests/test_migrate_to_phase2.py -v
   ```
   **預期輸出**：4 個測試 PASSED。

   **未實測**：需執行者實跑。

- [ ] **Step 7:** **[實際對正式 DB 執行遷移，需要網路連線外部統計系統]**。先確認網路可連：
   ```bash
   curl -sI --max-time 10 http://114.35.229.141 | head -1
   ```
   **預期輸出**：`HTTP/1.1 200 OK` 或類似成功回應。若失敗，**不要**跳過爬蟲直接用全 backfill 遷移——先解決網路問題，因為那樣會讓所有歷史週次都變成 `source='backfill'`，喪失本 phase 想要的「真實逐週登錄名單」價值。

- [ ] **Step 8:** 對正式 DB 執行遷移（`main()` 內已內建再備份一次）：
   ```bash
   python -m src.etl.migrate_to_phase2
   ```
   **預期輸出**：印出 `===== Phase 2 遷移完成 =====` 與統計數字，其中 `orphans_found: 0`、`stats_migrated` 應等於步驟 2 記錄的基準數字（未實測時的參考值 3807，執行時以當下實際基準為準）。

   **未實測**：本規劃 session 明確不執行任何會寫入正式 DB 的操作（邊界限制），此步驟必須由執行者實跑。若失敗，`data/db/tvl_database.db.bak-<timestamp>` 備份檔可直接複製回 `data/db/tvl_database.db` 復原。

- [ ] **Step 9:** 遷移後驗證（用真實查詢再次確認）：
   ```bash
   python3 -c "
   import sqlite3
   conn = sqlite3.connect('data/db/tvl_database.db')
   print('roster_registrations 總筆數:', conn.execute('SELECT COUNT(*) FROM roster_registrations').fetchone()[0])
   print('  match_page:', conn.execute(\"SELECT COUNT(*) FROM roster_registrations WHERE source='match_page'\").fetchone()[0])
   print('  backfill:', conn.execute(\"SELECT COUNT(*) FROM roster_registrations WHERE source='backfill'\").fetchone()[0])
   print('player_match_stats 總筆數:', conn.execute('SELECT COUNT(*) FROM player_match_stats').fetchone()[0])
   orphans = conn.execute('''
       SELECT COUNT(*) FROM player_match_stats s
       WHERE NOT EXISTS (SELECT 1 FROM roster_registrations r WHERE r.registration_id = s.registration_id)
   ''').fetchone()[0]
   print('孤兒 FK 數:', orphans)
   "
   ```
   **預期輸出**：`player_match_stats 總筆數` 與遷移前一致、`孤兒 FK 數: 0`。`backfill` 筆數應遠小於 `match_page`（若 `backfill` 佔比異常高，代表 `crawl_all_rosters` 沒有真的抓到資料，需回頭檢查 Task 3 的爬蟲是否正常運作，不要就此接受）。

- [ ] **Step 10:** 確認行尾乾淨：`git diff --stat -w`（本 task 只 commit 程式碼與測試，`data/db/tvl_database.db` 的變更也要 commit，因為 CLAUDE.md 規定 `.db` 要進 git；確認 `git status` 有看到 DB 檔案變更）。

- [ ] **Step 11:** **[STOP：等待使用者同意 commit]**：
    ```
    feat: Phase 2 遷移腳本 — players 拆表 + roster_registrations 回填

    新增 migrate_to_phase2.py，把既有 3,807 筆 player_match_stats
    重新掛到 roster_registrations；真實出賽名單來源標記
    source='match_page'，查無資料的少數週次標記 'backfill' 並用
    當時快照回推（不插補假資料）。已備份於
    data/db/tvl_database.db.bak-<timestamp>（未進 git，本機保留）。
    ```
    > 提醒：commit message 中的 `<timestamp>` 要換成實際檔名；備份檔本身不進 git（`.gitignore` 目前排除 `data/db/*` 但白名單放行 `*.db`，`.bak-*` 檔名不吃白名單，天然就不會被 commit，執行者可用 `git status` 確認）。

---

## Task 5：`db_loader.py` 改為身分層 only

**Files:**
- Modify: `src/etl/db_loader.py`（`upsert_players` 改為身分層邏輯，移除 `team_id/jersey_number/position` 寫入）
- Modify: `src/etl/crawler.py`（docstring 與 `main()` 輸出說明加註：`team_id/jersey_number/position` 僅供參考，不再入庫）
- Modify: `tests/test_db_loader_idempotent.py`（Phase 1 建立的測試，改用新 schema 假資料）

**Interfaces:**
- Produces：
  ```python
  def upsert_player_identity(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
      """取代 Phase 1 的 upsert_players()，用 (name, gender) 自然鍵 upsert 身分層欄位。"""
  ```
- Consumes：`src.etl.cleaner.load_raw/clean/quality_report`（不變，CSV 仍然含 `team_id/jersey_number/position` 欄位，只是 `db_loader` 現在只挑身分欄位寫入 `players`，其餘欄位不寫進資料庫——`teams` 表照樣要維護，因為 `roster_registrations` 的 FK 需要 `teams` 存在）。

**設計決策（鎖定，不要重新發明）**：Phase 2 之後，`roster_registrations` 是團隊/背號/位置的**唯一**權威來源，一律由 `stats_crawler.crawl_all_rosters()`（Task 3）從 `Match.ashx` 寫入。`db_loader`／`crawler.py` 這條「官網名單」管線**不再**寫入任何團隊/背號/位置資訊到資料庫，只用來補齊 `roster_registrations` 拿不到的身分屬性（生日、身高、體重）。這是為了避免兩條管線各自維護一份「目前隊伍」定義而互相打架。

**步驟：**

- [ ] **Step 1:** 編輯 `src/etl/db_loader.py`，把 `upsert_players()`（Phase 1 版本）改名為 `upsert_player_identity()`，自然鍵從 `(team_id, gender, jersey_number, name)` 改為 `(name, gender)`，且只更新身分欄位：
   ```python
   def _find_existing_player_id(conn: sqlite3.Connection, gender: str, name: str) -> int | None:
       """用自然鍵 (name, gender) 找既有 player_id，找不到回傳 None。"""
       row = conn.execute(
           "SELECT player_id FROM players WHERE name = ? AND gender = ?",
           (name, gender),
       ).fetchone()
       return row[0] if row else None


   def upsert_player_identity(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
       """
       用自然鍵 (name, gender) upsert players 表的身分欄位（dob/height_cm/weight_kg）。
       team_id/jersey_number/position 由 roster_registrations 維護，本函式不寫入。
       """
       identity_cols = ["gender", "name", "dob", "height_cm", "weight_kg"]
       identities = df[identity_cols].drop_duplicates(subset=["name", "gender"])

       n_inserted = 0
       n_updated = 0
       for row in identities.itertuples(index=False):
           existing_id = _find_existing_player_id(conn, row.gender, row.name)
           if existing_id is None:
               conn.execute(
                   "INSERT INTO players (name, gender, dob, height_cm, weight_kg) VALUES (?, ?, ?, ?, ?)",
                   (row.name, row.gender, row.dob, row.height_cm, row.weight_kg),
               )
               n_inserted += 1
           else:
               conn.execute(
                   "UPDATE players SET dob = ?, height_cm = ?, weight_kg = ? WHERE player_id = ?",
                   (row.dob, row.height_cm, row.weight_kg, existing_id),
               )
               n_updated += 1

       conn.commit()
       logger.info("players 身分層 upsert 完成：新增 %d 筆、更新 %d 筆", n_inserted, n_updated)
   ```
   刪除 Phase 1 版本的 `upsert_players()` 與 `_find_existing_player_id(conn, team_id, gender, jersey_number, name)`（被上面的新版取代）。

- [ ] **Step 2:** 更新 `verify()` 函式（原第 85-98 行），因為 `players` 表已無 `position`/`height_cm` 以外的可 JOIN 欄位改變 —— 實際上 `position`/`height_cm` 都還在（`height_cm` 一直都在身分層），只有 `position` 這個查詢欄位消失了（`position` 現在只存在 `roster_registrations`）。改寫驗證查詢為改查「身高前 10 高的球員」（不再依賴 `position`，因為那已不是 `players` 的欄位）：
   ```python
   def verify(conn: sqlite3.Connection) -> pd.DataFrame:
       """
       驗證查詢：女子組身高最高的 10 位球員（身分層驗證，不再依賴 position，
       position 已搬到 roster_registrations，見 Phase 2 遷移）。
       """
       query = """
           SELECT name, gender, height_cm
           FROM players
           WHERE gender = 'F' AND height_cm IS NOT NULL
           ORDER BY height_cm DESC
       """
       return pd.read_sql_query(query, conn)
   ```

- [ ] **Step 3:** 更新 `main()` 呼叫點，把 `upsert_players(conn, df)` 改成 `upsert_player_identity(conn, df)`；`upsert_teams(conn, df)` 維持不變（`teams` 表結構沒變）：
   ```python
   def main():
       conn = get_connection()

       try:
           init_db(conn)
           df = load_csv()
           upsert_teams(conn, df)
           upsert_player_identity(conn, df)

           result = verify(conn)
           print("\n===== 驗證查詢：女子組身高前 10 高的球員 =====")
           print(result.head(10).to_string(index=False))
       finally:
           conn.close()

       logger.info("資料庫載入完成：%s", DB_PATH)
   ```

- [ ] **Step 4:** 編輯 `src/etl/crawler.py`，明示這條「官網名單」管線在 Phase 2 之後只維護球員身分層——CSV 仍會輸出 `team_id`/`jersey_number`/`position`，但這些欄位僅供人工參考，`db_loader.upsert_player_identity()` 不會把它們寫入資料庫（權威來源是 `roster_registrations`，見本 Task 開頭的設計決策）。先改模組 docstring（檔案開頭），原文：
   ```python
   """
   TVL 球員名單爬蟲模組
   從企業排球聯賽官網抓取球隊球員資料並匯出為 CSV。
   """
   ```
   改為：
   ```python
   """
   TVL 球員名單爬蟲模組
   從企業排球聯賽官網抓取球隊球員資料並匯出為 CSV。

   Phase 2 起：本模組只負責維護球員「身分層」（name/gender/dob/height_cm/weight_kg）。
   輸出的 CSV 仍含 team_id/jersey_number/position 欄位，但這些欄位在 Phase 2 之後
   僅供人工參考——db_loader.upsert_player_identity() 不會把它們寫入資料庫。
   team_id/jersey_number/position 的唯一權威來源是 roster_registrations
   （見 src/etl/stats_crawler.py 的 crawl_all_rosters()）。
   """
   ```
   再編輯 `main()` 內儲存 CSV 之後的區塊（原文）：
   ```python
       df.to_csv(output_path, index=False, encoding="utf-8-sig")
       logger.info("已儲存至 %s", output_path)
   ```
   改為：
   ```python
       df.to_csv(output_path, index=False, encoding="utf-8-sig")
       logger.info("已儲存至 %s", output_path)
       logger.info(
           "提醒：team_id/jersey_number/position 僅供參考，"
           "Phase 2 之後由 roster_registrations 權威維護，"
           "db_loader 不會把這些欄位寫入資料庫。"
       )
   ```
   驗證：
   ```bash
   python3 -c "import ast; ast.parse(open('src/etl/crawler.py', encoding='utf-8').read())"
   grep -n "僅供人工參考\|roster_registrations 權威維護" src/etl/crawler.py
   ```
   **預期輸出**：`ast.parse` 無錯誤；`grep` 至少匹配 2 行（docstring 與 `main()` 各一處）。

- [ ] **Step 5:** 靜態語法檢查：
   ```bash
   python3 -c "import ast; ast.parse(open('src/etl/db_loader.py', encoding='utf-8').read())"
   ```

- [ ] **Step 6:** 更新 `tests/test_db_loader_idempotent.py`（Phase 1 建立），改用新 schema 並測試新函式：
   ```python
   import sqlite3

   import pandas as pd
   import pytest

   from src.etl.db_loader import init_db, upsert_teams, upsert_player_identity


   @pytest.fixture
   def conn(tmp_db_path):
       connection = sqlite3.connect(tmp_db_path)
       connection.execute("PRAGMA foreign_keys = ON")
       yield connection
       connection.close()


   SAMPLE_DF = pd.DataFrame([
       {"team_id": 1, "team_name": "測試隊", "gender": "M",
        "jersey_number": 10, "name": "測試球員", "position": "OH",
        "dob": "2000-01-01", "height_cm": 190.0, "weight_kg": 80.0},
   ])


   def test_init_db_is_idempotent(conn):
       init_db(conn)
       init_db(conn)
       tables = {
           row[0] for row in
           conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
       }
       assert {"teams", "players", "roster_registrations", "player_match_stats", "matches"} <= tables


   def test_upsert_player_identity_does_not_write_team_or_jersey(conn):
       init_db(conn)
       upsert_teams(conn, SAMPLE_DF)
       upsert_player_identity(conn, SAMPLE_DF)

       cols = {row[1] for row in conn.execute("PRAGMA table_info(players)")}
       assert "team_id" not in cols
       assert "jersey_number" not in cols
       assert "position" not in cols

       row = conn.execute(
           "SELECT name, gender, dob, height_cm, weight_kg FROM players WHERE name = '測試球員'"
       ).fetchone()
       assert row == ("測試球員", "M", "2000-01-01", 190.0, 80.0)


   def test_upsert_player_identity_preserves_player_id_on_rerun(conn):
       init_db(conn)
       upsert_teams(conn, SAMPLE_DF)
       upsert_player_identity(conn, SAMPLE_DF)
       first_id = conn.execute(
           "SELECT player_id FROM players WHERE name = '測試球員'"
       ).fetchone()[0]

       upsert_player_identity(conn, SAMPLE_DF)

       rows = conn.execute(
           "SELECT player_id FROM players WHERE name = '測試球員'"
       ).fetchall()
       assert len(rows) == 1
       assert rows[0][0] == first_id


   def test_init_db_does_not_wipe_existing_registrations(conn):
       init_db(conn)
       upsert_teams(conn, SAMPLE_DF)
       upsert_player_identity(conn, SAMPLE_DF)
       player_id = conn.execute(
           "SELECT player_id FROM players WHERE name = '測試球員'"
       ).fetchone()[0]
       conn.execute(
           """INSERT INTO roster_registrations
              (player_id, team_id, gender, week_label, jersey_number, position, source)
              VALUES (?, 1, 'M', '例行賽 Week 1', 10, 'OH', 'match_page')""",
           (player_id,),
       )
       conn.commit()

       init_db(conn)
       upsert_teams(conn, SAMPLE_DF)
       upsert_player_identity(conn, SAMPLE_DF)

       remaining = conn.execute("SELECT COUNT(*) FROM roster_registrations").fetchone()[0]
       assert remaining == 1
   ```

- [ ] **Step 7:** 跑測試：
   ```bash
   python -m pytest tests/test_db_loader_idempotent.py -v
   ```
   **預期輸出**：4 個測試 PASSED。

- [ ] **Step 8:** 跑全部測試，確認沒有破壞前面任何 task：
   ```bash
   python -m pytest tests/ -v
   ```

- [ ] **Step 9:** 確認行尾乾淨：`git diff --stat -w`

- [ ] **Step 10:** **[STOP：等待使用者同意 commit]**：
   ```
   refactor: db_loader 改為只維護球員身分層，團隊/背號/位置交給 roster_registrations

   upsert_players 改名 upsert_player_identity，自然鍵從
   (team_id, gender, jersey_number, name) 收斂為 (name, gender)；
   不再寫入 team_id/jersey_number/position，避免與
   roster_registrations 出現兩份互相打架的「目前隊伍」定義。
   crawler.py 加註 docstring 與執行提示，明示 CSV 的
   team_id/jersey_number/position 僅供參考、不再入庫。
   ```

---

## Task 6：`helpers.py` — 新增 `get_current_roster()`、改寫 `get_league_aggregated_stats()`

**Files:**
- Modify: `src/app/helpers.py`

**Interfaces:**
- Produces：
  ```python
  def get_current_roster(team_id: int, gender_code: str) -> pd.DataFrame:
      """
      回傳指定隊伍「最新一週」的登錄名單：columns = player_id, jersey_number, name, position。
      「最新一週」定義為該隊 roster_registrations.week_start_date 的最大值。
      """

  def get_league_aggregated_stats(gender_code: str) -> pd.DataFrame:
      """
      （既有函式簽名不變）撈取該組別所有球員的聚合統計，position/team_name 取自
      「該球員 week_start_date 最大的那筆登錄」（每位球員一個代表性位置/球隊，
      避免球員換位置/換隊時 GROUP BY 出現同一 player_id 多列）。
      """
  ```
- Consumes：`src.utils.db_config`（間接，透過既有 `load_data`）。

**設計說明（三種不同粒度的 JOIN，刻意不同，不要統一成一種）：**
- **單場（box_score.py）**：直接用該筆統計自己的 `registration_id` 取 position/jersey/team——那就是「那一場當下」的正確登錄，不需要任何「最新」邏輯（Task 9 處理）。
- **球員彙總（`get_league_aggregated_stats`、`player_deep.py` 側欄標籤）**：需要「每位球員一個代表性 position/team」，用該球員 `week_start_date` 最大的登錄。
- **球隊目前名單（main.py 側欄下拉、`get_current_roster`）**：需要「這支隊伍現在有哪些人」，用該隊 `week_start_date` 最大的那一批登錄（不是單一球員的最新，是整隊同一週次的名單）。

**步驟：**

- [ ] **Step 1:** 編輯 `src/app/helpers.py`，在 `enrich_box_score()` 之前新增 `get_current_roster()`：
   ```python
   @st.cache_data
   def get_current_roster(team_id: int, gender_code: str) -> pd.DataFrame:
       """回傳指定隊伍最新一週（week_start_date 最大）的登錄名單。"""
       return load_data(
           """
           SELECT r.player_id, r.jersey_number, p.name, r.position
           FROM roster_registrations r
           JOIN players p ON r.player_id = p.player_id
           WHERE r.team_id = ? AND r.gender = ?
             AND r.week_start_date = (
                 SELECT MAX(week_start_date) FROM roster_registrations
                 WHERE team_id = r.team_id AND gender = r.gender
             )
           ORDER BY r.jersey_number
           """,
           (team_id, gender_code),
       )
   ```

- [ ] **Step 2:** 改寫 `get_league_aggregated_stats()`（原第 231-287 行），把 `JOIN players p ON s.player_id = p.player_id` 那條路徑改成透過 `roster_registrations`，並用「每位球員最新一筆登錄」取得代表性 position/team_name：
   ```python
   @st.cache_data
   def get_league_aggregated_stats(gender_code: str) -> pd.DataFrame:
       """
       撈取該組別所有球員的聚合統計數據。position/team_name 取自該球員
       week_start_date 最大的那筆登錄（每位球員一個代表性值，避免換隊/換位置
       造成 GROUP BY 產生重複列）。僅保留總局數 >= 5 的球員，排除極端值。
       """
       raw = load_data(
           """
           SELECT p.player_id,
                  p.name,
                  latest.position AS position,
                  t.team_name,
                  SUM(s.sets_played)       AS total_sets,
                  SUM(s.attack_points)     AS atk_pts,
                  SUM(s.attack_total)      AS atk_tot,
                  SUM(s.block_points)      AS blk_pts,
                  SUM(s.serve_points)      AS srv_pts,
                  SUM(s.serve_total)       AS srv_tot,
                  SUM(s.receive_excellent) AS rcv_exc,
                  SUM(s.receive_total)     AS rcv_tot,
                  SUM(s.dig_excellent)     AS dig_exc,
                  SUM(s.dig_total)         AS dig_tot,
                  SUM(s.set_excellent)     AS set_exc,
                  SUM(s.set_total)         AS set_tot,
                  SUM(s.total_points)      AS total_points,
                  COUNT(*)                 AS n_games
           FROM player_match_stats s
           JOIN roster_registrations r ON s.registration_id = r.registration_id
           JOIN players p ON r.player_id = p.player_id
           JOIN (
               SELECT rr.player_id, rr.position, rr.team_id, rr.gender
               FROM roster_registrations rr
               WHERE rr.week_start_date = (
                   SELECT MAX(week_start_date) FROM roster_registrations rr2
                   WHERE rr2.player_id = rr.player_id
               )
           ) latest ON latest.player_id = p.player_id
           JOIN teams t ON t.team_id = latest.team_id AND t.gender = latest.gender
           WHERE latest.gender = ?
           GROUP BY p.player_id
           HAVING SUM(s.sets_played) >= 5
           """,
           (gender_code,),
       )
       # 計算進階比率指標（向量化）—— 以下邏輯完全不變
       raw["asr"] = vec_pct(raw["atk_pts"], raw["atk_tot"])
       raw["gp_pct"] = vec_pct(raw["rcv_exc"], raw["rcv_tot"])
       raw["ace_pct"] = vec_pct(raw["srv_pts"], raw["srv_tot"])
       raw["dig_pct"] = vec_pct(raw["dig_exc"], raw["dig_tot"])
       raw["ppg"] = np.where(raw["n_games"] > 0, raw["total_points"] / raw["n_games"], 0.0)
       raw["set_pct"] = vec_pct(raw["set_exc"], raw["set_tot"])
       raw["blk_per_set"] = np.where(raw["total_sets"] > 0, raw["blk_pts"] / raw["total_sets"], 0.0)
       raw["def_load"] = raw["rcv_tot"] + raw["dig_tot"]
       raw["def_pct"] = vec_pct(raw["rcv_exc"] + raw["dig_exc"], raw["rcv_tot"] + raw["dig_tot"])

       pr_cols = ["asr", "gp_pct", "ace_pct", "dig_pct", "set_pct", "blk_per_set", "def_pct"]
       for col in pr_cols:
           raw[f"{col}_pr"] = (
               raw.groupby("position")[col]
               .rank(pct=True)
               .mul(100)
               .round(1)
           )

       return raw
   ```
   （`enrich_box_score()` 不用改，它只操作已經撈出來的 DataFrame，跟 schema 無關。）

- [ ] **Step 3:** 靜態語法檢查：
   ```bash
   python3 -c "import ast; ast.parse(open('src/app/helpers.py', encoding='utf-8').read())"
   ```

- [ ] **Step 4:** 建立 `tests/test_helpers_phase2_queries.py`，用合成資料驗證新 SQL 邏輯（繞過 `st.cache_data` 裝飾器直接測 SQL 字串的正確性，用 `sqlite3` 直接跑同一段 query 字串，不透過 Streamlit runtime）：
   ```python
   import sqlite3
   from pathlib import Path

   SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"

   GET_CURRENT_ROSTER_SQL = """
       SELECT r.player_id, r.jersey_number, p.name, r.position
       FROM roster_registrations r
       JOIN players p ON r.player_id = p.player_id
       WHERE r.team_id = ? AND r.gender = ?
         AND r.week_start_date = (
             SELECT MAX(week_start_date) FROM roster_registrations
             WHERE team_id = r.team_id AND gender = r.gender
         )
       ORDER BY r.jersey_number
   """


   def _seed(conn):
       conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
       conn.execute("INSERT INTO teams VALUES (5, '新北中纖', 'F')")
       conn.execute("INSERT INTO players (name, gender) VALUES ('球員A', 'F')")
       conn.execute("INSERT INTO players (name, gender) VALUES ('球員B', 'F')")
       pid_a = conn.execute("SELECT player_id FROM players WHERE name='球員A'").fetchone()[0]
       pid_b = conn.execute("SELECT player_id FROM players WHERE name='球員B'").fetchone()[0]
       # 球員A：第1週背號2，第2週背號5（換背號）
       conn.execute(
           "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, week_start_date, jersey_number, position, source) "
           "VALUES (?, 5, 'F', '例行賽 Week 1', '2025-11-01', 2, 'OP', 'match_page')", (pid_a,),
       )
       conn.execute(
           "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, week_start_date, jersey_number, position, source) "
           "VALUES (?, 5, 'F', '例行賽 Week 2', '2025-11-08', 5, 'OP', 'match_page')", (pid_a,),
       )
       # 球員B：只在第1週出現過
       conn.execute(
           "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, week_start_date, jersey_number, position, source) "
           "VALUES (?, 5, 'F', '例行賽 Week 1', '2025-11-01', 9, 'MB', 'match_page')", (pid_b,),
       )
       conn.commit()
       return pid_a, pid_b


   def test_get_current_roster_uses_latest_week_per_team(tmp_db_path):
       conn = sqlite3.connect(tmp_db_path)
       pid_a, pid_b = _seed(conn)

       rows = conn.execute(GET_CURRENT_ROSTER_SQL, (5, "F")).fetchall()

       # 只有球員A有第2週的紀錄（week_start_date 最大），球員B停留在第1週不應出現
       assert rows == [(pid_a, 5, "球員A", "OP")]
       conn.close()
   ```

- [ ] **Step 5:** 跑測試：
   ```bash
   python -m pytest tests/test_helpers_phase2_queries.py -v
   ```
   **預期輸出**：1 個測試 PASSED。

   **未實測**：需執行者實跑。

- [ ] **Step 6:** 確認行尾乾淨：`git diff --stat -w`

- [ ] **Step 7:** **[STOP：等待使用者同意 commit]**：
   ```
   feat: helpers.py 新增 get_current_roster，get_league_aggregated_stats 改走 roster_registrations

   球隊目前名單（給側欄下拉）用該隊最新一週登錄；球員彙總統計的
   position/team_name 用該球員自己最新一筆登錄，避免換位置/換隊
   造成 GROUP BY 重複列。
   ```

---

## Task 7：`main.py` 側欄改用 `get_current_roster()`

**Files:**
- Modify: `src/app/main.py`（第 68 行 import、第 96-100 行球員下拉查詢）

**Interfaces:**
- Consumes：`src.app.helpers.get_current_roster(team_id, gender_code) -> pd.DataFrame`（Task 6 新增）。
- Produces：`ctx` dict 介面不變（仍是 `player_id, player_name, player_position, gender_code, gender, team_name, team_id` 七個 key），下游 tab 完全不用因為這個 task 而改動介面。

**步驟：**

- [ ] **Step 1:** 編輯 `src/app/main.py` 第 68 行，把：
   ```python
   from src.app.helpers import load_data, inject_mobile_css
   ```
   改為：
   ```python
   from src.app.helpers import load_data, inject_mobile_css, get_current_roster
   ```

- [ ] **Step 2:** 編輯第 96-100 行，原文：
   ```python
   players_df = load_data(
       "SELECT player_id, jersey_number, name, position FROM players "
       "WHERE team_id = ? AND gender = ? ORDER BY jersey_number",
       (team_id, gender_code),
   )
   ```
   改為：
   ```python
   players_df = get_current_roster(team_id, gender_code)
   ```

- [ ] **Step 3:** `_player_label()`（第 105-110 行）與後續 `player_id`/`player_name`/`player_position` 取值邏輯完全不用改，因為 `get_current_roster()` 回傳的欄位名（`player_id, jersey_number, name, position`）與原查詢一致。

- [ ] **Step 4:** 靜態語法檢查：
   ```bash
   python3 -c "import ast; ast.parse(open('src/app/main.py', encoding='utf-8').read())"
   ```

- [ ] **Step 5:** 更新 `tests/test_main_tabs.py`（Phase 1 建立）新增一個檢查，確保 `main.py` 不再對 `players` 表直接查詢 `team_id`（那個欄位已經不存在於新 schema）：
   ```python
   def test_sidebar_uses_get_current_roster_not_raw_players_query():
       source = MAIN_PY.read_text(encoding="utf-8")
       assert "get_current_roster" in source
       assert "FROM players \"" not in source  # 舊的直接查 players.team_id 寫法已移除
   ```

- [ ] **Step 6:** 跑測試：
   ```bash
   python -m pytest tests/test_main_tabs.py -v
   ```
   **預期輸出**：4 個測試 PASSED（Phase 1 的 3 個 + 本 task 新增的 1 個）。

- [ ] **Step 7:** 確認行尾乾淨：`git diff --stat -w`

- [ ] **Step 8:** **[STOP：等待使用者同意 commit]**：
   ```
   refactor: main.py 側欄球員下拉改用 get_current_roster

   players 表已無 team_id/jersey_number/position，側欄選單改用
   roster_registrations 驅動的 get_current_roster()，介面（ctx dict）
   對下游 tab 完全不變。
   ```

---

## Task 8：`player_deep.py` 與 `match_trend.py` 改查詢

**Files:**
- Modify: `src/app/tabs/player_deep.py`（`_load_league_agg` 第 29-42 行、主查詢第 68-71 行、`pos_filter` 第 162-164 行）
- Modify: `src/app/tabs/match_trend.py`（第 21-24 行）

**Interfaces:**
- Consumes：既有 `ctx["player_id"]`、`ctx["player_position"]`、`ctx["gender_code"]`（不變）。
- Produces：兩個 tab 對外行為（畫面呈現）完全不變，只改變 SQL 的 JOIN 路徑。

**步驟：**

- [ ] **Step 1:** 編輯 `src/app/tabs/player_deep.py`，把 `_load_league_agg()`（原第 29-42 行）的 JOIN 從 `players` 改成 `roster_registrations`：
   ```python
   def _load_league_agg(gender_code: str, pos_filter: str = "", params: tuple = ()):
       """撈取聯盟聚合數據（全組別或特定位置）。position 過濾直接用該筆統計對應的
       registration.position（該球員該場當下的位置），不是球員層級的代表值——
       這是「聯盟整體同位置平均」，用逐筆真實登錄位置加總最符合語意。"""
       return load_data(
           f"""
           SELECT SUM(s.attack_points) AS atk_pts, SUM(s.attack_total) AS atk_tot,
                  SUM(s.receive_excellent) AS rcv_exc, SUM(s.receive_total) AS rcv_tot,
                  SUM(s.serve_points) AS srv_pts, SUM(s.serve_total) AS srv_tot,
                  SUM(s.dig_excellent) AS dig_exc, SUM(s.dig_total) AS dig_tot,
                  SUM(s.set_excellent) AS set_exc, SUM(s.set_total) AS set_tot,
                  SUM(s.block_points) AS blk_pts, SUM(s.sets_played) AS tot_sets,
                  SUM(s.total_points) AS tot_pts, COUNT(*) AS n_games
           FROM player_match_stats s
           JOIN roster_registrations r ON s.registration_id = r.registration_id
           WHERE r.gender = ? {pos_filter}
           """,
           params,
       ).iloc[0]
   ```

- [ ] **Step 2:** 把 `pos_filter`/`pos_params`（原第 162-164 行）的欄位前綴從 `p.position` 改成 `r.position`：
   ```python
   pos_filter = "AND r.position = ?" if player_position else ""
   pos_params = (gender_code, player_position) if player_position else (gender_code,)
   lg = _parse_agg(_load_league_agg(gender_code, pos_filter, pos_params))
   ```

- [ ] **Step 3:** 把主查詢（原第 68-71 行）：
   ```python
   stats_df = load_data(
       "SELECT * FROM player_match_stats WHERE player_id = ? ORDER BY match_date",
       (player_id,),
   )
   ```
   改為：
   ```python
   stats_df = load_data(
       """
       SELECT s.* FROM player_match_stats s
       JOIN roster_registrations r ON s.registration_id = r.registration_id
       WHERE r.player_id = ?
       ORDER BY s.match_date
       """,
       (player_id,),
   )
   ```
   （`s.*` 確保回傳欄位仍是 `player_match_stats` 的所有欄位，下游 `stats_df["attack_points"]` 等既有存取方式完全不用改。）

- [ ] **Step 4:** 編輯 `src/app/tabs/match_trend.py`，把第 21-24 行同樣的查詢模式做一樣的改動：
   ```python
   match_df = load_data(
       """
       SELECT s.* FROM player_match_stats s
       JOIN roster_registrations r ON s.registration_id = r.registration_id
       WHERE r.player_id = ?
       ORDER BY s.match_date
       """,
       (player_id,),
   )
   ```

- [ ] **Step 5:** 靜態語法檢查：
   ```bash
   python3 -c "import ast; ast.parse(open('src/app/tabs/player_deep.py', encoding='utf-8').read())"
   python3 -c "import ast; ast.parse(open('src/app/tabs/match_trend.py', encoding='utf-8').read())"
   ```

- [ ] **Step 6:** 建立 `tests/test_tab_queries_phase2.py`，用合成資料驗證這兩條查詢語意正確（球員換隊/多週資料都能撈到，且欄位集合與原本一致）：
   ```python
   import sqlite3
   from pathlib import Path

   SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"

   PLAYER_STATS_SQL = """
       SELECT s.* FROM player_match_stats s
       JOIN roster_registrations r ON s.registration_id = r.registration_id
       WHERE r.player_id = ?
       ORDER BY s.match_date
   """


   def _seed(conn):
       conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
       conn.execute("INSERT INTO teams VALUES (5, '新北中纖', 'F')")
       conn.execute("INSERT INTO players (name, gender) VALUES ('球員A', 'F')")
       pid = conn.execute("SELECT player_id FROM players").fetchone()[0]
       conn.execute(
           "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, jersey_number, position, source) "
           "VALUES (?, 5, 'F', '例行賽 Week 1', 2, 'OP', 'match_page')", (pid,),
       )
       rid1 = conn.execute("SELECT registration_id FROM roster_registrations").fetchone()[0]
       conn.execute(
           "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, jersey_number, position, source) "
           "VALUES (?, 5, 'F', '例行賽 Week 2', 5, 'OP', 'match_page')", (pid,),
       )
       rid2 = conn.execute(
           "SELECT registration_id FROM roster_registrations WHERE week_label = '例行賽 Week 2'"
       ).fetchone()[0]
       conn.execute(
           "INSERT INTO player_match_stats (registration_id, match_date, total_points) VALUES (?, '2025-11-01', 10)", (rid1,),
       )
       conn.execute(
           "INSERT INTO player_match_stats (registration_id, match_date, total_points) VALUES (?, '2025-11-08', 15)", (rid2,),
       )
       conn.commit()
       return pid


   def test_player_stats_query_returns_all_weeks_across_registrations(tmp_db_path):
       conn = sqlite3.connect(tmp_db_path)
       pid = _seed(conn)

       rows = conn.execute(PLAYER_STATS_SQL, (pid,)).fetchall()

       assert len(rows) == 2, "應該撈到該球員橫跨兩週不同 registration 的全部統計"
       conn.close()
   ```

- [ ] **Step 7:** 跑測試：
   ```bash
   python -m pytest tests/test_tab_queries_phase2.py -v
   ```
   **預期輸出**：1 個測試 PASSED。

- [ ] **Step 8:** 確認行尾乾淨：`git diff --stat -w`

- [ ] **Step 9:** **[STOP：等待使用者同意 commit]**：
   ```
   refactor: player_deep/match_trend 改透過 roster_registrations 查球員統計

   player_id 直查 player_match_stats 已不可行（FK 改指向
   registration_id），統一改成 JOIN roster_registrations 取得。
   ```

---

## Task 9：`box_score.py` 改查詢

**Files:**
- Modify: `src/app/tabs/box_score.py`（第 157-170 行、第 182-195 行兩段幾乎相同的查詢）

**Interfaces:**
- Consumes：既有的 `bs_team_id`、`bs_gender_code`、`sel_date`、`sel_opponent`、`opp_team_id`、`opp_gender`（皆為函式內既有區域變數，不變）。
- Produces：`team_a_df`/`team_b_df` 兩個 DataFrame，欄位集合與原本完全一致（`name, position, sets_played, attack_points, attack_total, ...`），因此 `_format_box_score()`/`enrich_box_score()`/`_style_box_score()` 完全不用改。

**設計說明**：這裡不需要「最新一週」邏輯——`registration_id` 本身就是「那一場比賽當下」的正確登錄，直接照 `match_date`/`opponent`/`team_id` 過濾即可，是三種查詢模式中最簡單的一種。

**步驟：**

- [ ] **Step 1:** 編輯 `src/app/tabs/box_score.py`，把 Team A 查詢（原第 156-170 行）：
   ```python
   team_a_df = load_data(
       """
       SELECT p.name, p.position, s.sets_played,
              s.attack_points, s.attack_total,
              s.block_points,
              s.serve_points, s.serve_total,
              s.receive_excellent, s.receive_total,
              s.dig_excellent, s.dig_total,
              s.set_excellent, s.set_total,
              s.total_points
       FROM player_match_stats s
       JOIN players p ON s.player_id = p.player_id
       WHERE p.team_id = ? AND p.gender = ?
         AND s.match_date = ? AND s.opponent = ?
       ORDER BY s.total_points DESC
       """,
       (bs_team_id, bs_gender_code, sel_date, sel_opponent),
   )
   ```
   改為：
   ```python
   team_a_df = load_data(
       """
       SELECT p.name, r.position, s.sets_played,
              s.attack_points, s.attack_total,
              s.block_points,
              s.serve_points, s.serve_total,
              s.receive_excellent, s.receive_total,
              s.dig_excellent, s.dig_total,
              s.set_excellent, s.set_total,
              s.total_points
       FROM player_match_stats s
       JOIN roster_registrations r ON s.registration_id = r.registration_id
       JOIN players p ON r.player_id = p.player_id
       WHERE r.team_id = ? AND r.gender = ?
         AND s.match_date = ? AND s.opponent = ?
       ORDER BY s.total_points DESC
       """,
       (bs_team_id, bs_gender_code, sel_date, sel_opponent),
   )
   ```

- [ ] **Step 2:** 對 Team B 查詢（原第 181-195 行）做完全相同的改法：
   ```python
   team_b_df = load_data(
       """
       SELECT p.name, r.position, s.sets_played,
              s.attack_points, s.attack_total,
              s.block_points,
              s.serve_points, s.serve_total,
              s.receive_excellent, s.receive_total,
              s.dig_excellent, s.dig_total,
              s.set_excellent, s.set_total,
              s.total_points
       FROM player_match_stats s
       JOIN roster_registrations r ON s.registration_id = r.registration_id
       JOIN players p ON r.player_id = p.player_id
       WHERE r.team_id = ? AND r.gender = ?
         AND s.match_date = ?
       ORDER BY s.total_points DESC
       """,
       (opp_team_id, opp_gender, sel_date),
   )
   ```

- [ ] **Step 3:** 靜態語法檢查：
   ```bash
   python3 -c "import ast; ast.parse(open('src/app/tabs/box_score.py', encoding='utf-8').read())"
   ```

- [ ] **Step 4:** 在 `tests/test_tab_queries_phase2.py`（Task 8 建立）追加一個測試，驗證 box_score 這種「按比賽當下 registration」查詢在球員換隊/換背號後仍能正確反映「那一場」的位置（不是最新位置）：
   ```python
   BOX_SCORE_SQL = """
       SELECT p.name, r.position, s.total_points
       FROM player_match_stats s
       JOIN roster_registrations r ON s.registration_id = r.registration_id
       JOIN players p ON r.player_id = p.player_id
       WHERE r.team_id = ? AND r.gender = ?
         AND s.match_date = ?
       ORDER BY s.total_points DESC
   """


   def test_box_score_query_reflects_position_at_time_of_match(tmp_db_path):
       conn = sqlite3.connect(tmp_db_path)
       conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
       conn.execute("INSERT INTO teams VALUES (5, '新北中纖', 'F')")
       conn.execute("INSERT INTO players (name, gender) VALUES ('球員A', 'F')")
       pid = conn.execute("SELECT player_id FROM players").fetchone()[0]

       # 第1週登記為 OP，第2週改登記為 MB（模擬位置調整）
       conn.execute(
           "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, jersey_number, position, source) "
           "VALUES (?, 5, 'F', '例行賽 Week 1', 2, 'OP', 'match_page')", (pid,),
       )
       rid1 = conn.execute("SELECT registration_id FROM roster_registrations").fetchone()[0]
       conn.execute(
           "INSERT INTO roster_registrations (player_id, team_id, gender, week_label, jersey_number, position, source) "
           "VALUES (?, 5, 'F', '例行賽 Week 2', 2, 'MB', 'match_page')", (pid,),
       )
       rid2 = conn.execute(
           "SELECT registration_id FROM roster_registrations WHERE week_label = '例行賽 Week 2'"
       ).fetchone()[0]

       conn.execute(
           "INSERT INTO player_match_stats (registration_id, match_date, total_points) VALUES (?, '2025-11-01', 10)", (rid1,),
       )
       conn.execute(
           "INSERT INTO player_match_stats (registration_id, match_date, total_points) VALUES (?, '2025-11-08', 15)", (rid2,),
       )
       conn.commit()

       week1_rows = conn.execute(BOX_SCORE_SQL, (5, "F", "2025-11-01")).fetchall()
       week2_rows = conn.execute(BOX_SCORE_SQL, (5, "F", "2025-11-08")).fetchall()

       assert week1_rows == [("球員A", "OP", 10)]
       assert week2_rows == [("球員A", "MB", 15)]
       conn.close()
   ```

- [ ] **Step 5:** 跑測試：
   ```bash
   python -m pytest tests/test_tab_queries_phase2.py -v
   ```
   **預期輸出**：2 個測試 PASSED（Task 8 的 1 個 + 本 task 新增的 1 個）。

- [ ] **Step 6:** 確認行尾乾淨：`git diff --stat -w`

- [ ] **Step 7:** **[STOP：等待使用者同意 commit]**：
   ```
   refactor: box_score 改透過 registration_id 查該場當下的登錄資訊

   單場 box score 不需要「最新一週」邏輯，registration_id 本身就
   是那一場比賽當下的正確登錄，直接 JOIN 即可反映當週真實背號/位置。
   ```

---

## Task 10：`league_pr.py` 迴歸驗證（不預期需要改動）

**Files:**
- Modify（如驗證後發現需要）: `src/app/tabs/league_pr.py`
- Create: `tests/test_league_pr_regression.py`

**Interfaces:**
- Consumes：`src.app.helpers.get_league_aggregated_stats(gender_code)`（Task 6 已改寫其內部 SQL，但回傳的 DataFrame 欄位集合——`player_id, name, position, team_name, ...` 全部沿用——完全沒變）。

**步驟：**

- [ ] **Step 1:** 檢查 `src/app/tabs/league_pr.py` 全文，確認它只透過 `get_league_aggregated_stats()` 存取資料庫，本身不含任何 SQL 字串：
   ```bash
   grep -n "SELECT\|player_match_stats\|FROM players" src/app/tabs/league_pr.py
   ```
   **預期輸出**：無匹配（exit code 1）。若有匹配，代表這裡藏了一段本規劃 session 沒發現的直接查詢，需要另外評估是否要比照 Task 6-9 的模式改寫（在此情況下，**不要**盲目套用某一種 JOIN 粒度，先判斷這段查詢屬於「單場」「球員彙總」還是「球隊目前名單」哪一種語意，再對應套用 Task 6 說明的三種模式之一）。

- [ ] **Step 2:** 若步驟 1 確認無直接 SQL（本規劃 session 的原始碼檢視結果是如此），新增 `tests/test_league_pr_regression.py` 做純結構性迴歸測試（確認 `league_pr.py` 依賴的 `ctx` 欄位與 `get_league_aggregated_stats()` 回傳欄位仍然對得上）：
   ```python
   from pathlib import Path

   LEAGUE_PR_PY = Path(__file__).resolve().parents[1] / "src" / "app" / "tabs" / "league_pr.py"
   HELPERS_PY = Path(__file__).resolve().parents[1] / "src" / "app" / "helpers.py"


   def test_league_pr_has_no_direct_sql():
       source = LEAGUE_PR_PY.read_text(encoding="utf-8")
       assert "SELECT" not in source.upper()


   def test_league_pr_uses_get_league_aggregated_stats():
       source = LEAGUE_PR_PY.read_text(encoding="utf-8")
       assert "get_league_aggregated_stats" in source


   def test_helpers_get_league_aggregated_stats_still_returns_position_column():
       source = HELPERS_PY.read_text(encoding="utf-8")
       assert "latest.position AS position" in source
   ```

- [ ] **Step 3:** 跑測試：
   ```bash
   python -m pytest tests/test_league_pr_regression.py -v
   ```
   **預期輸出**：3 個測試 PASSED。

- [ ] **Step 4:** 確認行尾乾淨：`git diff --stat -w`

- [ ] **Step 5:** **[STOP：等待使用者同意 commit]**（若步驟 1 確認無需修改 `league_pr.py` 本體，commit 只包含新測試檔）：
   ```
   test: league_pr 迴歸測試，確認其不受 Phase 2 schema 變動影響

   league_pr.py 全部透過 get_league_aggregated_stats() 存取資料，
   該函式回傳欄位集合在 Phase 2 前後保持一致，故本身不需改動。
   ```

---

## Task 11：全套整合驗證 + `prediction.py` 迴歸檢查

**Files:**
- 不改動任何檔案（純驗證 task）。

**步驟：**

- [ ] **Step 1:** 確認 `prediction.py` 只透過 `get_league_aggregated_stats()` 存取資料庫，本身不含任何 SQL 字串（`prediction.py` 已在規劃 session 確認只透過 `helpers.get_league_aggregated_stats` 存取資料，此處仍需實際跑一次確認）：
   ```bash
   grep -n "SELECT\|player_match_stats\|FROM players" src/app/tabs/prediction.py
   ```
   **預期輸出**：無匹配（exit code 1）。若有匹配，代表這裡藏了一段本規劃 session 沒發現的直接查詢，需要另外評估是否要比照 Task 6-9 的模式改寫（在此情況下，**不要**盲目套用某一種 JOIN 粒度，先判斷這段查詢屬於「單場」「球員彙總」還是「球隊目前名單」哪一種語意，再對應套用 Task 6 說明的三種模式之一）。

- [ ] **Step 2:** 跑全部測試：
   ```bash
   python -m pytest tests/ -v
   ```
   **預期輸出**：全部 PASSED（累計 Phase 1 全部 + Phase 2 Task 1-10 全部測試）。

- [ ] **Step 3:** 用真實資料庫（遷移後）跑一次每個 tab 依賴的核心查詢，確認回傳非空、無 SQL 錯誤（在 repo 根目錄執行）：
   ```bash
   python3 -c "
   import sqlite3
   conn = sqlite3.connect('data/db/tvl_database.db')

   # get_current_roster 邏輯
   rows = conn.execute('''
       SELECT r.player_id, r.jersey_number, p.name, r.position
       FROM roster_registrations r JOIN players p ON r.player_id = p.player_id
       WHERE r.team_id = 5 AND r.gender = 'F'
         AND r.week_start_date = (SELECT MAX(week_start_date) FROM roster_registrations WHERE team_id=5 AND gender='F')
   ''').fetchall()
   print('current roster (team_id=5,F):', len(rows), '筆')
   assert len(rows) > 0

   # player_deep / match_trend 邏輯：抓任一球員的全部統計
   any_player = conn.execute('SELECT player_id FROM players LIMIT 1').fetchone()[0]
   rows2 = conn.execute('''
       SELECT s.* FROM player_match_stats s
       JOIN roster_registrations r ON s.registration_id = r.registration_id
       WHERE r.player_id = ?
   ''', (any_player,)).fetchall()
   print('player stats via registration join:', len(rows2), '筆')

   # 孤兒檢查
   orphans = conn.execute('''
       SELECT COUNT(*) FROM player_match_stats s
       WHERE NOT EXISTS (SELECT 1 FROM roster_registrations r WHERE r.registration_id = s.registration_id)
   ''').fetchone()[0]
   print('孤兒 FK 數:', orphans)
   assert orphans == 0

   print('全套查詢驗證通過')
   "
   ```
   **預期輸出**：`全套查詢驗證通過`，且 `孤兒 FK 數: 0`。

   **未實測**：需執行者在完成 Task 4 遷移之後，於自己的環境對正式 DB 實跑確認。

- [ ] **Step 4:** 若有能力啟動 Streamlit（有完整依賴環境），實際跑一次 app 做視覺確認：
   ```bash
   streamlit run src/app/main.py
   ```
   手動檢查：側欄選單能正常選組別/球隊/球員、五個 tab 都能正常渲染無紅頁例外、`box_score` 分頁選一場過去的比賽能看到雙方名單且位置正確。

   **未實測**：需執行者手動操作瀏覽器確認，此步驟無法自動化驗證，是最終上線前的人工把關。

- [ ] **Step 5:** 確認整體 repo 狀態：
   ```bash
   git status
   git log --oneline -15
   ```
   **預期輸出**：working tree clean（所有 task 的 commit 都已完成），`git log` 可看到 Phase 2 每個 task 各自獨立的 commit 歷史。

---

## Phase 2 完工檢查清單

- [ ] `sql/schema.sql` 的 `players` 表不含 `team_id`/`jersey_number`/`position`
- [ ] `roster_registrations` 表存在，`UNIQUE (player_id, team_id, gender, week_label)` 約束生效
- [ ] `player_match_stats.registration_id` FK 指向 `roster_registrations`，無孤兒列
- [ ] 遷移後 `player_match_stats` 總筆數與遷移前一致（不得減少）
- [ ] `roster_registrations` 中 `source='match_page'` 筆數遠大於 `source='backfill'`
- [ ] `db_loader.py` 不再寫入任何 team/jersey/position 到 `players`
- [ ] `main.py`、`player_deep.py`、`match_trend.py`、`box_score.py`、`league_pr.py`、`prediction.py` 六個 tab 全部驗證過能正常運作
- [ ] `python -m pytest tests/ -v` 全數 PASSED
- [ ] 遷移前備份檔（`data/db/tvl_database.db.bak-*`）確實存在於本機
- [ ] 所有 commit 都經使用者同意後才執行
- [ ] `git diff --stat -w` 對每次 commit 都確認過無行尾雜訊

完成以上檢查後，Phase 2 結束，資料模型正式支援「每週登錄名單可能不同」的企業排球聯賽現實。
