# 排行榜浮動門檻 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 比率榜門檻隨所屬球隊已打局數浮動、只算例行賽、每局攔網改用球隊局數分母、加 MLB 9.22(a) 補算條款。

**Architecture:** 匯出器在 `site_data` 算出每個（賽季、組別、球隊）的已打局數與例行賽合計，傳進 `leaders.py` 的純函式；純函式負責門檻、補算與門檻文字；`leaders.json` 多兩個欄位（`scope`、列的 `makeup`），前端只多一個標籤與一句附註。

**Tech Stack:** Python 3 標準庫（匯出器不得引入第三方套件）、pytest、SvelteKit 5 + vitest、JSON Schema → `npm run gen` 產 `types.d.ts`。

**Spec:** GitHub issue #163（spec: 排行榜浮動門檻）；原排行榜 spec #106。

## Global Constraints

- 匯出器只用標準庫；排榜函式不碰資料庫。
- `web/src/lib/data/types.d.ts` 由 `web/schema/leaders.schema.json` 經 `npm run gen` 產生，不可手改；schema 皆 `additionalProperties: false`，新欄位要同時加進 `required`。
- 文案繁體中文，用詞過 `tests/test_vocabulary.py`（詞彙來源 CONTEXT.md）；不出現「排名」。
- web/ 縮排一律 tab；Python 沿用檔案既有風格。
- 行尾 LF；`.db` 與 `.pkl` 不動。
- 分支 `feat/issue-<N>-<簡述>`，每張子票一個 PR，PR 內文 `Closes #<N>`。
- 收工四項：`pytest`（`OMP_NUM_THREADS=2`）、`npm run check`、`npm run test:unit`、`npm run test:e2e`（在 web/ 下）。

## Review Focus

1. 球員本季只有季後賽統計（例行賽零列）→ 不出現在任何榜，頁首場次數也不含該場（Task 1 測試 `test_playoff_only_player_is_absent`）。
2. 球員所屬球隊不在 `team_sets`（例如登錄隊本季尚無單場數據）→ 門檻取下限 10，不得 KeyError（Task 2 測試 `test_missing_team_uses_floor`）。
3. 季初所有隊 ⌊係數 × S⌋ < 10 → 門檻文字為「…，季初至少 10 次（目前 10 次）」且榜單用 10（Task 2 測試 `test_threshold_text_at_floor`）。
4. 補算候選人攻擊 0 次（分母 0）→ 不是候選人；候選人補算值等於 V1 → 不觸發（Task 3 測試 `test_makeup_requires_strictly_higher`、`test_makeup_ignores_zero_attempts`）。
5. 兩隊 S 不同、同率並列的每局攔網 → 攔網得分多者在前，名次同號（Task 2 測試 `test_blocks_per_set_ties_order_by_block_points`）。

---

### Task 1: 例行賽範圍與球隊已打局數（匯出器接線）

**Files:**
- Modify: `src/export/site_data.py:424-435`（`_parent_matches`）、`src/export/site_data.py:524-542`（`_leaders_payload`）
- Modify: `src/export/leaders.py`（新增兩個純函式）
- Modify: `web/schema/leaders.schema.json`（GroupLeaders 加 `scope`）
- Test: `tests/test_leaders_rules.py`、`tests/test_export_leaders.py`

**Interfaces:**
- Consumes: `_box_score_games(conn)` 的 games dict（`(cup_id, gender)` → `game_id` → `{"game_id","date","home":[stat_entry],"away":[...],"golden_set"}`），stat_entry 含 `player_id`、`team_id` 與 STAT_FIELDS 13 欄。
- Produces:
  - `src/export/leaders.py`：`def regular_totals(games: Iterable[Mapping]) -> dict[int, dict]`，輸入例行賽 game dict（含 `home`/`away` 列，不含黃金局列），輸出 `player_id → {"games": int, 其餘 STAT_FIELDS 加總（sets_played 全非 None 才加總否則 None）}`。
  - `src/export/leaders.py`：`def count_team_sets(games: Iterable[Mapping]) -> dict[int, int]`，輸入例行賽 game dict 加上 `"home_team_id"`、`"away_team_id"`、`"sets"` 三鍵，輸出 `team_id → 該隊已打局數 S`。
  - `_parent_matches` 的 value tuple 由 `(cup_id, game_id, match_date, home_id)` 擴為 `(cup_id, game_id, match_date, home_id, away_id, round_name, sets)`，`sets = home_sets_won + away_sets_won`（未開打為 None）；`_box_score_games` 把 `home_team_id`、`away_team_id`、`round_name`、`sets` 寫進 game dict（它解包 `found` 的那行要同步改；`_match_for_stat` 只傳遞 tuple，不用改）。`_box_scores_payload` 輸出前已用 `k != "date"` 過濾鍵，改為同時排除這四個新鍵，`box_scores.json` 不變。
  - `_leaders_payload` 每個條目新增 `"scope": "regular"`；`through_date`、`games` 只算 `round_name` 以「例行賽」開頭的場次；傳給 `compute_leaders` 的列是 `_advanced_scopes` 的身分欄位（player_id、name、team_id、team_short、position）合併 `regular_totals` 的數字，沒有例行賽統計的球員不傳。

- [ ] **Step 1: 寫純函式失敗測試（tests/test_leaders_rules.py）**

```python
def game(game_id, home_rows, away_rows, home_team_id=1, away_team_id=2, sets=4, round_name="例行賽 Week 1"):
    return {"game_id": game_id, "date": "2025-11-01", "home": home_rows, "away": away_rows,
            "golden_set": None, "home_team_id": home_team_id, "away_team_id": away_team_id,
            "round_name": round_name, "sets": sets}

def test_regular_totals_sum_stat_fields_and_count_games():
    g1 = game(1, [entry(7, 1, attack_total=10, attack_points=4)], [])
    g2 = game(2, [], [entry(7, 1, attack_total=5, attack_points=1)])
    totals = regular_totals([g1, g2])
    assert totals[7]["games"] == 2 and totals[7]["attack_total"] == 15 and totals[7]["attack_points"] == 5
    assert totals[7]["sets_played"] is None

def test_team_sets_sum_both_sides():
    assert count_team_sets([game(1, [], [], 1, 2, sets=4), game(2, [], [], 2, 3, sets=5)]) == {1: 4, 2: 9, 3: 5}
```

`entry(player_id, team_id, **stats)` 是測試輔助：回 `{"player_id", "team_id", "jersey_number": None, "position": None, 13 個 STAT_FIELDS 預設 0（sets_played 預設 None）}` 再套 `**stats`。

- [ ] **Step 2: 跑測試確認失敗**

Run: `OMP_NUM_THREADS=2 pytest tests/test_leaders_rules.py -k "regular_totals or count_team_sets" -v`
Expected: FAIL，ImportError。

- [ ] **Step 3: 在 `src/export/leaders.py` 實作 `regular_totals` 與 `count_team_sets`**

`regular_totals` 走每場 `home + away` 列，每列算一場；`sets_played` 只要任一列為 None 就整季為 None（與 `_player_seasons` 同口徑）。

- [ ] **Step 4: 跑測試確認通過**

Run: 同 Step 2。Expected: PASS。

- [ ] **Step 5: 寫匯出器失敗測試（tests/test_export_leaders.py）**

新增 `MATCH_SQL_ROUND`（多 `round_name, home_sets_won, away_sets_won` 三欄）。

```python
def test_leaders_only_count_regular_season_and_expose_scope():
    def build(conn):
        seed_player(conn, 1, "甲"); seed_registration(conn, 1, 1, 1)
        conn.execute(MATCH_SQL_ROUND, (1, "M", 21, "2025-11-01", "屏東台電", "雲林美津濃", 1, 2, "例行賽 Week 1", 3, 1))
        conn.execute(MATCH_SQL_ROUND, (2, "M", 21, "2026-03-01", "屏東台電", "雲林美津濃", 1, 2, "總決賽 Week 19", 3, 0))
        seed_stat(conn, 1, stat(12), "2025-11-01", "雲林美津濃")
        seed_stat(conn, 1, stat(30), "2026-03-01", "雲林美津濃")
    payload, _ = export_json(build, "leaders")
    scope = payload["leaders"][0]
    assert scope["scope"] == "regular" and scope["games"] == 1 and scope["through_date"] == "2025-11-01"
    total = next(b for b in scope["boards"] if b["key"] == "total_points")
    assert total["rows"][0]["value"] == 12

def test_playoff_only_player_is_absent():
    # 球員 2 只有總決賽統計 → 不在任何榜，頁首 games 仍為例行賽場次數
```

- [ ] **Step 6: 跑測試確認失敗**

Run: `OMP_NUM_THREADS=2 pytest tests/test_export_leaders.py -v`
Expected: 新測試 FAIL（schema 無 `scope` 或數字含季後賽）。

- [ ] **Step 7: 改 `web/schema/leaders.schema.json`、`_parent_matches`、`_box_score_games`、`_leaders_payload`**

schema：GroupLeaders 加 `"scope": {"type": "string", "enum": ["regular"], "description": "排行榜只算例行賽"}` 並進 `required`。`_leaders_payload` 內：篩例行賽 game → `regular_totals` → 與 `_advanced_scopes` 列合併 → 暫時仍呼叫既有 `compute_leaders(named)`（Task 2 再換簽名）。

- [ ] **Step 8: 跑 pytest 全套與 `npm run gen`**

Run: `OMP_NUM_THREADS=2 pytest -q` 與 `cd web && npm run gen`
Expected: 全 PASS；`types.d.ts` 的 `GroupLeaders` 多 `scope: "regular"`。

- [ ] **Step 9: Commit**

```bash
git add src/export/leaders.py src/export/site_data.py web/schema/leaders.schema.json web/src/lib/data/types.d.ts tests/test_leaders_rules.py tests/test_export_leaders.py
git commit -m "feat: 排行榜只算例行賽並算出球隊已打局數（#<子票>）"
```

---

### Task 2: 浮動門檻、每局攔網分母與門檻文字（純函式）

**Files:**
- Modify: `src/export/leaders.py:9-40`（BoardSpec、RATE_BOARDS）、`:51-90`（`_board`、`compute_leaders`）、`:100-116`（`compute_volume_rates`）
- Modify: `src/export/site_data.py` `_leaders_payload`（傳入 team_sets）
- Test: `tests/test_leaders_rules.py`、`tests/test_export_leaders.py`

**Interfaces:**
- Consumes: Task 1 的 `team_sets` 與合併後的列（列含 STAT_FIELDS、games、team_id、position、name、team_short）。
- Produces:
  - `BoardSpec` 欄位：去掉 `attempts: str | None`，改為 `attempts_key: str | None`（`"attack_total"`、`"serve_total"`、`"receive_total"`、`"dig_total"`、`"set_total"`，場均得分與每局攔網為 `"attack_serve_total"`）、`attempts_label: str | None`（「攻擊次數」「發球次數」「接發球次數」「防守次數」「舉球次數」「攻擊＋發球次數」）、`coefficient: float | None`（2.0、1.3、1.3、1.0、5.0、2.0、2.0）。
  - 常數 `THRESHOLD_FLOOR = 10`。
  - `def threshold_for(spec: BoardSpec, sets: int) -> int`：`max(THRESHOLD_FLOOR, int(spec.coefficient * sets))`。
  - `def threshold_text(spec: BoardSpec, team_sets: Mapping[int, int]) -> str`：規則在下方測試。
  - `def compute_leaders(rows: list[dict], team_sets: Mapping[int, int], boards: Sequence[BoardSpec] = COUNT_BOARDS + RATE_BOARDS) -> list[dict]`；`def compute_volume_rates(rows: list[dict], team_sets: Mapping[int, int]) -> list[dict]`。
  - 列的參賽次數：`attempts(row, spec)`，`attack_serve_total` = `attack_total + serve_total`。
  - 每局攔網 `value = round(block_points / S * 1.0, 2)`，`denominator = S`；S 查無或 0 → 不上榜。
  - 不再 import `MIN_SETS`、`MIN_GAMES`；`compute_leaders` 不再因 sets_played 全 None 略過每局攔網。

- [ ] **Step 1: 改寫既有門檻測試並新增（tests/test_leaders_rules.py）**

刪除 `test_unknown_sets_use_games_and_omit_blocks_board`、`test_mixed_sets_describe_the_unknown_sets_fallback`；把其餘呼叫改成 `compute_leaders(rows, {1: 10})`。新增：

```python
def test_threshold_floats_with_team_sets():
    rows = [rate_row(1, attack_total=25, attack_points=10), rate_row(2, attack_total=24, attack_points=20)]
    attack = lambda sets: next(b for b in compute_leaders(rows, {1: sets}) if b["key"] == "attack_success_rate")
    assert [r["player_id"] for r in attack(12)["rows"]] == [2, 1]   # T=24：兩人達標，率高者在前
    assert attack(13)["rows"] == []                                  # T=26：都不達標

def test_threshold_uses_floor_below_ten():
    rows = [rate_row(1, attack_total=10, attack_points=5), rate_row(2, attack_total=9, attack_points=9)]
    board = next(b for b in compute_leaders(rows, {1: 3}) if b["key"] == "attack_success_rate")
    assert [r["player_id"] for r in board["rows"]] == [1]

def test_missing_team_uses_floor():
    rows = [rate_row(1, attack_total=10, attack_points=5)]
    board = next(b for b in compute_leaders(rows, {}) if b["key"] == "attack_success_rate")
    assert board["rows"][0]["player_id"] == 1

def test_each_rate_board_uses_its_own_coefficient():
    # 發球 1.3、接發 1.3、防守 1.0、舉球 5.0、場均得分與每局攔網看攻擊＋發球 2.0；S=20 → 26、26、20、100、40、40

def test_blocks_per_set_uses_team_sets_and_is_always_output():
    rows = [rate_row(1, block_points=6, sets_played=None)]
    board = next(b for b in compute_leaders(rows, {1: 20}) if b["key"] == "blocks_per_set")
    assert board["rows"][0]["value"] == 0.3 and board["rows"][0]["denominator"] == 20

def test_blocks_per_set_ties_order_by_block_points():
    rows = [rate_row(1, team_id=1, block_points=4), rate_row(2, team_id=2, block_points=8)]
    board = next(b for b in compute_leaders(rows, {1: 20, 2: 40}) if b["key"] == "blocks_per_set")
    assert [r["player_id"] for r in board["rows"]] == [2, 1] and [r["rank"] for r in board["rows"]] == [1, 1]

def test_threshold_text_formats():
    spec = next(s for s in RATE_BOARDS if s.key == "attack_success_rate")
    assert threshold_text(spec, {1: 10, 2: 14}) == "門檻：攻擊次數達所屬球隊已打局數 × 2（目前 20 到 28 次）"
    assert threshold_text(spec, {1: 12, 2: 12}) == "門檻：攻擊次數達所屬球隊已打局數 × 2（目前 24 次）"
    assert threshold_text(spec, {1: 3, 2: 6}) == "門檻：攻擊次數達所屬球隊已打局數 × 2，季初至少 10 次（目前 10 到 12 次）"
    assert threshold_text(spec, {}) == "門檻：攻擊次數達所屬球隊已打局數 × 2，季初至少 10 次（目前 10 次）"
    serve = next(s for s in RATE_BOARDS if s.key == "serve_point_rate")
    assert threshold_text(serve, {1: 20}) == "門檻：發球次數達所屬球隊已打局數 × 1.3（目前 26 次）"
    ppg = next(s for s in RATE_BOARDS if s.key == "points_per_game")
    assert threshold_text(ppg, {1: 20}) == "門檻：攻擊＋發球次數達所屬球隊已打局數 × 2（目前 40 次）"
```

`rate_row` 改成預設 `team_id=1`、所有 `*_total` 為 0，由測試明給。係數文字：整數係數寫 `2`、`5`，非整數寫 `1.3`。

- [ ] **Step 2: 跑測試確認失敗**

Run: `OMP_NUM_THREADS=2 pytest tests/test_leaders_rules.py -v`
Expected: 新測試 FAIL（TypeError 或 ImportError）。

- [ ] **Step 3: 實作 BoardSpec 新欄位、`threshold_for`、`threshold_text`、`_board` 過濾與每局攔網分母，改 `compute_leaders`／`compute_volume_rates` 簽名**

`_board(spec, rows, team_sets)`：比率榜先依 position 篩，再留下 `attempts(r, spec) >= threshold_for(spec, team_sets.get(r["team_id"], 0))` 且分母 > 0 的列；每局攔網的值與分母改用 S；排序鍵比率榜為 `(-value, -tiebreak, name)`，tiebreak 每局攔網用 `block_points`，其餘用分母。

- [ ] **Step 4: 跑純函式測試確認通過**

Run: 同 Step 2。Expected: 全 PASS。

- [ ] **Step 5: 匯出器接線測試（tests/test_export_leaders.py）**

```python
def test_rate_threshold_follows_each_teams_regular_sets():
    # 兩隊：隊 1 打 2 場共 8 局（T=16）、隊 2 打 1 場 3 局（T=10）；隊 1 球員攻擊 15 次不上榜、隊 2 球員攻擊 10 次上榜；threshold 文字「…季初至少 10 次（目前 10 到 16 次）」
def test_blocks_per_set_board_exists_for_official_source_season():
    # cup 22 風格：sets_played 為 None 的統計列 → leaders 含 blocks_per_set 榜
```

既有 `stat()` 的 sets_played=3 可保留；新測試用 `seed_stat` 傳 sets_played None。

- [ ] **Step 6: 改 `_leaders_payload` 傳入 `count_team_sets(...)`，跑測試**

Run: `OMP_NUM_THREADS=2 pytest -q`。Expected: 全 PASS。

- [ ] **Step 7: 前端 vitest 既有測試調整**

`web/src/routes/[season]/[group]/leaders/page.test.ts` 內「只傳入計數榜與六個比率榜時，不產生每局攔網卡與 chip」保留（前端行為不變）；fixture 若含門檻字串改為新格式。Run: `cd web && npm run test:unit`。Expected: PASS。

- [ ] **Step 8: Commit**

```bash
git add src/export/leaders.py src/export/site_data.py tests/ web/src
git commit -m "feat: 比率榜門檻隨球隊已打局數浮動，每局攔網改用球隊局數（#<子票>）"
```

---

### Task 3: 補算條款（純函式、schema、前端標籤與附註）

**Files:**
- Modify: `src/export/leaders.py`（BoardSpec 加 `makeup: bool = False`；`_board`）
- Modify: `web/schema/leaders.schema.json`（LeaderRow 加 `makeup`）、`web/src/lib/components/LeaderBoard.svelte`
- Test: `tests/test_leaders_rules.py`、`web/src/routes/[season]/[group]/leaders/page.test.ts`

**Interfaces:**
- Consumes: Task 2 的 `_board`、`threshold_for`。
- Produces:
  - `BoardSpec.makeup: bool`，攻擊、發球、接發球、防守、舉球五榜為 True。
  - 每列新增 `"makeup": bool`（所有榜都有，計數榜恆 False）。
  - `def makeup_candidate(spec: BoardSpec, rows: list[dict], team_sets: Mapping[int, int], leader_rate: float | None) -> dict | None`：未達標且 attempts > 0 的列中，補算值 `numerator / threshold_for(...) * 100` 嚴格大於 `leader_rate`（未四捨五入的 `numerator / denominator * 100`）者取最高；同補算值取 attempts 多者、再姓名。`leader_rate` 為 None（無達標者）回 None。
  - 前端：列有 `row.makeup` 時名字後 `<span class="tag">補算達標</span>`；卡內 table 之後、展開鈕之前一行 `<p class="note">補算達標：次數未達門檻，但把差額全算失敗後仍居首（MLB 規則 9.22(a) 同款）</p>`，只在該榜有 makeup 列時渲染。
  - `compute_volume_rates` 的點排除 `makeup` 列。

- [ ] **Step 1: 純函式失敗測試**

```python
def test_makeup_lists_short_player_first_with_actual_rate():
    rows = [rate_row(1, attack_total=40, attack_points=20), rate_row(2, attack_total=30, attack_points=25)]
    board = next(b for b in compute_leaders(rows, {1: 20}) if b["key"] == "attack_success_rate")   # T=40
    assert [(r["player_id"], r["rank"], r["makeup"]) for r in board["rows"]] == [(2, 1, True), (1, 2, False)]
    assert board["rows"][0]["value"] == 83.33 and board["rows"][0]["denominator"] == 30

def test_makeup_requires_strictly_higher():
    # 球員 2：attack_total=30, attack_points=20 → 補算 20/40=50% == V1 50% → 不觸發

def test_makeup_ignores_zero_attempts():
    # attack_total=0 的列不是候選

def test_makeup_skips_when_nobody_qualifies():
    # 無達標者 → 空榜，不補算

def test_makeup_takes_only_the_best_candidate_and_demotes_tied_leaders():
    # 兩位達標並列第一 + 兩位候選皆超過 → 只一位補算，名次 1、2、2、4

def test_makeup_not_applied_to_points_per_game_and_blocks_per_set():
    # 這兩榜 spec.makeup 為 False

def test_volume_rates_exclude_makeup_rows():
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `OMP_NUM_THREADS=2 pytest tests/test_leaders_rules.py -k makeup -v`。Expected: FAIL。

- [ ] **Step 3: 實作 `makeup_candidate` 並在 `_board` 接上；schema 加 `makeup`；`npm run gen`**

補算列插在 ordered 之前後再跑 `rank_ordered`（補算者實際率必大於 V1，名次自然為 1）。schema：LeaderRow `"makeup": {"type": "boolean", "description": "補算達標：次數未達門檻但補算後仍居首，只會出現在第一名"}` 進 `required`。

- [ ] **Step 4: 跑 pytest 全套**

Run: `OMP_NUM_THREADS=2 pytest -q`。Expected: PASS（含 schema 驗證）。

- [ ] **Step 5: 前端失敗測試（page.test.ts）**

```ts
it('補算達標的列顯示標籤，卡底有一句說明', ...)   // fixture 一列 makeup: true → 出現「補算達標」標籤與含「MLB 規則 9.22(a)」的附註
it('沒有補算列時不渲染附註', ...)
```

- [ ] **Step 6: 跑 vitest 確認失敗，改 `LeaderBoard.svelte`，再跑確認通過**

Run: `cd web && npm run test:unit`。Expected: 先 FAIL 後 PASS。

- [ ] **Step 7: Commit**

```bash
git add src/export/leaders.py web/schema/leaders.schema.json web/src tests/
git commit -m "feat: 比率榜加 MLB 9.22(a) 補算條款（#<子票>）"
```

---

### Task 4: 詞彙、ADR、實跑驗證與跟進票

**Files:**
- Modify: `CONTEXT.md`（排行榜節後新增三詞條、更新「排行榜」詞條）
- Create: `docs/adr/0011-leader-thresholds-on-team-sets.md`
- Modify: `web/src/lib/data/leaders.json`（由匯出器重產，gitignore 不進 commit）

**Interfaces:**
- Consumes: Task 1–3 全部。
- Produces: 詞條「浮動門檻」「補算達標」「球隊已打局數」；ADR 0011。

- [ ] **Step 1: CONTEXT.md 詞條**

```
**球隊已打局數**:
一個組別一個賽季內，某球隊例行賽、非黃金決勝局、已有單場數據的場次的總局數。排行榜比率榜的門檻與每局攔網的分母都以它為基準，不用個人出賽局數（資料來源沒有）。
_Avoid_: 球隊局數、team sets

**浮動門檻**:
比率榜的門檻＝該項次數達所屬球隊已打局數乘以係數，季初至少 10 次；隨賽季推進自動長高。係數：攻擊 2、發球 1.3、接發球 1.3、防守 1、舉球 5；場均得分與每局攔網看攻擊加發球次數乘 2。
_Avoid_: 規定次數、規定打席、動態門檻

**補算達標**:
比率榜次數未達門檻的球員，把差額全算失敗後若仍高於達標者的第一名，列為第一名並加此標籤；只影響第一名、只取一人。
_Avoid_: 補算條款、追加計算
```

「排行榜」詞條末句補「比率榜門檻見浮動門檻；排行榜只算例行賽」。「每局攔網」若有既有詞條，改分母為球隊已打局數。

- [ ] **Step 2: 跑詞彙測試**

Run: `OMP_NUM_THREADS=2 pytest tests/test_vocabulary.py -q`。Expected: PASS（若程式碼用了避免詞就改程式碼，不改詞條）。

- [ ] **Step 3: ADR 0011**

沿用 0009 格式（frontmatter status/date、H1、選項與決定、理由、同時定下的規則、`## Consequences`）。內容：資料庫沒有個人出賽局數（cup 20–21 `sets_played` 逐列等於整場局數，cup 22 起 NULL）；決定排行榜門檻與每局攔網以球隊已打局數為基準、只算例行賽、補算條款；Consequences 含「球員頁 PR 值篩人仍用舊判定，待跟進票」。

- [ ] **Step 4: 實跑匯出器與四項驗證**

Run（repo 根目錄）：`python -m src.export.site_data`；然後 `cd web && npm run check && npm run test:unit && npm run test:e2e`；再 `OMP_NUM_THREADS=2 pytest -q`。
Expected: 全過；`web/src/lib/data/leaders.json` 本季男女都有 `blocks_per_set` 榜、每個比率榜 `threshold` 為新格式。

- [ ] **Step 5: 本機預覽抽查**

`npm run preview` 開本季男子排行榜，確認：比率卡門檻文字一行、每局攔網卡出現、無補算列時沒有附註。

- [ ] **Step 6: 開跟進票**

`gh issue create`：「球員頁 PR 值與分佈改用浮動門檻篩人（#163 跟進）」，內文說明 `player_advanced.insufficient_sets` 仍用假局數。

- [ ] **Step 7: Commit**

```bash
git add CONTEXT.md docs/adr/0011-leader-thresholds-on-team-sets.md
git commit -m "docs: 排行榜浮動門檻詞條與 ADR 0011（#<子票>）"
```
