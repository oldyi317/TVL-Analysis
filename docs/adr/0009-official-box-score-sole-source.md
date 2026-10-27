---
status: accepted
date: 2026-10-27
---

# 單場數據以官網賽事頁為唯一來源，停用外部統計系統的逐場路徑

`player_match_stats` 原由外部統計系統（`stats_crawler --incremental`）供應，但外部系統每季要等對方建季：2026-27（cup_id=22）到 2026-10-27 仍回 500，本季單場數據全空，且 `.aspx` 頁面已 404。官網賽事頁 `/game/N`、`/wgame/N` 的 server-side HTML 已含兩隊單場數據（`#boxscore_1`、`#boxscore_11`），`match_crawler` 每日本來就在下載同一頁。選項：(a) 官網為唯一來源；(b) 官網與外部系統並存，外部優先、官網補缺；(c) 維持外部系統、等建季。決定 **(a)**。

理由：(b) 兩個來源口徑不同（官網無出賽局數、姓名／背號綁定不同），並存要定義衝突規則，且外部系統建季時機不可控；(c) 本季資料持續缺席。官網來源每季不必手動更新常數。

同時定下的規則：

- **只對當日有比分且尚無統計的場次抓頁，寫入後不覆寫**：官網會把歷史頁姓名與背號重綁到現行名單（實測 cup 21 男子總決賽頁 28 列中 8 列姓名空白、屏東台電 12 筆非零列只剩 10 筆），只有比賽當日的頁可信。因此歷史賽季（cup 20、21）保留外部系統既有資料，不從官網回填；候選場次限 `cup_id >= OFFICIAL_BOX_SCORE_FROM_CUP`（22）。
- **獨立步驟，不寫 matches**：`match_stats_writer` 排在 `match_crawler` 之後（ADR 0008 的順序要求不變），`matches` 欄位所有權不變。
- **欄位**：攻擊、攔網、發球、接發球、防守、舉球各取「得分／成功 - 總次數」前者為 *_points／*_excellent、後者為 *_total（攔網總次數不存），得分欄為 total_points。官網沒有出賽局數，`sets_played` 從此來源起為 NULL，下游不得以整場局數代替個人局數。
- **0-0 的球員不產生統計列**：官網連沒上場的人也列出，且無局數可區分；週登錄仍為他們寫入。與 cup 20、21（外部系統含全 0 列）口徑不同，跨季的「出賽場數」不可直接比較。
- **球員身分沿用賽季報名名單**（同隊姓名 → 同隊背號 → 略過），不新增官網球員 id 欄位。
- **週登錄由官網單場數據取代 `--rosters`**：`source='match_page'`。

## Consequences

- `daily_update.sh` 移除 `stats_crawler --rosters` 與 `--incremental`；`stats_crawler.py` 與 `EXT_*` 常數留作歷史補抓工具，`main`、`crawl_all_rosters` 與 CLI 對 `cup_id >= 22` 以 `SystemExit` 拒絕執行並提示改走 `match_stats_writer`（`tests/test_stats_crawler_official_source_guard.py`）。`EXT_CUP_ID` 固定 21，`EXT_TEAM_MAPS` 不補 cup 22，換季 checklist 不再有外部系統項目。
- ADR 0003 與 ADR 0008 中「排在 `stats_crawler` 之後／之前」的順序句子改指 `match_stats_writer`。
- `player_advanced`、`features.py` 與前端須處理 `sets_played` 為 NULL（#78、#79）。
- 已知限制：入庫後官網若更正數字不會更新（需手動重匯）；同隊同名或賽季中改名會被略過並列清單；官網日後若移除某球員，當日之後才補抓的場次可能出現空白姓名列。
