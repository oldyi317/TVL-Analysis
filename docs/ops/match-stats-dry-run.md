# 官網單場數據 dry-run

從 repo 根目錄執行（#76）：

```bash
python -m src.etl.official_box_score --dry-run --game-id 281 --gender F
python -m src.etl.official_box_score --dry-run --game-id 331
```

組別預設 `M`（`/game/N`），`F` 使用 `/wgame/N`。每次只請求一頁，UA 與 15 秒逾時沿用 `match_crawler`；沒有連續請求需間隔。輸出為繁體中文未跳脫的 JSON，列出兩隊原始隊名、背號、官網球員 ID、位置、六項技術的成功／得分與總次數、得分與頁尾累計。此命令不開啟資料庫。

`parse_box_score(html)` 是純函式，回傳 `teams`、`warnings`、`errors`、`writable`。目前只解析官網原始隊名，後續寫入器才透過既有隊名別名解析球隊身分。`position` 取自共用 `POSITION_MAP`，查無時為 null，原文保留在 `position_text`，並對具名球員加一條「未知位置」警告。`official_player_id` 取自姓名儲存格的 `/player/N` 或 `/wplayer/N` 連結，沒有連結為 null。

- `source_row_count` 是原始球員列數；`players` 只包含具名球員，移除姓名尾端 `(C)`。女子樣本原始列數為 16／17，略過高雄台電的空白佔位列後為 16／16。
- `all_zero` 只標記六項技術共 12 個數值是否皆零；保留該球員列，是否入庫交由後續寫入器決定。`block_total` 也保留供核對。
- `player_points` 為表內球員累計，`summed_player_points` 為頁面所有非零列（含姓名空白者）的得分加總，兩者不符時整場不可寫入，錯誤列入 `gate_errors`（含隊名與差額）；這道閘門寫入器不可繞過。
- 姓名空白但有非零數值時，只列入 `errors`，原始數據保留在 `unresolved_rows`，依 #75 後補規則整場不可寫入。黃金決勝局等歷史頁因官網重綁名單常有此情形，可解析但不可寫入；寫入器可用 `--force-partial` 人工放行其餘球員。
- 球員累計＋對方失誤與頁面總計不符只警告；頁尾重複顯示的球員累計與表內不符也警告。此票不查 `matches` 比分、不比對本地球員身分、不寫入資料庫。
- 缺少區塊、欄位順序變動或數值無法解析時拋出 `ValueError`，不將未知值補成零。

CLI 在可寫入時回傳 exit 0（仍可能有警告），不可寫入、網路失敗或頁面格式異常時回傳 exit 1；命令參數錯誤回傳 exit 2。解析失敗會在 stderr 列出組別、game_id、URL 與原因。「可寫入」只代表解析閘門通過，未包含後續寫入器的身分與交易檢查。

## 寫入器（#77）

```bash
python -m src.etl.match_stats_writer                    # 每日步驟同款：本季候選場次全部處理
python -m src.etl.match_stats_writer --force-partial 281 --gender F   # 該場有未解析列時仍寫入其餘球員
```

候選場次＝`cup_id >= OFFICIAL_BOX_SCORE_FROM_CUP`（22）、`matches` 已有比分（`has_result`）、兩隊皆無同日同黃金局旗標的統計列；歷史賽季永不成為候選。已知限制：同隊同日兩場正規賽（單日雙賽）只有第一場會寫入，第二場因「同日同隊已有統計」不再成為候選。每場抓一頁、一個 transaction：閘門（`gate_errors`）、隊名對回主客隊、球員身分（賽季報名名單：同隊姓名 → 同隊背號 → 略過）、週登錄 upsert（`source='match_page'`、`week_label` 取該場 `matches.round_name`）、統計 insert 全部成功才 commit；任一失敗 rollback、log 以 `[F/281]` 開頭列出原因，下次每日更新重試。insert-only：寫過的場次不再成為候選，官網日後更正不會自動更新。

- 0-0 的球員只寫週登錄，不產生統計列；`sets_played` 一律 NULL；黃金決勝局頁寫 `is_golden_set=1`，匯出器照既有機制歸到母場次。
- 有球員對不上名單、或姓名空白但數值非零 → 整場不寫；確認後用 `--force-partial <game_id>` 寫入其餘球員（同 game_id 男女皆為候選時加 `--gender`）。Σ球員得分 ≠ 球員累計的閘門不可放行。
- 頁面總計與 `matches` 該隊總分不符只 warning 不擋。
- 週登錄的 `week_start_date` 取同賽季、同組別、同 `round_name` 的最早 `match_date`（與 `resolve_week_label` 不同處在於分組別）。
- 有比分但 `home_team_id`／`away_team_id` 仍為 NULL 的場次不成為候選，每次執行會 warning 列出（補 `sql/team_seed.sql` 後自然納入）。
- 單場遇到非預期例外同樣只 rollback 該場並記 log（含例外型別），其餘候選照常處理。
- 該賽季沒有 `season_rosters` 時整季略過（官網來源不動態新增球員）。
- exit code：全部候選寫入為 0，任一場略過為 1；`daily_update.sh` 以 `|| echo` 容錯。

## 測試樣本來源

2026-10-27 實抓後僅保留兩隊 `<h1>` 與 `#boxscore_1`、`#boxscore_11` 區段，放在 `tests/fixtures/boxscore_*.html`：

| 頁面 | 用途 |
|---|---|
| https://tvl.ctvba.org.tw/wgame/281 | 女子欄位對照、空白佔位列、16／17 原始列 |
| https://tvl.ctvba.org.tw/game/331 | cup 22 男子欄位與 `/player/N` 連結 |
| https://tvl.ctvba.org.tw/game/328 | cup 21 黃金決勝局、隊長 `(C)` 後綴 |

男子當季頁已先抓取確認：兩隊區塊 ID、欄位順序與前方 `<h1>` 均與女子頁一致；ID 實際位於外層 `<div>`，球員列在其中的 `<tbody>`，累計在 `<tfoot>`。黃金決勝局沿用同一解析流程。歷史頁可能已重綁球員姓名，樣本只供解析驗證，不用來回填歷史統計。
