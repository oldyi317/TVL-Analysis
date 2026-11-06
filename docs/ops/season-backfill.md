# 歷史賽季回填（season_backfill）

依 ADR 0010，從 `https://statistics.ctvba.org.tw` 一次性補回舊賽季（目前 cup 16–19，2020-21、2021-22、2022-23、2023-24）的賽事各局得分、週登錄與單場數據。

## 怎麼跑

```bash
cp data/db/tvl_database.db /tmp/backfill.db        # 或用已套用 schema.sql 的空 DB；team_seed 由工具開跑前套用
python -m src.etl.season_backfill --cup-id 19 --db /tmp/backfill.db --review-out /tmp/review19.json
# 只補指定場次：--match-ids 1-150 或 1,2,101；掃描模式連續 --gap-limit（預設 20）個空號就停
```

- `--db` 必填、不給預設正式 DB：回填只在隔離 DB 上跑，何時搬進正式 DB 由 #146 決定（正式 seasons 多一列就會上站）。
- 每場一頁（只抓 `Match.ashx`），請求間隔 1–2 秒；遇 403 或驗證挑戰立即停止（exit code 2），已 commit 的場次保留，不重試、不換標頭。
- 可重跑：賽事以 `(game_id, gender)`、統計以 `(match_date, is_golden_set)` 去重，補缺不清表。`game_id = cup_id * 1000 + 外部 MatchID`。
- cup 22 起拒絕執行（ADR 0009）；新增賽季要先在 `constants.py` 補 `BACKFILL_SEASONS`、`EXT_TEAM_MAPS`（與必要的 `EXT_TEAM_SEASON_NAMES`）及 `sql/team_seed.sql`。
- 季後賽（季後賽／挑戰賽／總決賽，外部標題寫「總冠軍賽」）的外部標題沒有週次，首輪只寫 matches 並列入 `week_unresolved`；季後賽在外部 MatchID 201 起（cup 17、18、19 皆同），掃描模式會先在例行賽結束後連續空號停下（cup 17 於 101、cup 18 與 19 於 111），要用 `--match-ids 201-240` 明確指定。補法：用 `match_crawler.scrape_match_page` 抓官網對應單場頁（cup 19 為 `game/188-196`、`wgame/173-180`；cup 18 為 `game/120-128`、`wgame/112-119`，官網賽程列表 `schedule/111`；cup 17 為 `game/52-58`、`wgame/52-58`，賽程列表 `schedule/110`，單場頁沒有週次字樣，季後賽依日期接著例行賽編 Week 16–18）取得 `round_name`（如「總決賽 Week 21」），以日期＋組別＋兩隊對到隔離 DB 的列後 UPDATE `matches.round_name`，再以同一組 `--match-ids` 重跑，工具會依 `matches.round_name` 補週登錄與單場數據。官網只供週次標籤與比分核對（ADR 0010 列明的例外），球員資料仍只取外部系統。
- cup 16（2020-21）：外部 MatchID 男 1–60、女 61–120 連號，標題「編號：1」不帶男女字；女 編號 4、20、50（外部 MatchID 64、80、110；皆中國人纖勝，維基戰績表的差額正好是這 3 場）回 500、無頁面，該季女子組只有 57 場；季後賽（挑戰賽）8 場（2021-02-27 至 03-01）外部系統沒有頁面（121–310 皆 500），官網也沒有 2020-21 賽程頁（`/schedule/109` 落回當季）；這 8 場依使用者決定以維基百科各局比分（2/28 兩場與 3/1 女子組經自由體育報導核對）手動寫入 `matches`（game_id 16201–16208，時間與主客依官方海報），沒有週登錄與單場數據，是全站唯一不來自官網或外部系統的賽事列。核對來源改為 TVL 官方 2020 上／下半季賽程海報（使用者提供的圖檔，逐場核對日期、時間、主客、週次；無比分）與維基百科「企業十六年甲級男女排球聯賽」條目（女子組有逐場比分，男子組只有戰績表，以各隊勝敗場數核對）。場館沿用外部系統的簡寫（如「雲科大體育館」），不改成海報全名。
- 不進 `scripts/daily_update.sh`。

## 待人工確認清單（`--review-out` JSON）

| kind | 意思 |
|---|---|
| `new_player` | 其他賽季完全沒有週登錄、本次回填才出現的球員身分 |
| `same_name_multiple_teams_in_season` | 同一 player_id 在該賽季登錄於多隊（姓名比對會把同季同名異隊併成一人） |
| `same_name_other_group` | players 表另有同名但不同組別的球員（evidence 列出對方 player_id） |
| `same_name_other_team_in_history` | 該球員在其他賽季登錄的隊伍與本賽季不同（轉隊或同名者） |
| `week_unresolved` | 標題沒有週次、只寫了 matches，未寫週登錄與統計（以賽事為單位） |

確認後如何處理（搬進正式 DB、合併或拆分球員）交給 #146。
