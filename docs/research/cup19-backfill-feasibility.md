# 2023-24（cup 19）回填探路

> 調查日期：2026-11-05。對象為後續回填票的維護者；本票只調查來源、保存代表頁與提出流程，不寫正式資料庫。
> 起點為 [#144](https://github.com/oldyi317/TVL-Analysis/issues/144)，依 [ADR 0010](../adr/0010-historical-backfill-from-statistics-domain.md) 與母票 [#139](https://github.com/oldyi317/TVL-Analysis/issues/139) 執行；前次調查見 [來源偵察](tvl-pre-cup20-data-sources.md)。本分支已依使用者指示合併 #139 的 `9ad1aca` 並重讀兩份文件；以下以本次已保存頁面重新核對，未追加網路抓取。

## 結論與範圍

**現有 `EXT_HEADERS` 能取得 cup 19 外部單場頁，樣本同時提供比分、月日、週次與球員統計。依 ADR 0010，正式回填以外部系統為來源，官網舊頁僅供交叉比對。** 這是抽樣可行的結論，尚未證明全季每場都有完整單場數據。外部隊伍編號已找到 11 隊，其中 10 隊能對應既有身分；極速超跑為新增非現役隊候選。凱撒飯店連鎖與義力營造為同一隊的身分關係，由使用者於 2026-11-05 明確確認。

來源存取、隊伍表與姓名樣本是本次觀察；回填、確認檔與發布流程是下文建議，並非已存在的工具。**目前不能直接對正式 DB 執行既有 `stats_crawler` 回填 cup 19。**

## 本次取得的主要來源

代表頁放在 [`tests/fixtures/cup19/`](../../tests/fixtures/cup19/)。每份 `*_response.json` 記錄請求時間（UTC）、實際網址、沿用的請求標頭、HTTP 狀態，以及原始回應與 fixture 各自的長度、SHA-256。HTML 已正規化為 LF，部分官網頁擷取相關區塊；具體轉換記在 `fixture_transform`，不能把 fixture 雜湊當原始回應雜湊。後續測試應讀本地 fixture，不必再次連線。

| 來源 | 本次結果 | 本地代表頁 |
|---|---|---|
| [外部單場 MatchID 1](https://statistics.ctvba.org.tw/_handler/Match.ashx?CupID=19&MatchID=1&SetNum=0) | HTTP 200，男子第 1 週，雙方完整球員姓名與技術統計 | `match_1_response.html` |
| [外部單場 MatchID 2](https://statistics.ctvba.org.tw/_handler/Match.ashx?CupID=19&MatchID=2&SetNum=0) | HTTP 200，男子第 1 週，官方場次編號為 3 | `match_2_response.html` |
| [外部單場 MatchID 3](https://statistics.ctvba.org.tw/_handler/Match.ashx?CupID=19&MatchID=3&SetNum=0) | HTTP 200，男子第 1 週，官方場次編號為 5 | `match_3_response.html` |
| [外部單場 MatchID 101](https://statistics.ctvba.org.tw/_handler/Match.ashx?CupID=19&MatchID=101&SetNum=0) | HTTP 200，女子第 15 週，官方場次編號為 89 | `match_101_response.html` |
| [外部得分排行](https://statistics.ctvba.org.tw/_handler/Statistics.ashx?CupID=19&CatID=1) | HTTP 200，男女各前 20，連結涵蓋 11 支隊伍 | `statistics_response.html` |
| [極速超跑球員選單](https://statistics.ctvba.org.tw/_handler/PlayerList.ashx?CupID=19&TeamID=10)、[凱撒飯店連鎖球員選單](https://statistics.ctvba.org.tw/_handler/PlayerList.ashx?CupID=19&TeamID=11) | HTTP 200，分別有 14、16 個球員選項；包括姓名、背號、外部 PlayerID | `players_10_response.html`、`players_11_response.html` |
| [外部首頁](https://statistics.ctvba.org.tw/Default.aspx?CupID=19)、[場次選單頁](https://statistics.ctvba.org.tw/Match.aspx?CupID=19) | HTTP 404；不能依賴這兩頁取得場次清單 | `default_response.html`、`match_page_response.html` |
| [官網男子舊列表](https://tvl.ctvba.org.tw/schedule/112) | HTTP 200，70 個不同連結，`/game/129`–`/game/198` | `schedule_response.html` |
| [官網女子舊列表](https://tvl.ctvba.org.tw/wschedule/112) | HTTP 200，59 個不同連結，`/wgame/122`–`/wgame/180` | `wschedule_response.html` |
| [官網男子首場](https://tvl.ctvba.org.tw/game/129) | HTTP 200，2023-10-14，Week 1，逐局比分 | `official_game_129_response.html` |
| [官網男子 game 130](https://tvl.ctvba.org.tw/game/130) | HTTP 200，2023-10-14，Week 1，Game 3，與外部 MatchID 2 對應 | `official_game_130_response.html` |
| [官網女子 game 166](https://tvl.ctvba.org.tw/wgame/166) | HTTP 200，2024-02-25，Week 16，Game 94 | `official_wgame_166_response.html` |
| [官網女子末場](https://tvl.ctvba.org.tw/wgame/180) | HTTP 200，2024-04-01，總冠軍賽 Week 21，逐局比分 | `official_wgame_180_response.html` |
| [官網新聞 423](https://tvl.ctvba.org.tw/news-detail/423) | HTTP 200，2024-10-17 文章提到連莊的外援得分王「布萊恩」 | `news_423_response.html` |

共 16 次 HTTP 請求：14 個 200、2 個 404；未收到 HTTP 403、`cf-mitigated: challenge` 或要求完成驗證的阻擋頁。兩份 404 原回應附帶 Cloudflare `challenge-platform` 注入腳本，已在 metadata 另記並從 fixture 移除；沒有執行腳本或取得驗證憑證，`challenge_detected=false` 僅代表當時標記檢測結果，不能當作不存在任何防護程式的證據。它只證明這次請求可取得資料，不能保證後續政策不變。實際請求紀錄以各 metadata 為準，後續仍須每頁間隔 1–2 秒，遇 HTTP 403 或驗證挑戰立即停止，不換工具、網路或標頭繞過。

## 比分、日期、週次怎麼取得

### 觀察

- 外部 `Match.ashx` 自身有隊名、各局得分、勝局數、月日與「第 1 週」等標題資訊；只有月日，需由已確認賽季區間轉完整日期。球員列有背號、姓名、位置及技術統計。
- 官網舊列表提供可列舉的賽事網址，單場頁的 `.game_header` 提供完整日期、週次與階段，`.match_table` 提供各局比分。上述結構已被現有 [`match_crawler.py`](../../src/etl/match_crawler.py) 使用。
- 外部 MatchID 1 與官網 game 129 同為 2023-10-14 臺北Conti 對桃園臺灣產險：前三局分別 29–27、25–21、25–16，總分 79–64，臺北Conti 3–0 獲勝。外部標題「第1週」與官網「例行賽 Week 1」相符。
- 外部 MatchID 2 與官網 game 130 同為當日連莊對屏東台電，官方編號 3；五局分別 20–25、25–22、23–25、25–21、15–10，總分 108–103，連莊 3–2 獲勝。
- 外部 MatchID 101 標題為女子第 15 週、2 月 18 日、官方編號 89；雙方為臺北鯨華與凱撒飯店連鎖。cup 19 區間下對應 2024-02-18，前三局 25–11、25–22、25–17。這是女子外部單場樣本；尚未與該場官網頁逐局核對。
- 女子 wgame 180 為 2024-04-01 臺北鯨華對極速超跑，前三局 23–25、16–25、16–25，總分 55–75。這證明末段賽事仍可取得比分及階段；不代表整個區間已逐場核過。
- **外部 MatchID、外部標題的官方場次編號、官網 game_id 是不同識別值。** MatchID 2 對應官方編號 3，MatchID 101 對應官方編號 89，不能把網址 ID 作等差映射。外部 MatchID 1–3 全是男子組，不能由奇偶數猜組別。

### 建議

採「外部單場比分／月日／週次／名單／技術統計 → 季內日期解析與階段正規化 → 隔離匯入」；官網列表與單場頁只用來核對覆蓋範圍、日期、週次與逐局比分，不能作正式回填的備援資料來源。這也避免使用已重綁當季名單的官網歷史姓名，符合 ADR 0010 的來源邊界。

完整日期須用經確認的 cup 19 賽季區間，將外部月日解析為季內唯一日期，例如 10/14 對應 2023-10-14、2/18 對應 2024-02-18；可沿用 [`resolve_date_in_season()`](../../src/utils/seasons.py)。無唯一日期、週次或階段不明時列待確認，不拿官網數值補缺。官網末場樣本能作季後賽對照，但本票尚未取得外部季後賽樣本，不能據此宣稱季後賽已可正式回填。

外部場次需以組別、完整日期、雙方隊伍、官方場次編號與是否黃金決勝局交叉對應；逐局比分再驗一次，不依兩邊網址 ID 推算。官網列表 70＋59 個連結是本次取得的網址清單大小，可能含明星賽、季後賽與黃金決勝局；回填前須逐場分類，不能直接宣稱有 129 場有效聯賽單場數據。

現有 [`resolve_week_label()`](../../src/etl/stats_crawler.py) 只信 `matches` 同日期、同 cup 的 `round_name`，不把外部標題當後備週次；查無就拒寫。後續需先實作以外部單場資訊建立隔離 `matches` 的流程，再建立週登錄與單場數據；不能直接用官網 `match_crawler` 寫入舊季來避開此依賴。另須決定外部場次識別與既有 `matches.game_id`、賽事頁連結的對應，不能直接把外部 MatchID 當官網 ID。若同日期存在不同階段，還須核對既有按日期查週次是否足夠，避免把當天兩組或特殊賽事錯歸一週。

## cup 19 外部隊伍對照草稿

外部編號與當季顯示名均取自本次 [`statistics_response.html`](../../tests/fixtures/cup19/statistics_response.html) 的 `Team.aspx?CupID=19&TeamID=…` 連結；組別依男女分表。既有身分來自 [`sql/team_seed.sql`](../../sql/team_seed.sql)，主鍵是 `(team_id, gender)`，不能省略組別。

| 外部 TeamID | 組別 | cup 19 來源隊名 | 本地身分／處理 |
|---|---|---|---|
| 1 | M | 屏東台電 | `(1,M)` 屏東台電 |
| 2 | M | 雲林美津濃 | `(2,M)` 雲林美津濃 |
| 3 | M | 彰化三大有線 | `(8,M)` 已有非現役隊，不新增 |
| 4 | M | 臺北Conti | `(9,M)` 已有非現役隊，不新增 |
| 5 | M | 桃園臺灣產險 | `(5,M)` 現行龍邁建設；既有別名已確認同一隊，補 cup 19 當季隊名 |
| 6 | M | 連莊排球隊 | `(10,M)` 連莊，已有非現役隊；補來源寫法別名 |
| 7 | F | 高雄台電 | `(4,F)` 高雄台電 |
| 8 | F | 臺北鯨華 | `(3,F)` 臺北鯨華 |
| 9 | F | 新北中纖 | `(5,F)` 新北中纖 |
| 10 | F | 極速超跑 | 種子查無；新增非現役隊候選，本票不配正式 ID、不寫 DB |
| 11 | F | 凱撒飯店連鎖 | `(7,F)` 現行義力營造；使用者於 2026-11-05 確認為同隊沿革，補 cup 19 當季隊名「凱撒飯店連鎖」及來源別名，不另建隊 |

這是 cup 19 專屬對照；外部 TeamID 每季重編，不能拿其他季 `EXT_TEAM_MAPS` 或 `team_aliases` 的數字別名套用。球員頁連結中的 PlayerID 同樣須帶 cup_id，不視為跨季穩定身分。

## 球員姓名抽樣與確認方式

### 已觀察樣本

外部姓名與外部 PlayerID 出自本地單場頁／排行 fixture；本地 player_id 來自本次主 agent 的正式 DB 只讀查詢，證據見 [`local_identity_snapshot.json`](../../tests/fixtures/cup19/local_identity_snapshot.json)，沒有據此合併或寫入。

| 類型 | 來源樣本 | 與本地的比對結果與限制 |
|---|---|---|
| 同名候選 | MatchID 1 臺北Conti「郭柏毅」，外部 PlayerID 47 | 本地 `(35,M)` 同名，作既有身分候選；外部 PlayerID 47 不是本地 player_id 47 |
| 同名候選 | MatchID 1「雷諾」；MatchID 101「林良黛」「林書荷」 | 分別有本地 `(47,M)`、`(81,F)`、`(84,F)` 同名；仍要核對隊伍與跨季履歷，不能只憑字串斷言同一人 |
| 譯名／外文寫法待確認 | MatchID 2 連莊「B.BAGUNAS」，外部 PlayerID 76；官網新聞 423 的連莊外援「布萊恩」 | 本地查無 B.BAGUNAS 原字。當季排行第一名為 B.BAGUNAS，次季開季新聞稱連莊得分王外援為布萊恩，支持兩種寫法為同人的候選推論；文章未明列 B.BAGUNAS，仍保留人工確認，不自動合併或指定本地 ID |
| 新面孔候選 | MatchID 1 臺北Conti「周冠宇」，背號 16 | 本地查無同名；只是「本地尚無此名字」，可能是新身分或不同寫法，先列新增候選 |

同名並非同人證明；跨組別同名應保留不同身分，同季同名異隊則可能是轉隊或同名者。現有 `normalize_name()` 只去空白與轉小寫，不處理中英文譯名。`build_name_to_pid()` 對同名同性別只保留一個 ID，會遮住碰撞；後續人工確認工具不能直接拿這個 dict 當已核定結果。

抽樣限制：本次沒有確認「外部譯名不同、已能對上既有球員 ID」的案例，也未以完整名單證實同季同名異隊。上述譯名是待核候選，不能視為這兩類問題已排除。

### 建議採用的人工確認與回寫方式

1. 在隔離 DB／候選資料中先產清單，每列至少保留 `cup_id`、外部 TeamID、外部 PlayerID、組別、原始姓名、背號、來源網址、候選本地 player_id、比對原因。來源原字不覆寫。
2. 維護者在版本控制的確認檔填 `existing`、`new`、`unresolved`，並填目標本地 player_id 或新身分、證據連結、確認人與日期。相同外部身分的決定應一次套用全部場次；模糊譯名與同名異隊不得默認通過。
3. `existing` 在匯入前直接指定既有 player_id；`new` 才建立身分；`unresolved` 阻止該季發布。依確認檔重建隔離結果，避免先寫正式 DB 再人工修改。
4. 如候選階段已重複建人，先計算合併後 `roster_registrations` 的五元唯一鍵與 `season_rosters` 的三元唯一鍵衝突。將單場數據指向保留的登錄後，檢查同球員同日期／黃金決勝局是否出現重複；衝突列保留供覆核，不相加、不靜默丟棄。
5. 最終只在隔離結果通過人工確認、筆數核對、去重及外鍵檢查後，才由後續票處理正式回填。**本票沒有提供可直接執行的球員合併工具或 SQL。**

## 後續實作必先補的條件

- `EXT_BASE` 尚是舊 IP；`EXT_TEAM_MAPS` 尚無 cup 19；`seasons` 尚無 cup 19。先在隔離環境完成來源與對照配置，正式資料的加入由後續票處理。
- 現有抓取有 0.2／0.5 秒間隔；部分函式不檢查 HTTP 狀態，名單批次遇抓取錯誤會繼續。後續必補每頁 1–2 秒、403／挑戰全流程即停，並在真正停止後保存進度；不能只把網域改掉就跑。
- 場次選單頁 404，既有逐號掃描也把 500 視為不存在；500 同時可能表示壞資料。完整性驗收須對官網清單逐場核對，不能用掃描停止或「新增 0 筆」當完整證據。
- 尚未逐場確認男女所有外部場次、季後賽、黃金決勝局的 ID 與缺頁；雖已補女子例行賽樣本，仍需跨階段樣本及逐場覆蓋核對，再決定正式掃描邊界。
- 凱撒飯店連鎖已由使用者確認對應義力營造，後續依此補當季隊名。外籍球員譯名與未匹配姓名仍須解決；新增非現役隊仍需核對隊名、別名與隊色資料來源，不猜值。
- 在隔離 DB 驗證「fixture → 賽事／登錄／單場數據與待確認清單」；確認後才讓該季進入正式網站匯出，不把半季候選資料發布。

## 如果後續證實比分不全

本次已取得男女代表場的比分，**尚未證實比分取不到，因此目前不啟用降級**。若後續盤點證實某些場次無法取得可靠比分，應先列缺口與來源，再依 ADR 0010 決定只供歷史紀錄。

備案須另做明確的匯入與發布設計：保存可確認的外部單場日期、隊伍、球員及統計，週次只有來源明確時才採用；不造假比分、日期或週次。不要為了通過 `resolve_week_label()` 而新增虛構 `matches`。只供歷史紀錄時，不輸出戰績與可點擊的賽事頁，單場紀錄保留文字日期／對手，並標示資料涵蓋缺口。

這需要後續票調整既有依賴 `matches` 的登錄流程，以及「歷史紀錄可見、完整賽季仍隱藏」的匯出門檻；現有 `seasons` 本身沒有發布狀態欄位，不能宣稱已有此開關。若連日期或球員身分也無法確認，保留候選資料供覆核，不寫正式單場數據。

## 驗收對照

| #144 項目 | 本次結果 |
|---|---|
| 現有標頭能否取得 cup 19 單場頁 | 可；已保存四頁（男女皆有）成功回應及 metadata |
| 比分、日期、週次的來源 | 外部單場頁提供比分、月日及週次，日期按已確認賽季解析；官網僅交叉核對，全季完整性留給實際回填盤點 |
| 外部隊伍編號及本地對照 | 已列 11 隊；10 隊可對應（包含使用者確認的義力營造沿革）、1 隊新增候選 |
| 姓名抽樣 | 同名、新面孔與外文譯名疑似案例已列；尚無已證實對應既有 ID 的異譯案例，須保留此驗收限制 |
| 人工確認與回寫 | 建議版本化決策檔＋隔離重建；未新增正式合併工具 |
| 代表頁 fixture | 已保存代表頁、請求證據及原始／擷取後雜湊，供後續離線測試 |
| 結論寫回 issue | 由本票交付時留言並連到本報告 |
| 不改正式資料庫 | 本票只保存來源／報告及執行只讀比對，由主 agent 另核對 DB 基準 |
