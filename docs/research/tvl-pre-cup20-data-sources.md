# TVL cup 19 以前球員數據來源偵察

> 偵察日期：2026-11-05。只抓了少量頁面確認來源在不在、長什麼樣，沒有大量下載。
> 下文「實測」指本次實際取得並檢視過內容；「推估」指只依 Wayback 索引時間戳推論、未逐頁打開。

## 結論摘要

1. **cup 13–19（2017-18 到 2023-24）可救回逐場球員統計，粒度與 cup 20、21 相同。** 外部統計系統其實還活著，網域是 `statistics.ctvba.org.tw`（Cloudflare 代理），`_handler/Match.ashx` 對 CupID 13、14、15、16、17、18、19、21 都實測回傳有球員姓名的完整 box score。這和 `stats_crawler` 打的是同一支端點，`.aspx` 一樣是 404，但 `.ashx` 正常。CupID 12 回 500，所以這套系統最早只到 cup 13，和使用者記得的「過去 7、8 季」相符。
2. **官網 cup 17–19 的比賽頁也有逐場 box score**（`/game/1–198`、`/wgame/1–180`），欄位格式和 cup 22 一樣。缺點是名單只留當季，舊賽季大量球員列只剩背號、沒有姓名。要補姓名得靠 Wayback 舊快照或上面的統計系統。
3. **cup 7（2011）有一小塊逐場統計 PDF**：協會舊站每場都有一份，但 Wayback 只存到 2 份。**cup 1–6、8–12 查無球員數據**，只有 `/champions` 列出的團隊冠軍。
4. 存取限制：`114.35.229.141` 整站回 403（換網路也一樣，見下文）。`tvl.ctvba.org.tw` 和 `statistics.ctvba.org.tw` 都在 Cloudflare 後面，用 curl 預設 UA 會拿到 403，用 WebFetch 則能取得內容。依交辦規則，本次沒有偽裝 UA 去試。

## 逐賽季表格

cup_id 與年份對照取自官網 `/champions`（見〈cup_id 對應賽季〉一節）。

| cup_id | 賽季 | 來源 | 粒度 | 格式 | 取得難度 |
|---|---|---|---|---|---|
| 1–6 | 2004–2009 | 查無 | 僅團隊冠軍（官網 `/champions`） | — | 無 |
| 7 | 2011（4 月開打，曆年制） | 協會舊站 `record_of_statistic/statistic_*.pdf` 的 Wayback 存檔 | 逐場球員統計，但只存到 2 場（m48、w01） | 有文字層的 PDF（VIS 計分軟體輸出） | 高：其餘場次未被存檔，現站 404 |
| 8–12 | 2012–2016 | 查無（統計系統 CupID 12 回 500；協會舊站 Wayback 沒有對應 PDF） | 僅團隊冠軍 | — | 無 |
| 13 | 2017-18 | `statistics.ctvba.org.tw` 的 `Match.ashx` / `Statistics.ashx`（實測）；Wayback 另存有 114.35 的得分前 20 與 5 場 Match.ashx | 逐場球員統計（含姓名） | HTML 片段（ASP.NET handler），和 cup 20/21 同一格式 | 中：直連可取，但 Cloudflare 擋非瀏覽器 UA |
| 14 | 2018-19 | 同上（實測 MatchID=1） | 逐場球員統計 | 同上 | 中 |
| 15 | 2019-20 | 同上（實測 MatchID=1） | 逐場球員統計 | 同上 | 中 |
| 16 | 2020-21 | 同上（實測 MatchID=1） | 逐場球員統計 | 同上 | 中 |
| 17 | 2021-22 | ① 統計系統（實測）② 官網 `/game/1–59`、`/wgame/1–59`（實測，部分無姓名）③ 官網 gamebook PDF 的 Wayback 存檔（男女各約 14 份） | 逐場球員統計 | ① HTML ② HTML，格式同 cup 22 ③ P2 圖片 PDF | 中 |
| 18 | 2022-23 | ① 統計系統（實測 MatchID=201，女季後賽）② 官網 `/game/60–128`、`/wgame/60–121`（推估）③ `statistics.ctvba.org.tw/P2/18_*.pdf`（現站實測可取） | 逐場球員統計 | 同上；P2 為圖片 PDF | 中 |
| 19 | 2023-24 | ① 統計系統（實測 MatchID=1，且 `Statistics.ashx` 有季賽得分前 20）② 官網 `/game/129–198`、`/wgame/122–180`（實測）③ P2 PDF（實測 `19_1.pdf`） | 逐場球員統計 | 同上 | 中 |

## 各來源詳細發現

### 1. 外部統計系統：`114.35.229.141` 與 `statistics.ctvba.org.tw`

- **`http://114.35.229.141/` 整站回 403**（本次實測，2026-11-05）。協調者補充說，用手機網路（不同 IP）連一樣是 403，所以是整站拒絕存取，不是只擋特定 IP。本次沒有再額外驗證。
- **`statistics.ctvba.org.tw` 是同一套系統的網域入口**。DNS 解析到 Cloudflare IPv6（`2606:4700:3033::…`）。
  - curl 對根目錄回 403；WebFetch 對根目錄也回 403。`robots.txt` 回 404（查無 robots 規則）。
  - `Default.aspx`、`Match.aspx` 回 404。`stats_crawler.py` 註解早就記錄「自 2026-09-29 起所有 .aspx 頁面 404（_handler/*.ashx 仍正常）」，和這次看到的一致。
  - `_handler/*.ashx` 能用 WebFetch 取得（2026-11-05 實測）：
    - `https://statistics.ctvba.org.tw/_handler/Match.ashx?CupID=13&MatchID=1&SetNum=0`：男子組第 1 週 10/28，國訓中心 3:0 桃園石易，雙方都有完整球員表（攻擊、攔網、發球、接發、防守、舉球、得分，含姓名與 `Player.aspx?CupID=13&PlayerID=…` 連結）。
    - 同一端點 CupID=14、15、16、17、19、21 的 MatchID=1，以及 CupID=18 的 MatchID=201，都回傳有姓名的完整 box score。
    - CupID=12 的 MatchID=1 回 **HTTP 500**（`stats_crawler` 也是把 500 當成「查無該場」）。
    - `_handler/Statistics.ashx?CupID=13&CatID=1`、`CupID=19&CatID=1`：男女「總得分 季賽前20名排行（累計）」，欄位有隊伍、背號、球員、位置、場數、局數、攻擊得分、攔網得分、發球得分、總得分、每局平均。CatID=2 回空頁，其他 CatID 沒試。
    - `_handler/PlayerList.ashx?CupID=13&TeamID=1`：14 名球員的下拉選單（背號＋姓名＋PlayerID）。
  - **判定兩者是同一系統的依據**：路徑、參數完全相同；`Statistics.ashx?CupID=13&CatID=1` 的內容（黃世豪 437 分、吳宗軒 420 分…）和 Wayback 存的 `114.35.229.141` 同一頁（2021-02-03 快照）逐字相同。來源是不是同一台主機，本次無法證實。
  - **P2 比賽結果表 PDF**：`https://statistics.ctvba.org.tw/P2/19_1.pdf` 現站可取（2026-11-05 實測）。內容是 2023-10-14 臺北Conti 對 桃園臺產，「企業十九年男女甲級排球聯賽」，有雙方名單、背號、每局得分。PDF 內文是圖片（pdftotext 抽不到字），而且技術統計只列每項前 3 名，**不能取代 box score**，但可以拿來對照背號與姓名。`P2/18_1.pdf` 也可取；`P2/13_1.pdf`、`16_1.pdf`、`17_1.pdf` 回 404，所以 P2 只從 cup 18 起才有。
- **Wayback 對 `114.35.229.141` 的存檔**（CDX：`https://web.archive.org/cdx/search/cdx?url=114.35.229.141&matchType=prefix&output=json&collapse=urlkey`，共 323 筆、非靜態資源 209 筆）：
  - `Default.aspx` 的 CupID 13–21 都有存，但多半只是殼頁，內容靠 `.ashx` 動態載入，殼頁本身沒有數據。`Default.aspx?CupID=16`（2020-11-29）有當時的戰績表。
  - 有存到的 handler：`Statistics.ashx` CatID=1 的 CupID 13、14、15、17（得分前 20）；`Match.ashx` 的 CupID 13 有 5 場、CupID 17 有 8 場；`Player.ashx` 的 CupID 17 有 6 人；`PlayerList.ashx` 的 CupID 17 有 5 隊。
  - `P2/18_*.pdf` 有 103 份（cup 18）。`statistics.ctvba.org.tw/P2/18_*.pdf` 另有 42 份（2025-01 快照）。
  - `Award.aspx?CupID=13`（2025-06 快照）**不是個人獎項頁**，內容是賽程比分；而且雖然標題寫「企業十三年」，比賽日期卻是 2023-10 到 2024-03（cup 19）。這是系統本身的錯置。
  - 結論：Wayback 對這套系統只存到零星片段，**現站 handler 才是主要來源**，Wayback 只能拿來交叉核對。

### 2. 官網 `tvl.ctvba.org.tw`

- **存取方式**：curl 預設 UA 對任何頁面（包括當季 `/game/394` 和 `robots.txt`）都回 Cloudflare 403「Attention Required!」，WebFetch 則可取得。`robots.txt` 允許賽程、球隊、球員、戰績等頁面，建議 crawl-delay 5 秒（經 WebFetch 摘要，2026-11-05）。
- **舊賽季賽程頁**：`/schedule/{民國年}`、`/wschedule/{民國年}`。
  - `/schedule/110`：企業17年男子組，`/game/1–59`（59 場，含明星賽與季後賽）。
  - `/schedule/111`：企業18年男子組，`/game/60–128`（69 場）。
  - `/schedule/112`：企業19年男子組，`/game/129–198`（70 場）。
  - `/wschedule/110`：企業17年女子組，`/wgame/1–59`。
  - `/wschedule/112`：企業19年女子組，`/wgame/122–180`。
  - `/wschedule/111` 沒打開；依前後區間推估為 `/wgame/60–121`。
  - `/schedule/109` 會落回當季內容，所以**官網最早只到企業17年**。
  - 和 DB 互相印證：cup 20 男子組從 game_id 199、女子組從 181 開始。
- **舊比賽頁 box score**：欄位是「球員、位置、攻擊、攻擊%、攔網、攔網%、發球、發球%、接發球、接發球%、防守、防守%、舉球、舉球%、得分」，和 cup 22 一樣（2026-11-05 實測）。
  - `/game/1`：2021-10-30 雲林美津濃 0:3 桃園臺產，頁題「110年-企業17年」，每隊 14 列。
  - `/wgame/1`：2021-10-30 愛山林 0:3 臺北鯨華，愛山林多數列沒有姓名。
  - `/game/100`：2023-02-05，企業18年。
  - `/game/120`：2023-03-11，企業18年季後賽，28 列中 19 列無姓名。
  - `/game/198`：2024-03-10，企業19年，28 列中 13 列無姓名。
  - `/wgame/180`：2024-04-01，企業19年總冠軍賽，臺北鯨華 14 列中 9 列只有背號，連結是 `/wplayer/`、沒有 ID。
  - **數字完整，但姓名只對得上現役名單**，和既有記憶「官網歷史賽事名單只保留當季」一致。
- **Wayback 的官網比賽頁快照可以補回姓名**（CDX：`url=tvl.ctvba.org.tw/game/&matchType=prefix`、`url=tvl.ctvba.org.tw/wgame/&matchType=prefix`）：
  - 實測 3 頁，姓名都完整：`/game/100` 2023-06-10 快照（布萊恩、劉鴻杰…）、`/game/150` 2024-10-13 快照（施宇祥、林謙…）、`/game/30` 2022-07-02 快照（黃韋程、高偉誠…）。
  - 依時間戳推估的覆蓋情形（未逐頁驗證）：

    | 組別 | 賽季 | 有快照 / 總場數 | 備註 |
    |---|---|---|---|
    | 男子 | cup 17 | 36 / 59 | 其中 18 場有賽季結束後的快照 |
    | 男子 | cup 18 | 69 / 69 | |
    | 男子 | cup 19 | 70 / 70 | |
    | 女子 | cup 17 | 0 / 59 | |
    | 女子 | cup 18 | 60 / 62 | 快照多拍在 2022-10 到 12 月，很可能早於比賽日，是空頁 |
    | 女子 | cup 19 | 59 / 59 | |

  - 有了統計系統（第 1 節），這條路主要用來交叉核對。
- **官網 gamebook PDF**：`/download/gamebook/mens_gameN.pdf`、`/download/wgamebook/womens_gameN.pdf`，Wayback 男女各存約 14 份。
  - 實測 `mens_game1.pdf`（2021-11-13 快照）是企業十七年第 1 週的 P2 比賽結果表，圖片 PDF。
  - 現站 `mens_game100.pdf` 回 404。
- **其他官網頁面**：
  - `/stat-player`、`/stat-ranking` 現站與 2025-04 快照都是 PHP 錯誤「Table 'tvl.p_season' doesn't exist」，沒有數據。
  - 選單上的「統計排行」都是 `#`。
  - `/player/{id}` 頁註明「紀錄取自企排17年起至最新一季」。
- **2017–2021 舊 WordPress（SportsPress）站**：Wayback 存有 452 個 `/event/…` 頁。
  - 實測 `中國人纖-vs-台電女排-2`（2019-01-09 快照）只有名單，「得分」欄全是 0，不能當數據來源。
  - `男子得分排行` 頁（2017-11-01 快照，標題「106企業十三年」）是季初的累計得分排行，只是某個時間點的片段。

### 3. 中華民國排球協會（ctvba.org.tw）

- **現站**（`https://www.ctvba.org.tw/`，2026-11-05）：選單沒有企業聯賽、技術統計、年報或獎項專區。
  - 「國內賽事」列表（`/article/front/inGame`）共 13 頁，第 1 頁最舊是 2025/11/14，沒有企業聯賽相關文章。
  - 「資訊系統」連到 `https://www.sportsstream.idv.tw/CTVBA`（沒打開）。
- **舊站 Wayback**：
  - `http://www.ctvba.org.tw/NewPage/enterprise_game/record_of_statistic.htm`（2011-09-29 快照）是「100年度企業七年甲級男女排球聯賽」的賽程表，每場附「攻守統計」PDF 下載（`record_of_statistic/statistic_m01…m30+、w01…w15+`，2011/04/16 起）。
  - Wayback 只存到 `statistic_m48.pdf`（2011-11-11 快照）和 `statistic_w01.pdf`（2012-12-24 快照）兩份。
  - 實測 m48：「TVL7-企業七年排球聯賽(男) 球隊攻守統計」，2011/10/30 台電 3:0 貫一，有文字層。每位球員有發球、攻擊、攔網、舉球、防守、接發球的得、失、總，比現行格式更細。
  - 現站同路徑回 404。
  - 同目錄另有 `all_players.htm`（2007）、`result_of_team.htm`（2008）、`schedule.htm`（2008）、`Result.doc`（2012），內容未檢視。
- **Wayback 上 ctvba.org.tw 網域共 890 份 PDF**：多數在 `/files/articleAttr/`（2016 年後）和 `/download/`（2009–2014）。以「企、聯賽、統計、獎」篩檔名，除上述兩份外沒有命中。不過 `/download/2014…pdf` 這類以時間戳命名的檔案沒逐一打開，**可能漏掉**。

### 4. cup_id 對應賽季

- **開辦年**：官網 `/about` 寫「企業排球聯賽於93年草創」，男子組 2004 年 9 月成立、女子組 2009 年 11 月成立（WebFetch 摘要，2026-11-05）。
- **屆次對年份**：官網 `/champions` 依序列出：

  | 屆次 | 年份 |
  |---|---|
  | 企業元年 | 2004（男子冠軍註明「例行賽排名」） |
  | 企業2年 | 2005 |
  | 企業3年 | 2006 |
  | 企業4年 | 2007 |
  | 企業5年 | 2008 |
  | 企業6年 | 2009 |
  | 企業7年 | 2011 |
  | 企業8年 | 2012 |
  | … | … |
  | 企業21年 | 2025 |

  **沒有 2010**。女子組從企業6年起才有冠軍。
- **與本 repo 的 cup_id 一致**：DB `seasons` 表 cup 20 = 2024-25、cup 21 = 2025-26；官網比賽頁標題「110年-企業17年」「111年-企業18年」「112年-企業19年」；協會舊站「100年度企業七年」；舊 WP 站「106企業十三年」。這些都符合 **cup N = 企業N年**，cup 7 起民國年 = N + 93。
- **賽制時間**：企業七年 2011/04/16 開打，是曆年制。cup 13 起（2017-10-28 開打）是跨年制（10 月到隔年 3 月）。中間哪一屆改制，查無。

### 5. 二手來源（只當線索）

- WebSearch「企業排球聯賽 得分王 歷屆 個人獎項」只找到單季新聞（udn 企排21年頒獎：https://udn.com/news/story/7005/9437049 ），**查無歷屆得分王完整名單**。
- 有一篇 2007 年企業男排得分因素的期刊論文（airitilibrary），只有團隊層級統計。
- 英文維基「Top Volleyball League」/「Enterprise Volleyball League」只列冠軍。

## 未解問題

1. **Cloudflare 是否放行爬蟲**：`statistics.ctvba.org.tw` 和官網都擋 curl 預設 UA。`stats_crawler` 用的 `EXT_HEADERS` 是瀏覽器 UA，本次依規則沒有用它測試。要不要用、能不能用，請使用者決定。
2. **MatchID 編碼與場次總數**：`Match.aspx` 404，只能逐號探測（`stats_crawler._scan_match_ids` 已實作）。女子組的編號每季不一樣：cup 18 女季後賽是 201，cup 21 女子組從 1 起。cup 13–19 各季實際場數，還有有沒有缺場或資料壞掉的 500，都還沒盤點。
3. **球員身分對齊**：統計系統的 PlayerID 和 TeamID 每季重編（`EXT_TEAM_MAPS` 目前只有 cup 20、21）。cup 13–19 要新增隊伍對照，跨季同名球員也要決定合併規則。外籍球員在不同來源的寫法不同（例：「B.BAGUNAS」對「布萊恩」）。
4. **統計口徑**：cup 7 PDF 的欄位比現行細（有失分、上場時間）。cup 13–19 handler 的欄位看起來和 cup 20、21 相同，但沒有逐欄核對。
5. **cup 8–12 與 cup 1–6**：協會舊站 `/download/` 底下以時間戳命名的 PDF、`Result.doc`、`all_players.htm` 都還沒打開。要找個人獎項（得分王等）可能得翻當年新聞稿，或向排協詢問。
6. **2010 年空缺與企業6年的實際起訖**：`/champions` 標企業6年為 2009、企業7年為 2011，和「民國年 = N + 93」在 cup 6 對不上（應為 99 年 = 2010），原因查無。
7. **官網 `/wschedule/111` 的實際區間**沒有打開確認。
