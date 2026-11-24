# 國外聯賽數據統計頁盤點：可加到本站的功能與統計量

> 查證日期：2026-11-24。對象是三個官方來源：日本 SV League（`svleague.jp`）、義大利甲級（男子 `legavolley.it`、女子 `legavolleyfemminile.it`）、Volleyball World（`en.volleyballworld.com`，以 VNL 2026 為樣本）。
> 做法：只看官網自己的頁面，用 curl 抓原始 HTML 後檢視表頭與欄位，每站約 20 到 40 個請求，沒有用瀏覽器渲染。
> 下文「實測」指在抓回的 HTML 裡實際看到；「推論」指由欄位名或數字反推、官網沒有明寫；「未找到」只代表已查範圍內沒有，不代表官網沒有。
> 複核：寫完後由另一個沒看過蒐集過程的 agent 拿本文對照原始頁面與本站程式碼，核對 25 個點，23 個相符、2 個不符（SV League 各局時間所在的報表、本站跨季累計的現況），兩處都已修正。
> 更正（2026-11-24，拆票前盤點時發現）：候選第 1 項「戰績依比數拆場次」本站早就有了，戰績表已顯示六種比數各幾場。原稿寫成沒有，下文已改；候選表「本站現況」欄只有一部分經過複核，見未確認事項第 9 點。
> 各站的完整筆記與抓回的原始頁面在 `~/work/tvl-stats-research-notes/`（`sv-league.md`、`serie-a.md`、`volleyball-world.md`、`raw/`），不在 repo 裡。

## 結論摘要

1. **本站的資料上限決定了大部分答案。** 逐場球員統計只有攻擊與發球的次數和得分、攔網得分、接發球／防守／舉球的次數和到位數（`sql/schema.sql:76-96`），官網 box score 的表頭也沒有任何失誤欄（`src/etl/official_box_score.py:13-14`）。國外三站都有的「失誤」與由它算出的效率值，本站算不出來。
2. **不需要新資料就能做的有 17 項**，其中四項最划算：賽後兩隊數據對照、單場各項前三名、每局平均項目、球員逐場紀錄與逐季累計表。前兩項在兩個站都看得到。戰績依比數拆場次是三個站的共同做法，本站已有。
3. **里程碑提示與擴充紀錄榜是義甲男子站獨有、而且本站做得到的東西**：賽前列出「某球員距離 300 分還差幾分」，以及單場總分最高的比賽、單局最高比分這類比賽層級的紀錄。
4. **逐球紀錄是 SV League 與義甲男子站最大的差異**，兩站都把每一球的動作公開在單場頁。本站沒有逐球資料，這一塊連同輪轉、破發得分、逐局個人數據都做不到。
5. **本站已有的門檻設計與 SV League 同型**：SV League 的攻擊決定率門檻是「出場局數 × 3 次攻擊」，本站是「係數 × 球隊已打局數」（ADR 0011、0012）。三站都沒有本站的 PR 值、同位置分佈與賽前勝率。

## 本站現況（比較基準）

依 2026-11-24 的程式碼盤點：

- **頁面**：首頁、賽事列表與單場頁、戰績、球隊列表與球隊頁、球員列表與球員頁、排行榜、歷史紀錄（單季／累計／單場）、通知設定。
- **排行榜**：計數榜 7 項、比率榜 7 項（含場均得分、每局攔網），只算例行賽（`web/src/lib/data/types.d.ts:399-401`），比率榜有門檻並在頁面上寫出門檻描述（`types.d.ts:424-426`）。
- **球員頁**：本季數字、十項比率、八項同位置 PR 值、同位置分佈、逐場趨勢圖。
- **球隊頁**：戰績、得分來源組成（含推算的對手失誤得分）、六項球隊指標與組內名次、逐場趨勢、週登錄。
- **單場頁**：賽前勝率、賽前對戰比較、本季交手、逐局比分、box score。
- **戰績**：勝敗、勝率、積分、六種比數各幾場、局數比、得失分比、近五場。
- **沒有的**：搜尋、自訂排序、依位置篩選排行榜、球員對球員比較、獨立的對戰紀錄頁、球員的逐季與跨季累計表（跨季數字只出現在歷史紀錄的累計紀錄前十名）、主客場或場館拆分、資料下載。
- **原始資料沒有的**：任何失誤、接發球的分級（只有「到位」一級）、逐局個人數據、逐球與輪轉、比賽時長、觀眾數、單場最佳球員。

## 候選清單

「規模」是依本站現有匯出器與頁面結構的粗估（小＝改一個匯出器加一個元件；中＝新的 JSON 合約或新頁面），屬推論，開票前要重估。

### 一、現有資料就做得出來

| # | 候選 | 誰在做 | 本站現況 | 規模 |
|---|---|---|---|---|
| 1 | 戰績依比數拆場次（3-0、3-1、3-2、2-3、1-3、0-3 各幾場） | Volleyball World、義甲男子、義甲女子（實測）；SV League 沒有這組欄位 | 已有：戰績表顯示六種比數各幾場，窄螢幕收在展開列 | 不用做 |
| 2 | 戰績下方寫明同分時的排序規則 | 義甲女子（勝場 → 局數比 → 得失分比，實測）；SV League 註明依勝率排序 | 未確認頁面有沒有寫 | 小 |
| 3 | 戰績可看「截至第 N 輪」 | 義甲女子（Filtra fino a giornata，實測） | 沒有 | 中 |
| 4 | 賽後兩隊數據對照（攻擊、攔網、發球、對手失誤、總分，加上防守、接發球、舉球） | Volleyball World 單場頁、SV League 單場頁的チーム記録（實測） | 只有賽前對戰比較；box score 沒有球隊合計對照 | 小 |
| 5 | 單場各項前三名；當輪各項前五名 | 義甲女子單場頁（Top scorers／servers／blockers 與 PRESTAZIONI TOP 5 DELLA GIORNATA，實測）；Volleyball World 單場頁的 Best Scorers | 沒有 | 小 |
| 6 | 每局平均項目擴充：每局得分、每局攻擊得分、每局發球得分、每局接發球到位 | 義甲男子（Punti/Set、Attacchi Vincenti/Set、Battute Vincenti/Set、Ricezioni Perfette/Set、Muri Vincenti/Set）；SV League（每局攻擊決定數、每局攔網決定數）（實測） | 只有場均得分與每局攔網 | 小 |
| 7 | 得分榜同列顯示攻擊／攔網／發球分項 | Volleyball World（Points、A Pts、B Pts、S Pts，實測）；SV League 総得点榜的アタック／ブロック／サーブ三欄（欄位定義為推論） | 計數榜每列只有一個值 | 小 |
| 8 | 球員逐場紀錄表（每場對手、日期、各項數字） | Volleyball World 球員頁（實測）；義甲男子球員頁有 Statistiche per Giornata 分頁（只確認分頁存在） | 只有逐場趨勢圖，沒有表 | 小到中 |
| 9 | 球員逐季與跨季累計表 | 義甲男子（Totali／A1／A2／A3，含賽季數、場數、勝場、得分）；SV League 球員頁逐季列與通算排行（`season=all`）（實測） | 球員頁沒有逐季與跨季累計表；歷史紀錄頁的累計紀錄只列四項得分的前十名 | 中 |
| 10 | 排行榜依位置篩選 | 義甲男子（Ruolo Centrale、Ruolo Schiacciatore 兩個榜；個人統計頁有位置下拉，實測） | 沒有；位置資料已有，但部分球員位置是空值 | 中 |
| 11 | 排行榜可選範圍：季後賽、單一輪次 | SV League（ラウンド、節）；義甲男子（Campionati、Giornata）；Volleyball World 的 Finals 另開一頁（實測） | 只算例行賽 | 中 |
| 12 | 全聯盟球隊數據總表與球隊排行 | SV League 球隊排行；義甲男子球隊統計總表與球隊排行（實測）；Volleyball World 未找到整季球隊統計 | 六項球隊指標只在各自的球隊頁 | 小到中 |
| 13 | 紀錄榜擴充到比賽與球隊：總分最高與最低的比賽、單局最高比分、決勝局最高比分、單場球隊最多發球得分與攔網、交手最多次的對戰 | 義甲男子 Curiosità Statistiche，分比賽／球隊／球員／教練四類並可選賽季（項目名稱實測，紀錄內容未開啟；「交手最多次」是對 `I match più giocati` 這個標題的推論） | 只有球員的四項得分紀錄 | 中 |
| 14 | 里程碑提示：賽前列出快要達到累計整數關卡的球員 | 義甲男子單場頁的 Traguardi importanti（例：300 punti、25 muri punto，實測） | 沒有；可接在賽前頁，也可接推播 | 中 |
| 15 | 兩隊歷年對戰紀錄，可選賽事範圍 | 義甲男子 Precedenti ed Ex（只確認篩選器，結果列表未取得） | 只有賽前頁的本季交手文字 | 中 |
| 16 | 球員搜尋 | SV League（依姓名、球隊、賽事）；義甲男子（姓名開頭或部分字串、字母索引）（實測） | 沒有 | 小到中 |
| 17 | 球隊頁的歷年名次，以及隊史得分榜與出賽榜 | 義甲男子球隊頁（Piazzamenti Stagioni Precedenti、Top Scorer、Top Presenze，實測） | 沒有 | 小到中 |
| 18 | 歷年冠軍 | 義甲男子 Albo d'oro（實測） | 沒有；官網 `/champions` 有歷年團隊冠軍（見 `tvl-pre-cup20-data-sources.md`），要另外抓 | 小到中 |

兩點補充（推論）：

- 第 14 項義甲是在逐球頁上呈現，但計算本身只需要每場累計，本站的逐場合計就夠。
- 第 6 項的「每局」分母，本站只能用球隊在該球員出賽場次的局數（現有每局攔網的算法），不是球員實際上場的局數；SV League 與義甲用的是個人出場局數。

### 二、要先補資料才做得出來

| 項目 | 誰在做 | 缺什麼 |
|---|---|---|
| 攻擊、發球、接發球的失誤，以及效率值 | Volleyball World 每個排行都有 Errors 欄（實測）；義甲男子球隊統計的 Effic. 由數字反推為（得分 − 失誤）÷ 次數（推論）；SV League 有失点與サーブ失点 | 官網 box score 沒有球員失誤欄，只有球隊層級的對方失誤合計 |
| 接發球分級的比率 | SV League：（優 × 100 ＋ 良 × 50）÷ 受數（公式實測，寫在 B 帳票註腳）；義甲男子單場表有 Pos%、Prf%，球隊統計總表有 Neg.、Prf. %（實測） | 本站接發球只有「到位」一級 |
| 發球效果率 | SV League：（ACE × 100 ＋ 效果 × 25 − 失誤 × 25）÷ 打數（公式實測） | 沒有發球效果與發球失誤 |
| 後排攻擊拆分 | SV League（前排、後排攻擊決定率與決定數，實測） | 沒有前後排標記 |
| 單場最佳球員與累計次數 | 義甲男子（Nomination MVP、球員頁 MVP 次數）、義甲女子（MVP，實測） | 沒有這項資料 |
| 各局時間、比賽總時長、觀眾數 | SV League、義甲男子、義甲女子（實測）；義甲男子另有票房 | matches 表沒有這些欄位 |
| 逐局的個人數據 | Volleyball World 單場頁可切 All Sets／Set 1–4（實測） | 只有整場合計 |
| 逐球紀錄 | SV League 單場頁 Live Score；義甲男子 Punto a Punto（實測） | 沒有逐球資料 |
| 破發得分（Break Point） | 義甲男子的球隊與球員排行、單場表都有 BP 欄（實測） | 需要知道每一分是誰發球（推論） |
| 先發與輪轉、每局上場記號 | 義甲男子 Rotazione iniziale；SV League 帳票的出場記號（實測） | 沒有 |

這一組要不要做，取決於官網或其他來源拿不拿得到資料，本次沒有查。

### 三、本站已有，國外做法可對照

- **上榜門檻**：SV League 的共通規定是出賽場數達球隊總場數 70%、出賽局數達球隊總局數 67%；攻擊決定率另要求打數達出場局數 × 3（實測）。義甲男子排行頁沒看到最低門檻，只有取前 20 到 100 名的選項。Volleyball World 的排行約 330 列，看起來是全部球員（推論）。本站的浮動門檻與 SV League 同型，差別在本站沒有個人出場局數，用球隊已打局數代替。
- **門檻與公式寫在頁面上**：SV League 把規定寫在排行頁、公式寫在單場報表註腳。本站排行榜已寫出門檻描述。
- **近五場**：義甲女子戰績有 Ultimi 5 match，本站已有。
- **局數比與得失分比**：四個站都有，本站已有。
- **本站有、三站沒看到的**：同位置 PR 值與分佈、量率散點、賽前勝率。三站也都沒找到球員對球員比較、圖表與資料下載（已查範圍見各站筆記）。

## 建議順序

1. **先做第 4、5 項**：都是小改動，兩個站都有，而且直接補在既有的單場頁上。
2. **再做第 6、7、8 項**：排行榜與球員頁的欄位擴充，不需要新頁面。
3. **第 9、13、14 項值得各開一張 spec**：球員跨季累計、比賽與球隊紀錄、里程碑提示，都要新的 JSON 合約，也都牽涉回填賽季（cup 16–19）要不要算進去。
4. **第二組先不排**：要先確認資料來源。

## 各站盤點摘要

### SV League（svleague.jp）

- **個人排行**（`https://www.svleague.jp/record/sv/women/?type=player_ranking`）：女子 18 個項目，包含 `総得点`、`アタック決定率`、前排與後排攻擊決定率、每局攻擊決定數、每局攔網決定數、`サーブ効果率`、接發球的比率榜（`サーブレシーブ`）、`出場試合数`、`出場セット数`。可選賽季（含通算）、ラウンド、節。
- **門檻**（`https://www.svleague.jp/record/sv/women/?type=player_ranking&season=758&round=410&category=1`）：見上一節。
- **球隊排行**（`https://www.svleague.jp/record/sv/women/?type=team_ranking`）與球隊數據頁（`https://www.svleague.jp/team/nec/record/`）：球隊總得分註明「※相手起因の得点は除く」（不含對手失誤造成的得分）。
- **戰績**（`https://www.svleague.jp/match/sv/women/standings/?s=2026`）：順位、試合数、勝、負、ポイント、勝率、得セット、失セット、セット率；註明依勝率排序。
- **單場頁**（`https://www.svleague.jp/match/detail/37428/`）：分頁有試合情報、Live Score、スタッツ、コメント。個人記錄有攻擊（打數、成功、失敗、決定率、後排攻擊）、攔網、發球（打數、ACE、失敗、效果、效果率）、接發球（受數、優、良、失敗、比率）。
- **逐球紀錄**（`https://www.svleague.jp/match/detail/37428/?tab=live`）：依局分段，每筆有球隊、動作、球員、當下比分。只看了一場。
- **官方報表**：B 帳票（`https://www.svleague.jp/match/form/?id=37428&sheet=b`）有比賽總時間、入場者數，註腳有三個公式；各局時間在 A 帳票（`https://www.svleague.jp/match/form/?id=37428&sheet=a`）。
- **球員頁**（`https://www.svleague.jp/player/detail/?id=3643`）：逐季累計表；逐場紀錄與圖表未找到。
- **未找到**：單場最佳球員、球員比較、對戰紀錄、主客場篩選、資料下載、圖表。

### 義大利甲級（legavolley.it、legavolleyfemminile.it）

- **男子個人排行**（`https://www.legavolley.it/rendimento/?Tipo=2&Classifica=2.1`）：15 個選項，計數與每局平均成對出現，另有三個加權平均榜（Media Ponderata，算法未確認）與兩個依位置的榜。可選級別、賽事範圍、輪次、取前幾名。
- **男子球隊統計總表**（`https://www.legavolley.it/statistiche/?TipoStat=1.4`）：得分與 BP、發球（次數、Ace、失誤、每局 Ace、效率）、接發球（次數、失誤、負向、完美、完美率、效率）、攻擊（次數、失誤、被攔、得分、得分率、效率）、攔網。
- **男子戰績**（`https://www.legavolley.it/classifica/?IdCampionato=999`）：積分、場次勝敗、六種比數各幾場、局數、得失分、兩個比值。同分排序規則頁面沒寫。
- **女子戰績**（`https://www.legavolleyfemminile.it/classifica/`）：同樣的欄位加上近五場，可選截至第幾輪，頁面寫明同分排序規則。
- **男子單場頁**（`https://www.legavolley.it/match/?idGara=38296&lang=`）：觀眾數、票房、場館、裁判、MVP 提名、各局時間與分段比分、起始輪轉、球員表。球員表的逐格對應沒有完全核對。
- **男子逐球頁**（`https://ww2.legavolley.it/tabellini_online/gara_38296.html`）：每行一個動作（發球、ACE、發球失誤、接發球失誤、攻擊得分、攻擊被攔、攻擊失誤、攔網得分），上方有里程碑提示。
- **女子單場頁**（`https://www.legavolleyfemminile.it/10a_giornata_andata_22_11_2026___lvf_a1_fineco___tabellini_785741/`）：文章形式，有觀眾數、各局時間、MVP、兩隊發球／攻擊／攔網摘要、單場各項前三名、當輪各項前五名。
- **男子球員頁**（`https://www.legavolley.it/player/REG-FER-03`）：歷年累計表依聯賽級別分欄，另有逐輪與逐年統計、各隊得分與出賽、MVP、歷年經歷、獎項等分頁。歷年累計表的總得分與分項加總對不上（1906 對 1933），欄位定義未確認。
- **男子球隊頁**（`https://www.legavolley.it/team/6858`）：隊史得分榜、出賽榜、名單、冠軍紀錄、歷年名次。
- **其他**：紀錄榜（`https://www.legavolley.it/pillole/`）、對戰紀錄（`https://www.legavolley.it/confronti/`）、歷年冠軍（`https://www.legavolley.it/albodoro/`）、搜尋（`https://www.legavolley.it/ricerca/`）。賽季下拉可回到 1945/46。頁面底部註明統計在賽後隔天 19:00 起才算經聯盟核定。
- **未取得**：女子站的統計中心（球員、球隊、表現榜、比較、紀錄）由第三方元件載入，靜態 HTML 沒有表格，欄位全部未確認。

### Volleyball World（VNL 2026）

- **個人排行**（`https://en.volleyballworld.com/volleyball/competitions/volleyball-nations-league/statistics/men/best-scorers/`，另有 best-attackers、best-blockers、best-servers、best-setters、best-receivers、best-diggers）：七個榜。得分榜是總分加三個分項；其餘各榜是「得分或成功、失誤、其他嘗試、場均、`Success %` 或 `Efficiency %`、總數」同一套欄位。以首列驗算，三項相加等於總數，`Success %` 等於得分除以總數（推論）。攔網榜的效率是負值，反推為（攔網得分 − 失誤）÷ 總數（推論）。
- **門檻與公式**：未找到。
- **戰績**（`https://en.volleyballworld.com/volleyball/competitions/volleyball-nations-league/standings/men`）：場次勝敗、六種比數各幾場、積分、局數與局數比、得失分與得失分比。
- **單場兩隊對照**（`https://en.volleyballworld.com/volleyball/competitions/volleyball-nations-league/schedule/26542/_libraries/live/_volley-match-statistics-by-team`）：Attack、Block、Serve、Opponent Error、Total，另有 Dig、Reception、Set，以及兩隊各自的得分前三名。
- **單場球員統計**（同路徑的 `_volley-match-statistics-by-player`）：可切 All Sets 與各局，分 Scoring、Attack、Block、Serve、Reception、Dig、Set 七個分頁，球員列帶位置。
- **球員頁**（`https://en.volleyballworld.com/volleyball/competitions/volleyball-nations-league/players/114422`）：賽事累計卡片（總得分、場均、攻擊／攔網／發球的得分、比率與場均）與逐場表。
- **未找到**：整季球隊統計、逐球紀錄、輪轉、各局時間、觀眾數、裁判、報表、對戰紀錄、球員比較。`World Ranking` 表由 JavaScript 載入，資料端點未確認。

## 未確認事項

1. **沒有用瀏覽器渲染**：三站都只看靜態 HTML。排序、篩選的互動行為，以及由 JavaScript 載入的內容（義甲女子統計中心、Volleyball World 的 `World Ranking` 與球隊賽事列表、義甲男子的即時比分）都沒有看到。
2. **否定結論的範圍有限**：「未找到」的項目只查過各站筆記列出的頁面。
3. **公式多半是反推**：Volleyball World 的 `Success %` 與義甲男子的效率值都是用列上的數字驗算出來的，官網沒有定義頁；兩站的縮寫（SE、REB、ABS、Media Ponderata、攻擊表的符號欄）定義未確認。
4. **各站規程沒有查**：SV League 的積分計算與同勝率排序、義甲男子的同分排序、Volleyball World 的同分排序與 `World Ranking` 計分，都沒有查官方規程文件。
5. **樣本小**：逐球紀錄各只看一場，不確定是否每場都有。SV League 的男子球隊排行、男子戰績與舊賽季站（`archive.svleague.jp`）沒有抓。
6. **Volleyball World 的場均分母不一致**：舉球與發球榜反推約 15 場，戰績是 12 場，不確定是否含 `Finals` 階段。
7. **SV League 排行頁標示「2026/06/24現在」**，與賽事頁已到 11 月的狀態不一致，資料新鮮度未確認。
8. **第二組的資料來源沒有查**：官網或其他來源有沒有球員失誤、單場最佳球員、比賽時長與觀眾數，本次不在範圍內。
9. **候選表的「本站現況」欄只有一部分核對過程式碼**：第 1、2、4、5、7、9、10、15、16 項核對過，其中第 2 項確認戰績頁沒有寫同分排序規則；第 3、6、8、11、12、13、14、17、18 項只依一次盤點，開票前要再確認。回填賽季（cup 16–19）實際進到正式資料的範圍也未確認。
