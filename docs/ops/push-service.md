# 發送服務維運：部署、驗證與日常查詢

發送服務替本站把推播（賽果摘要、今日賽事）送到球迷裝置。程式在 repo 的 `push/`，跑在 Cloudflare 的免費方案，狀態存在 D1。網站本身是 GitHub Pages 上的靜態站（`https://vstat.tw`），發送服務是本站唯一在執行期運作的部分。介面、入口、通知內容與資料表的細節在 [`push/README.md`](../../push/README.md)；為什麼狀態存 D1、不重發怎麼做，見 `docs/adr/0014-push-service-state-in-d1-claim-by-conditional-write.md`；網域為什麼是 `vstat.tw`，見 `docs/adr/0013-site-moves-to-own-domain.md`。

這份文件寫流程：做什麼、什麼時候做、預期看到什麼。指令以 `push/README.md` 為準，這裡只指到它的章節，不重抄；README 沒有的步驟才在這裡寫出完整命令。

## 兩個環境

| | 預覽環境 | 正式環境 |
|---|---|---|
| 用途 | 第 4 期驗收（#252）用，另一個發送服務、另一個 D1、另一組 VAPID 金鑰 | 上線用 |
| 服務名稱 | `tvl-push-preview` | `tvl-push` |
| 網址 | `https://tvl-push-preview.tvl-push-proto.workers.dev` | `https://push.vstat.tw`（綁自訂網域；`workers.dev` 網址沒開，連上去回 Cloudflare 的 1042） |
| D1 | `tvl-push-preview` | `tvl-push` |
| 對應的網站 | 預覽站 `https://tommyyi0317.github.io/tvl-shell-proto` | `https://vstat.tw` |
| 設定在哪 | `push/wrangler.jsonc` 的 `env.preview` | `push/wrangler.jsonc` 的頂層 |

兩個環境程式相同，差別全是設定（見 README「設定項目」）。D1 識別碼與 VAPID 公鑰已填在 `push/wrangler.jsonc`，本文不重抄。

目前狀態：兩個環境已於 2026-11-18 建好並驗過，回報在 #251（設定 PR 是 #280）。平常改了程式只要做「部署」；本文其餘部分在這幾種時候讀：

- 要從零重建環境（換 Cloudflare 帳號、環境被刪）。
- 要重新部署並確認沒壞。
- 第 4 期驗收（#252）要用預覽環境。
- 要查發送紀錄，或驗收完要清預覽環境的訂閱。

## 前置

1. Cloudflare 帳號，且 `vstat.tw` 已是該帳號下的 zone（ADR 0013）。正式環境要綁 `push.vstat.tw`，這個名稱不能已經有 CNAME 紀錄。
2. Node 22.12 以上（wrangler 的要求，README「本機」）。
3. 在 `push/` 底下安裝依賴：`npm ci`（README「本機」）。`push/` 有自己的 `package.json`，不與 `web/` 共用。之後的 wrangler 都用 `node_modules/.bin/wrangler`，版本由 `package.json` 鎖。
4. wrangler 登入 Cloudflare：`node_modules/.bin/wrangler login`，照它開的瀏覽器頁面授權。登入後 `node_modules/.bin/wrangler d1 list` 能列出資料庫就代表登入有效。
5. 手上有 `push/wrangler.jsonc` 與 `push/migrations/`。部署都從已合進 main 的 commit 做，不從未合併的分支部署。

`d1 execute`、`d1 migrations apply` 的 `--preview` 旗標指的是 D1 的 preview 資料庫，與預覽環境無關，不要用（README「部署」）。

## 建環境

從零重建才做。已經建好的環境跳到「部署」。預覽環境與正式環境各做一遍，**兩個環境的資源與金鑰互相獨立，不共用**。照 README「部署」的步驟，預覽環境在 `push/wrangler.jsonc` 的 `env.preview` 下操作並加 `--env preview`，正式環境操作頂層設定、不加。

先後順序與下面三小節的編號不同，照這個走（#251 實際的順序）：

1. 建 D1、套 migration（第 1 節）。
2. 產生 VAPID 金鑰，私鑰先暫存在筆電上（第 3 節第 1 步）。
3. 把 D1 識別碼與公鑰填進 `push/wrangler.jsonc`，開 PR 合進 main（第 2 節）。
4. 第一次部署（「部署」）。
5. 把私鑰存成 secret，再備份（第 3 節第 2、3 步）。
6. 做「部署後驗證」。

### 1. D1

做什麼：建資料庫、把識別碼填進設定、套用 `push/migrations/`。

- 指令：README「部署」，預覽環境第 1、2 步，正式環境第 1、2 步。資料庫名稱是上表的 D1 名稱。
- 識別碼填進哪：預覽環境填 `env.preview.d1_databases` 的 `database_id`，正式環境填頂層 `d1_databases` 的 `database_id`。
- 預期：建立時回傳一個識別碼；套 migration 時列出 `0001_init.sql` 並套用成功。
- 驗證：見「部署後驗證」的步驟 1、2。

### 2. 設定項目

`push/wrangler.jsonc` 要有的項目與兩個環境的差別，見 README「設定項目」的表。重建時要確認三件事：

- 預覽環境：允許的網站 origin（`ALLOWED_ORIGIN`）、網站根網址（`SITE_ROOT`）、推播賽事檔的網址（`FEED_URL`）都指向預覽站；發送時段（`SEND_WINDOW`）與今日賽事時段（`TODAY_WINDOW`）設成整天，驗收不必等時段。要改時段就改 `env.preview.vars` 再部署一次。
- 正式環境：同樣五項指向正式網址與正式的時段，值照 README 的表。部署前先確認 `FEED_URL` 那個網址打得開。
- `env.preview` 底下的 `"routes": []` 不能拿掉，否則部署預覽環境會把 `push.vstat.tw` 從正式環境搶走。`vars` 與 `d1_databases` 不被環境繼承，`env.preview` 底下要完整重寫（README「設定項目」）。

設定檔不放任何祕密。D1 識別碼與公鑰這類部署時才有的值，用一個 PR 寫進 `push/wrangler.jsonc` 並合進 main，合併之後才部署（#251 的做法是 PR #280）。

### 3. VAPID 金鑰

兩個環境各一組，不共用。**以下三件事都由使用者本人在自己的終端機做，agent session 不代跑，也不能用會把輸出帶進對話的方式跑（例如 Claude Code 的 `!` 前綴）。私鑰不進 repo、不進對話紀錄、不貼在票或留言。**

1. 產生金鑰：在筆電的 `push/` 底下跑 README「VAPID 金鑰」的產生指令（腳本是 `scripts/gen-vapid.mjs`）。標準輸出有 `VAPID_PUBLIC_KEY=…` 與 `VAPID_PRIVATE_KEY=…` 兩行；公鑰拿去填設定，私鑰先暫存在筆電上、repo 以外的地方。
2. 存成 secret：用 `wrangler secret put VAPID_PRIVATE_KEY` 存私鑰，預覽環境加 `--env preview`，執行後貼上私鑰（README「VAPID 金鑰」）。這一步排在第一次部署之後：服務還沒部署過時，wrangler 會問要不要先建一個同名的新服務（#251 回 n，改成部署後再存）。存好之前沒有任何推播訂閱，發不出通知是預期的。
3. 備份私鑰：存進自己的密碼管理器或離線備份。私鑰只存在 Cloudflare 的 secret 與你的備份裡。**私鑰掉了等於所有推播訂閱作廢**（訂閱綁的是公鑰，換金鑰後舊訂閱收不到）。存好之後從備份讀回私鑰，確認與筆電上暫存的是同一份，再刪掉暫存檔。

公鑰不是祕密，要寫進兩個地方：

- `push/wrangler.jsonc` 對應環境的 `VAPID_PUBLIC_KEY`（預覽環境在 `env.preview.vars`，正式環境在頂層 `vars`）。
- 網站端訂閱推播時要用同一把公鑰：`web/src/lib/pushService.ts` 的 `PUSH_PUBLIC_KEY` 預設值是正式環境的公鑰與 `PUSH_URL`（`https://push.vstat.tw`）。預覽站建置時用環境變數 `VITE_PUSH_URL` 與 `VITE_PUSH_PUBLIC_KEY` 換成預覽環境的網址與公鑰（PR #282）。

兩組公鑰要不同。兩個環境的公鑰都在 `push/wrangler.jsonc`，比對時看那個檔。

發送時的 JWT 標頭 `sub` 是 `https://vstat.tw`，不放信箱（README「通知內容」），這是程式寫死的，不用另外設定。

## 部署

不做自動部署，GitHub 不放任何 Cloudflare 金鑰；由使用者在筆電上手動 `wrangler deploy`。在 `push/` 底下、用本機的 wrangler 跑，指令見 README「部署」：預覽環境第 5 步（`--env preview`），正式環境第 6 步。

- 預覽環境先部署、先驗，過了再部署正式環境。
- 正式環境綁 `push.vstat.tw` 不是獨立的一步：`wrangler.jsonc` 頂層的 `routes` 已寫好，第一次部署時就會綁上，Cloudflare 自己建 DNS 紀錄與憑證（README「部署」正式環境第 5 步）。
- 預期輸出：部署成功並列出網址。預覽環境是 `workers.dev` 網址；正式環境是 `push.vstat.tw`。
- 新服務第一次部署可能回「This Worker does not exist [code: 10007]」，原樣重跑一次就好。
- 部署後定時呼叫約 6 分鐘才第一次觸發，驗證要等。
- 從零重建時，第一次部署成功後回「建環境」第 3 節，把私鑰存成 secret 並備份。

## 部署後驗證

預覽環境與正式環境都要做。命令在 `push/` 底下跑。命令結尾寫 `[--env preview]` 的，驗預覽環境時加上 `--env preview`（不含方括號），驗正式環境時不加；少加就是在驗正式環境。網址欄寫 `<網址>` 的，預覽環境用 `https://tvl-push-preview.tvl-push-proto.workers.dev`，正式環境用 `https://push.vstat.tw`。`<允許的 origin>` 預覽環境是 `https://tommyyi0317.github.io`，正式環境是 `https://vstat.tw`。

「預期」欄取自 #251 部署回報的實測結果，或程式能推出的結果。

| # | 驗證什麼 | 命令 | 預期 |
|---|---|---|---|
| 1 | D1 存在 | `node_modules/.bin/wrangler d1 list` | 列表裡有 `tvl-push-preview` 與 `tvl-push` |
| 2 | 資料表建好 | `node_modules/.bin/wrangler d1 execute DB --remote --command "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('first_feed_read', 'notifications', 'sent_matches', 'sent_today_dates', 'subscriptions') ORDER BY name" [--env preview]` | 剛好五列：`first_feed_read`、`notifications`、`sent_matches`、`sent_today_dates`、`subscriptions`（`migrations/0001_init.sql` 建的五張表）。少一列就是 migration 沒套完。剛建好、還沒部署的資料庫各表都是 0 筆 |
| 3 | 私鑰存好 | `node_modules/.bin/wrangler secret list [--env preview]` | 看得到 `VAPID_PRIVATE_KEY` 這個名稱，看不到值是正常的（#251 驗證） |
| 4 | 拒絕不允許的 origin | `curl -s -i -X POST -H 'Origin: https://example.com' <網址>/subscribe` | 狀態 403，本文 `{"error":"origin_not_allowed"}`（#251 預覽與正式環境都實測過） |
| 5 | 拒絕沒帶 Origin | `curl -s -i <網址>/subscribe` | 狀態 403，本文 `{"error":"origin_not_allowed"}`；是發送服務自己的 JSON，不是連線失敗或 Cloudflare 的錯誤頁（#251 實測 `GET`、`POST` 都是） |
| 6 | 接受允許的 origin | `curl -s -i -X OPTIONS -H 'Origin: <允許的 origin>' <網址>/subscribe` | 狀態 204，帶 `access-control-allow-origin: <允許的 origin>`（#251 實測；預檢標頭見 README「入口」的回應段） |
| 7 | 正式環境網域已綁上 | `curl -s -H 'accept: application/dns-json' 'https://cloudflare-dns.com/dns-query?name=push.vstat.tw&type=A'` | 回應的 `Answer` 內有 A 紀錄（2026-11-18 是 Cloudflare 的位址，#251 實測） |
| 8 | 定時呼叫每分鐘一次 | `node_modules/.bin/wrangler tail [--env preview]`（部署後等約 6 分鐘再看） | 每分鐘出現一筆定時呼叫，每筆的 outcome 都是 `ok`（#251 預覽環境實測 14:41–14:51 每次 `ok`）。排版依 wrangler 版本而定，看的是「每分鐘一筆、沒有 `ok` 以外的 outcome」；看完按 Ctrl+C 結束 |
| 9 | 讀檔成功 | `node_modules/.bin/wrangler d1 execute DB --remote --command "SELECT id, read_at FROM first_feed_read" [--env preview]` | 約每 10 分鐘讀一次檔，第一次成功讀檔後有一列，`id` 是 1，`read_at` 是 ISO 8601 UTC 時間（#251 正式環境 `2026-11-18T07:10:36.583Z`）。沒有列時代表還沒讀到檔：預覽站沒有 `push/matches.json`（404）時照設計什麼都不寫，這是預期的 |

發送服務沒有 `console` 輸出，即時紀錄只看得到每次呼叫的 outcome，所以「讀檔成功與否」要看步驟 9 的 D1，不看即時紀錄（#251 回報；`push/src/` 內沒有 `console` 呼叫）。

### 正式環境第一次啟用：已在站上的賽果被記成已發、沒有發出通知

正式環境第一次啟用（D1 是新建的、還沒讀過檔）時一定要確認這一項；之後的重新部署不用做，那時這些數字早就變了。第一次成功讀檔時，發送服務把推播賽事檔裡已有賽果的場次全記成已發，這次不建立任何通知（README「定時發送」的讀檔段）。等步驟 9 看到 `first_feed_read` 有一列之後，比對三個數字：

1. 站上有賽果的賽事場數（`home_sets_won` 與 `away_sets_won` 都有值）：

   ```bash
   curl -s https://vstat.tw/push/matches.json | jq '[.matches[] | select(.home_sets_won != null and .away_sets_won != null)] | length'
   ```

   預期：印出一個整數 N。取檔的時間要在第一次讀檔之後、且站上沒有新賽果進來（#251 實測當時是 22）。

2. 「已發過的賽事」與「發送紀錄」的筆數：

   ```bash
   node_modules/.bin/wrangler d1 execute DB --remote --command "SELECT (SELECT COUNT(*) FROM sent_matches) AS sent_matches, (SELECT COUNT(*) FROM notifications) AS notifications, (SELECT COUNT(*) FROM sent_today_dates) AS sent_today_dates, (SELECT COUNT(*) FROM subscriptions) AS subscriptions"
   ```

   預期：`sent_matches` 等於步驟 1 的 N；`notifications` 是 0；`sent_today_dates` 與 `subscriptions` 也是 0（尚無人登記時，#251 實測四個數字都符合）。

   例外：第一次讀檔之後的下一次讀檔若落在今日賽事時段、當天又有賽事，發送服務會建立一則今日賽事，沒有人登記也照建（`push/src/judge.ts` 的 `checkFeed`）。這時 `notifications` 與 `sent_today_dates` 各是 1，那一列的 `kind` 是 `today`、`target_count` 是 0，沒有送給任何人，不算不符。要避開就別在台北時間 10:00–11:00 前後做第一次部署。

三項不符時不要接受這個部署，把數字記在 #252 或相關票上，再查原因。注意 `sent_matches` 與 `notifications` 只在「沒有人登記推播訂閱、也沒有新賽果進站」的空窗內才會是這組值；之後有新賽果進站，這兩個數字就會變，不能拿來當日後的驗證。

所在的網路若擋了 `vstat.tw` 整個網域（含 `push.vstat.tw`），上面的 `curl` 會連不上，換一個網路再跑；#251 的做法是把筆電切到手機熱點。

## 查發送紀錄

每則通知在 `notifications` 資料表留一筆，送完時（`finished_at` 有值）這一列就是發送紀錄：時間、哪一種、該送幾筆、成功幾筆、刪了幾筆、當時的推播訂閱總筆數、三個版本的通知文字。

查詢指令、每個欄位的意思與「沒送到幾筆」怎麼算，都在 README「看發送紀錄」；預覽環境加 `--env preview`、正式環境不加。上面七項依序是 `started_at`／`finished_at`、`kind`、`target_count`、`success_count`、`deleted_count`、`subscription_count`、`content_men`／`content_women`／`content_both`。

**上限門檻**：推播訂閱上限是 500 筆。失效但清不掉的推播訂閱會佔名額（例如使用者把 app 從主畫面移除後，Apple 可能照樣回成功，那筆就清不掉）。目前不做清理機制；發送紀錄的 `subscription_count` 到 400 時再開票處理。看最新一則的值：

```bash
node_modules/.bin/wrangler d1 execute DB --remote --command "SELECT id, kind, started_at, subscription_count FROM notifications ORDER BY id DESC LIMIT 1"
```

預期：一列，`subscription_count` 小於 400。沒有任何通知時查無資料，這時改用 `SELECT COUNT(*) FROM subscriptions` 看目前總筆數。

發送紀錄與「發過」的記號都不刪，不寫清理程式（README「資料表」）。

## 清預覽環境的訂閱

驗收（#252）完要清掉預覽環境的訂閱，服務與 D1 留著。**只清預覽環境。這一節的命令寫的是資料庫名稱 `tvl-push-preview`，不是綁定名 `DB`**：`DB` 少打 `--env preview` 就是正式環境的資料庫，`DELETE` 會把真實球迷的推播訂閱全刪；寫資料庫名稱時，wrangler 先在設定檔找、找不到就拿這個名稱去問 Cloudflare，有沒有打旗標連到的都是 `tvl-push-preview`（依據：wrangler 4.162.0 的 `packages/wrangler/src/d1/utils.ts`，`getDatabaseByNameOrBinding`）。

清之前，先在每台驗收用的裝置上到通知頁把兩種通知都關掉（關掉就會退訂）。還開著通知的裝置，下次開預覽站時只要距上次登記滿 24 小時就會自動重新登記（`web/src/lib/push.svelte.ts` 的 `reregisterDaily`），清掉的那一筆會再出現。

1. 確認連到的是預覽環境的資料庫，看有幾筆：

   ```bash
   node_modules/.bin/wrangler d1 execute tvl-push-preview --remote --env preview --command "SELECT COUNT(*) AS n FROM subscriptions"
   ```

   預期：wrangler 開頭印的那一行是 `Executing on remote database tvl-push-preview (<識別碼>):`，識別碼與 `push/wrangler.jsonc` 的 `env.preview.d1_databases` 相同；結果一列，`n` 是還沒退訂的筆數。資料庫名稱或識別碼不對就停，不要做下一步。

2. 刪除：

   ```bash
   node_modules/.bin/wrangler d1 execute tvl-push-preview --remote --env preview --command "DELETE FROM subscriptions"
   ```

   預期：wrangler 回報執行成功、沒有錯誤。刪了幾筆以下一步的查詢為準。

3. 再跑一次步驟 1 的查詢，預期 `n` 是 0。

只清 `subscriptions`。預覽環境的發送紀錄（`notifications`）與「發過」的記號（`sent_matches`、`sent_today_dates`）不刪（README「資料表」）。`subscriptions.id` 只增不重用，清掉之後新登記的從更大的 `id` 開始，不影響分批。

## 已知行為與限制

- **讀檔間隔**：每分鐘一次定時呼叫（上一次沒結束時，下一次會並行），台北時間的分鐘數是 10 的倍數時才讀推播賽事檔，所以正式環境部署後約 20 分鐘內會看到第一次讀檔。推播賽事檔從更新到被發送服務讀到，最久約 20 分鐘（GitHub Pages 的 `max-age=600` 加每 10 分鐘讀一次）。
- **預覽環境的今日賽事**：時段是整天，但每個台北日期仍只發一次。預覽站要放 `push/matches.json`，預覽環境才讀得到檔；第一次成功讀檔只把已有賽果的場次記成已發、不發通知，要看到賽果摘要得在那之後才讓新賽果進檔（#251 交接事項）。
- **上限 500 筆**：滿了新的推播位址回 `503 subscription_limit_reached`，已登記過的不受影響。到 400 筆就要開票，見「查發送紀錄」。
- **寧可漏發不重發**：認領成功後呼叫中斷會漏發那一批；賽果事後更正不重發（README「定時發送」）。
- **沒有主動告警**：發送服務不通知維護者，出問題要靠查 D1。
- **自我移除版**：第 4 期上線後，若動用 service worker 的自我移除版（`docs/ops/service-worker.md`），推播的收取寫在同一支 service worker 裡，裝置會一起收不到通知。
- **Android**：安裝與離線上線前只在虛擬裝置上驗過，沒有實機；推播可能完全沒在 Android 上驗過（spec #231 的已知限制）。
- **私鑰掉了**：沒有救法，只能重產金鑰、改設定與網站的公鑰、重新部署，所有裝置要重新登記推播訂閱。
