# 發送服務（push/）

替本站把推播送到球迷裝置的服務：Cloudflare Workers（免費方案）加一個 D1 資料庫，TypeScript。正式環境網址 `https://push.vstat.tw`。

做的事：公開的登記入口與退訂入口（#244），以及定時讀推播賽事檔、判斷賽果摘要與今日賽事、分批發送、留發送紀錄（#245）。為什麼狀態存 D1、不重發怎麼做，見 `docs/adr/0014-push-service-state-in-d1-claim-by-conditional-write.md`。

自己一份 `package.json`，不與 `web/` 共用依賴。部署由使用者在筆電上手動 `wrangler deploy`，不做自動部署，GitHub 不放任何 Cloudflare 金鑰；CI（`.github/workflows/ci.yml` 的 `push` job）只跑型別檢查與單元測試。

## 本機

Node 要 22.12 以上（wrangler 的要求）。以下命令都在 `push/` 底下跑。

```bash
npm ci
npm run check            # 型別檢查（src 與 test）
npm test                 # 單元測試（vitest + @cloudflare/vitest-plugin，在本機的 workerd 裡跑，D1 是本機的）
npm run migrate:local    # 把 migrations/ 套到本機 D1（存在 push/.wrangler/，已 gitignore）
npm run dev              # wrangler dev，本機模式，預設 http://localhost:8787
```

本機要讓網站（例如 `http://localhost:5173`）打得到，在 `push/.dev.vars`（已 gitignore）寫 `ALLOWED_ORIGIN=http://localhost:5173`，它會蓋過 `wrangler.jsonc` 的 `vars`。單元測試用的 `ALLOWED_ORIGIN`、`SITE_ROOT`、`FEED_URL`、`SEND_WINDOW`、`TODAY_WINDOW` 寫死在 `vitest.config.ts`，不受 `.dev.vars` 影響。

看本機 D1 的內容：`node_modules/.bin/wrangler d1 execute DB --local --command "SELECT * FROM subscriptions"`。

手動觸發一次定時器（本機不會自己每分鐘觸發）：`npm run dev` 起來之後

```bash
curl "http://localhost:8787/cdn-cgi/local/scheduled?cron=*+*+*+*+*&time=1795226400000&format=json"
```

`time` 是這次呼叫的排程時間（epoch 毫秒，省略就是現在；例子是台北 2026-11-21 10:00）。回 `{"outcome":"ok",…}` 表示跑完沒丟錯。wrangler 4.162 實測可用。注意：
- 一次觸發等於一次每分鐘的呼叫；排程時間的分鐘數是 10 的倍數才會讀檔，其餘只送批次。要測讀檔、時段、今日賽事就用 `time` 指定時間。
- 本機會真的去抓 `FEED_URL`、真的對推播位址送出，而且要有 `VAPID_PUBLIC_KEY`、`VAPID_PRIVATE_KEY` 才送得出去。要蓋過這些值與時段，用 `node_modules/.bin/wrangler dev --env-file <檔>` 指到 `push/` 以外的檔（一行一個 `名稱=值`）；寫在 `push/.dev.vars` 也可以，但那個檔單元測試也會讀到，用完要刪。

注意：`src/index.ts` 只能 `export default`（其他 export 會讓 workerd 啟動失敗），`test/entry.test.ts` 會擋。

`score/` 的 `src/schedule.ts` 匯入了 `push/src/time.ts`；改它要連 `score/` 的測試一起跑（`cd ../score && npm test`）。

## 入口

兩個入口都只收 `POST`，請求本文是 JSON，本文上限 4,096 bytes。推播位址本身就是憑證：知道位址才改得到或刪得到那一筆，不另做身分驗證。

### `POST /subscribe`：登記與改偏好

推播位址已存在就覆寫金鑰與偏好，不存在就新增。

```json
{
	"endpoint": "https://web.push.apple.com/…",
	"keys": { "p256dh": "<base64url，65 bytes>", "auth": "<base64url，16 bytes>" },
	"results": true,
	"today": false,
	"group": "both"
}
```

| 欄位 | 型別 | 說明 |
|---|---|---|
| `endpoint` | 字串，1–2,048 字 | 推播位址。必須是 `https`、不帶帳號密碼與連接埠、主機在「推播服務網域」清單內 |
| `keys.p256dh` | 字串 | base64url，解開是 65 bytes（無填充 87 字），而且要是 P-256 曲線上的未壓縮公鑰（瀏覽器給的都是）；結尾的 `=` 填充可有可無 |
| `keys.auth` | 字串 | base64url，解開是 16 bytes（無填充 22 字） |
| `results` | 布林 | 收不收賽果摘要 |
| `today` | 布林 | 收不收今日賽事 |
| `group` | `"men"`、`"women"`、`"both"` | 組別 |

推播位址存的是正規化後的網址（主機轉小寫、去掉預設連接埠 `:443` 與 `#` 之後的片段），同一個位址的不同寫法算同一筆；退訂入口照同樣規則比對。

`results` 與 `today` 不能都是 `false`：兩種都不收時呼叫退訂入口。其他欄位不看，所以可以直接把瀏覽器的 `subscription.toJSON()`（含 `expirationTime`）展開後加上三個偏好欄位送出。

### `POST /unsubscribe`：退訂

```json
{ "endpoint": "https://web.push.apple.com/…" }
```

刪掉那一筆。位址不存在也回成功。

### 回應

成功與失敗都回 JSON，並帶 `Access-Control-Allow-Origin: <請求的 Origin>` 與 `Vary: Origin`（origin 不允許時除外）。

| 狀態碼 | 本文 | 什麼時候 |
|---|---|---|
| 200 | `{"ok": true}` | 登記、改偏好或退訂成功 |
| 204 | （無） | 預檢（`OPTIONS`）通過 |
| 400 | `{"error": "invalid_body"}` | 不是 JSON、結構不對、型別不對、長度不對、`p256dh` 不是曲線上的公鑰、兩種都不收、組別不認得 |
| 400 | `{"error": "endpoint_not_allowed"}` | 推播位址不是 `https`、主機不在清單內、帶帳號密碼或連接埠 |
| 403 | `{"error": "origin_not_allowed"}` | 沒帶 `Origin`，或 `Origin` 不是允許的網站 origin（預檢也一樣，且不帶 CORS 標頭） |
| 404 | `{"error": "not_found"}` | 路徑不是上面兩個 |
| 405 | `{"error": "method_not_allowed"}` | 不是 `POST` 或 `OPTIONS`（帶 `Allow: POST, OPTIONS`） |
| 413 | `{"error": "body_too_large"}` | 本文超過 4,096 bytes |
| 503 | `{"error": "subscription_limit_reached"}` | 推播訂閱已滿 500 筆，這是新的推播位址；已登記過的位址不受影響 |

被拒絕的請求一律不寫入。

預檢回應：`Access-Control-Allow-Methods: POST`、`Access-Control-Allow-Headers: Content-Type`、`Access-Control-Max-Age: 86400`。

## 推播服務網域

只收這些主機的推播位址（比對整個主機名或以點隔開的後綴，不用字串包含）：

| 推播服務 | 允許的主機 | 依據 |
|---|---|---|
| Apple | `*.push.apple.com`（實際是 `web.push.apple.com`） | Apple「Sending web push notifications in web apps and browsers」：「allow access for `https://*.push.apple.com`」；WebKit blog「Web Push for Web Apps on iOS and iPadOS」同樣寫 `*.push.apple.com` |
| Google（Chrome 與 Chromium 系） | `fcm.googleapis.com` | Chrome 的「Web Push interop wins」：「if the origin is `fcm.googleapis.com`, it's working」 |
| Mozilla（Firefox） | `updates.push.services.mozilla.com` | autopush-rs 文件列的 endpoint 主機 |
| Microsoft（Windows 上的 Edge，WNS） | `*.notify.windows.com` | Microsoft「Windows Push Notification Services (WNS) overview」：通道網址必須在 `notify.windows.com` 網域，「The subdomain of the channel URI is subject to change and should not be considered when validating」 |

查證日 2026-11-16。有人回報登記失敗、而推播位址屬於這四家但主機不在表內時，再對照該家文件加進 `src/validate.ts`。

## 設定項目

兩個環境：正式環境是 `wrangler.jsonc` 的頂層設定，預覽環境（只給驗收用，另一個 Worker、另一個 D1、另一組 VAPID）是 `env.preview`。兩者的差別全是設定，程式相同。

| 名稱 | 種類 | 正式環境 | 預覽環境（`env.preview`） |
|---|---|---|---|
| `DB` | D1 綁定 | `tvl-push` | `tvl-push-preview` |
| `ALLOWED_ORIGIN` | var | `https://vstat.tw` | `https://tommyyi0317.github.io` |
| `SITE_ROOT` | var | `https://vstat.tw` | `https://tommyyi0317.github.io/tvl-shell-proto` |
| `FEED_URL` | var | `https://vstat.tw/push/matches.json` | `https://tommyyi0317.github.io/tvl-shell-proto/push/matches.json` |
| `SEND_WINDOW` | var | `08:00-22:00` | `00:00-23:59`（整天，驗收用） |
| `TODAY_WINDOW` | var | `10:00-11:00` | `00:00-23:59`（整天，驗收用） |
| `VAPID_PUBLIC_KEY` | var | 正式環境那組的公鑰 | 預覽環境那組的公鑰 |
| `VAPID_PRIVATE_KEY` | secret | `wrangler secret put` 存，不進 repo | 另一組，`wrangler secret put --env preview` |

- 說明：`ALLOWED_ORIGIN` 只能一個 origin；`SITE_ROOT` 是通知連結的網站根，預覽站含子路徑；`FEED_URL` 是推播賽事檔的網址；`SEND_WINDOW` 是新的一則可以開始的時段（已開始的會送完）；`TODAY_WINDOW` 是今日賽事可以發的時段。
- 格式：`SITE_ROOT` 不帶結尾斜線；時段是 `HH:MM-HH:MM`、台北時間、閉區間（`08:00-22:00` 表示 08:00 到 22:00 這一分鐘都算），開始不可晚於結束。`SEND_WINDOW` 解析不出來時不開始任何新的一則；`TODAY_WINDOW` 解析不出來時不發今日賽事，賽果摘要照常。
- 兩個環境的 `database_id` 與 `VAPID_PUBLIC_KEY` 已填真值，值看 `wrangler.jsonc`。本機模式與單元測試用的是本機 D1，不碰遠端。
- `vars` 與 `d1_databases` 不會被 wrangler 的環境（`env.<名稱>`）繼承，所以 `env.preview` 底下兩者都要完整重寫。
- `triggers` 與 `routes` 會被環境繼承。定時器（`triggers.crons = ["* * * * *"]`）寫在頂層一次，兩個環境都有。`env.preview` 寫了 `"routes": []`，不要拿掉：正式環境在頂層綁了 `push.vstat.tw` 之後，沒有這一行的話部署預覽環境會把網域從正式環境搶走（wrangler 4.162 的警告：「Deploying this environment will reassign these custom domains away from the top-level Worker」）。
- 設定檔不放任何祕密；私鑰只存在 Cloudflare 的 secret。
- 本機要蓋過 `vars` 時寫 `push/.dev.vars`（見「本機」）。

## 定時發送

定時器每分鐘跳一次（`* * * * *`），不開任何對外的觸發入口。時間一律是台北時間（UTC+8）。

每次呼叫：
1. 台北時間的分鐘數是 10 的倍數時，先讀一次推播賽事檔（`FEED_URL`）。讀檔失敗（見下）不擋後面的批次；判斷途中真的丟錯（例如 D1 出錯）時這次呼叫失敗、留在平台的紀錄裡，批次下一分鐘照送。
2. 不論讀不讀檔，都送一批還沒送完的通知。

讀檔：
- 連不上、逾時（10 秒）、狀態碼不是 2xx、不是 JSON、`version` 不是 `1`、格式不對，都算讀檔失敗：什麼都不記、不發，下一次再試。
- 第一次成功讀檔：把檔裡已有賽果（`home_sets_won` 與 `away_sets_won` 都有值）的場次全記成已發，這次不建立任何通知（今日賽事也不建，下一次讀檔再判斷）。
- 現在不在 `SEND_WINDOW` 內就不開始新的一則，也不認領；已開始的照常送完。

兩種通知：

| | 賽果摘要 | 今日賽事 |
|---|---|---|
| 條件 | 檔裡有賽果、沒發過的場次；比賽日在最近 7 天內 | 現在在 `TODAY_WINDOW` 內、今天（台北日期）的還沒發過、檔裡有今天的賽事 |
| 合成 | 這次認領到的場次合成一則；同一天兩批不同賽果進站會有兩則 | 當時檔裡今天的賽事合成一則，一天最多一則；時段內沒成功就下次檢查再試，過了時段放棄 |
| 7 天 | 今天減比賽日（台北日曆日）不超過 7 天才通知；超過的直接記成已發、不通知 | — |
| 識別 | 一場是 `cup_id` 加 `group` 加 `game_id`；事後更正賽果不重發 | 台北日期 |

- 兩則同時要送時，今日賽事先送，賽果摘要暫停，等它送完再從停下處接著送。
- 誰收：依每筆訂閱的偏好（賽果摘要、今日賽事、組別）。沒有該訂閱所選組別的內容就不發給他。一則的文字在建立那一刻定下來，有男子組、女子組、兩組合併三種版本；訂閱選兩組但內容只有一組時用那一組的版本。
- 認領與「寧可漏發不重發」：先認領、認領成功才送。
  - 每場：`sent_matches` 的唯一鍵，寫入成功才算這次認領到。
  - 每天：`sent_today_dates` 的唯一鍵。
  - 每批：`UPDATE notifications SET next_offset = <新> WHERE id = <那則> AND next_offset = <舊>`，`meta.changes` 為 1 才送。
  - 認領成功後呼叫中斷會漏發那一批，接受。
- 分批：每批 15 筆（免費方案每次呼叫 10 ms CPU）。進度 `next_offset` 是「已認領到的最後一筆訂閱 id」，`subscriptions.id` 只增不重用，訂閱增減不會讓人被跳過或重複，新登記的排在最後。
- 回應處理：2xx 算成功；404、410 當場刪那筆訂閱；其他（429、5xx、逾時、連不上）不刪、不重試、不計數。
- 紀錄：每則在 `notifications` 留一筆（見「看發送紀錄」）。

## 通知內容

發送服務與 service worker 後備（#248）共同的合約。解密後的本文是 Declarative Web Push 的 JSON：

```json
{
	"web_push": 8030,
	"notification": {
		"title": "今日賽事",
		"body": "14:00 男子 臺電 62%–38% 雲林美津濃\n19:00 女子 臺電 對 雲林美津濃",
		"navigate": "https://vstat.tw/matches/#d-2026-11-08"
	}
}
```

（範例的路徑 `/matches/` 為示意，實際是推播賽事檔 `links` 給的路徑。）

| 欄位 | 說明 |
|---|---|
| `web_push` | 固定 `8030` |
| `notification.title` | 標題。賽果摘要：`男子組賽果`、`女子組賽果`、兩組都有時 `最新賽果`；今日賽事：`今日男子組賽事`、`今日女子組賽事`、兩組都有時 `今日賽事` |
| `notification.body` | 各行用換行（`\n`）分隔，一場一行，最多 8 行；超過時列 7 行，第 8 行是 `另有 N 場`。賽果摘要一行：`11/8 臺電 3–1 雲林美津濃`，兩組都有時 `11/8 男子 臺電 3–1 雲林美津濃`（日期後加組別）。今日賽事一行：`14:00 臺電 62%–38% 雲林美津濃`，無賽前勝率時 `14:00 臺電 對 雲林美津濃`，兩組都有時時間後加組別；開賽時間空值的那行不寫時間、排最後。主隊在左，比分用 en dash `–` |
| `notification.navigate` | 完整網址：`SITE_ROOT` ＋ 賽事頁路徑（推播賽事檔 `links` 的一組，內容只有一組用該組，兩組用 `links.both`）＋ `#d-YYYY-MM-DD`。今日賽事是當天，賽果摘要是內容裡最新的比賽日 |

加密與送出：
- 加密是 RFC 8291 的 `aes128gcm`，單一 record，純 WebCrypto、不加依賴。
- 標頭：`Authorization: vapid t=<JWT>, k=<公鑰>`、`Content-Encoding: aes128gcm`、`Content-Type: application/octet-stream`、`TTL: 43200`。
- 不送 `Topic`（同一天的兩則不會互相取代），不送 `Urgency`（用預設）。
- JWT：ES256，`aud` 是推播位址的 origin，`exp` 是 12 小時後，`sub` 是 `https://vstat.tw`（不放信箱）。

## VAPID 金鑰

兩個環境各用一組，不共用。在 `push/` 底下跑（Node 22 以上，只用內建模組）：

```bash
node scripts/gen-vapid.mjs
```

標準輸出是 `VAPID_PUBLIC_KEY=…` 與 `VAPID_PRIVATE_KEY=…` 兩行（無填充 base64url）。

- 公鑰：填進 `wrangler.jsonc` 對應環境的 `VAPID_PUBLIC_KEY`。網站端訂閱推播時要用同一把公鑰（該環境對應的網站）。
- 私鑰：只用 `node_modules/.bin/wrangler secret put VAPID_PRIVATE_KEY` 存（預覽環境加 `--env preview`），執行後貼上私鑰。不進 repo，不寫進設定檔。
- **私鑰要自己備份。掉了等於所有推播訂閱作廢**（訂閱綁的是公鑰，換金鑰後舊訂閱收不到）。

## 部署

不做自動部署，GitHub 不放任何 Cloudflare 金鑰。以下在 `push/` 底下、用本機的 wrangler 跑。兩個環境的 D1 與金鑰都已建好、值已填進 `wrangler.jsonc`，平常改了程式只要跑各自的「部署」那一步；建 D1、產生金鑰、填值是從零重建時才做。

`d1 execute`、`d1 migrations apply` 的 `--preview` 旗標是「D1 的 preview 資料庫」，與這裡的預覽環境無關，不要用。

預覽環境：
1. 建 D1：`node_modules/.bin/wrangler d1 create tvl-push-preview`，把回傳的識別碼填進 `env.preview.d1_databases` 的 `database_id`。不用 `--update-config`（它會自己改設定檔，改在哪個環境底下要自己確認）。
2. 套 migration：`node_modules/.bin/wrangler d1 migrations apply DB --remote --env preview`。
3. 產生金鑰（見「VAPID 金鑰」），存私鑰：`node_modules/.bin/wrangler secret put VAPID_PRIVATE_KEY --env preview`。
4. 把公鑰填進 `env.preview.vars.VAPID_PUBLIC_KEY`。
5. 部署：`node_modules/.bin/wrangler deploy --env preview`。部署出來的名稱是 `tvl-push-preview`，網址在 `workers.dev` 底下，看部署輸出。
6. 預覽環境的 `SEND_WINDOW`、`TODAY_WINDOW` 是整天（`00:00-23:59`），驗收不必等時段；要改成別的時段就改 `env.preview.vars` 後再部署一次。

正式環境：
1. 建 D1：`node_modules/.bin/wrangler d1 create tvl-push`，識別碼填進頂層 `d1_databases` 的 `database_id`。
2. 套 migration：`node_modules/.bin/wrangler d1 migrations apply DB --remote`。
3. 產生另一組金鑰，存私鑰：`node_modules/.bin/wrangler secret put VAPID_PRIVATE_KEY`。
4. 把公鑰填進頂層 `vars.VAPID_PUBLIC_KEY`。
5. 綁定 `push.vstat.tw`：`vstat.tw` 已經是 Cloudflare 上的 zone（ADR 0013）。`wrangler.jsonc` 頂層已有 `"routes": [{ "pattern": "push.vstat.tw", "custom_domain": true }]`，部署時就會綁上。Cloudflare 會自己建 DNS 紀錄與憑證；條件是 `push.vstat.tw` 不能已經有 CNAME 紀錄。依據：<https://developers.cloudflare.com/workers/configuration/routing/custom-domains/>。頂層的 `routes` 會被環境繼承，`env.preview` 已用 `"routes": []` 擋掉（見「設定項目」）。
6. 部署：`node_modules/.bin/wrangler deploy`。

## 看發送紀錄

預覽環境加 `--env preview`，正式環境不加：

```bash
node_modules/.bin/wrangler d1 execute DB --remote \
  --command "SELECT id, kind, started_at, finished_at, target_count, success_count, deleted_count, subscription_count FROM notifications ORDER BY id DESC LIMIT 10"
```

加 `--json` 輸出 JSON；要看三個版本的文字就選 `content_men`、`content_women`、`content_both`。

| 欄位 | 說明 |
|---|---|
| `kind` | `results`（賽果摘要）或 `today`（今日賽事） |
| `started_at`、`finished_at` | 建立與送完的時間（ISO 8601，UTC）；`finished_at` 是 `NULL` 表示還沒送完 |
| `next_offset` | 批次進度：已認領到的最後一筆訂閱 id |
| `target_count` | 建立當下該收這一則的推播訂閱數 |
| `success_count` | 推播服務回 2xx 的筆數 |
| `deleted_count` | 回 404、410 而刪掉的訂閱筆數 |
| `subscription_count` | 建立當下的推播訂閱總筆數 |
| `content_men`、`content_women`、`content_both` | 三個版本的通知內容，JSON（`title`、`body`、`navigate`）；該版本沒有內容時是 `NULL` |

`target_count` 減 `success_count` 減 `deleted_count` 是沒送到的筆數（其他錯誤、或認領後呼叫中斷漏掉的一批）。

## 資料表

`migrations/` 底下的 D1 migration（要改資料表就新增一個 migration，不改舊的）。

| 資料表 | 存什麼 |
|---|---|
| `subscriptions` | 推播訂閱：推播位址（唯一）、`p256dh`、`auth`、賽果摘要開關、今日賽事開關、組別、建立與更新時間。`id` 只增不重用 |
| `sent_matches` | 已發過的賽事，唯一鍵 `cup_id` 加 `match_group` 加 `game_id` |
| `sent_today_dates` | 今日賽事已發過的台北日期（`YYYY-MM-DD`），唯一鍵 |
| `notifications` | 每一則通知一筆：批次進度（`next_offset`，是已認領到的最後一筆訂閱 id、一開始是 0；各則分開記，暫停一則再接著送不需要別的狀態）、開始送時定下來的三個版本內容（`content_men`、`content_women`、`content_both`）。送完時（`finished_at`）這一列就是發送紀錄：時間、哪一種、該送幾筆、成功幾筆、刪了幾筆、當時的推播訂閱總筆數、三個版本的通知文字 |
| `first_feed_read` | 發送服務第一次成功讀推播賽事檔的記號，最多一列；用 `INSERT OR IGNORE` 寫，改到一列的那次呼叫就是第一次，由它把已有賽果的場次記成已發 |

發送紀錄與「發過」的記號都不刪，不寫清理程式。
