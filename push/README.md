# 發送服務（push/）

替本站把推播送到球迷裝置的服務：Cloudflare Workers（免費方案）加一個 D1 資料庫，TypeScript。正式環境網址 `https://push.vstat.tw`。

目前做到的（#244）：資料表、公開的登記入口與退訂入口。還沒有定時器、不讀推播賽事檔、不發任何通知，這些是 #245。為什麼狀態存 D1、不重發怎麼做，見 `docs/adr/0014-push-service-state-in-d1-claim-by-conditional-write.md`。

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

本機要讓網站（例如 `http://localhost:5173`）打得到，在 `push/.dev.vars`（已 gitignore）寫 `ALLOWED_ORIGIN=http://localhost:5173`，它會蓋過 `wrangler.jsonc` 的 `vars`。單元測試的 `ALLOWED_ORIGIN` 寫死在 `vitest.config.ts`，不受 `.dev.vars` 影響。

看本機 D1 的內容：`npx wrangler d1 execute DB --local --command "SELECT * FROM subscriptions"`。

注意：`src/index.ts` 是 Worker 的進入點，只能 export handler（`export default`）與型別。export 一般的值（例如常數）時單元測試照樣會過，但 `wrangler dev` 與部署會在啟動時失敗（`Incorrect type for map entry …`）。

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
| `keys.p256dh` | 字串 | base64url，解開是 65 bytes（無填充 87 字）；結尾的 `=` 填充可有可無 |
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
| 400 | `{"error": "invalid_body"}` | 不是 JSON、結構不對、型別不對、長度不對、兩種都不收、組別不認得 |
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

| 名稱 | 放哪裡 | 正式環境的值 | 說明 |
|---|---|---|---|
| `ALLOWED_ORIGIN` | `wrangler.jsonc` 的 `vars` | `https://vstat.tw` | 允許的網站 origin，只能一個。#245 替預覽環境換成預覽站的 origin |
| `DB` | `wrangler.jsonc` 的 `d1_databases` | `database_name` 為 `tvl-push` | D1 綁定。`database_id` 目前是佔位值 `00000000-0000-0000-0000-000000000000`，本機模式與單元測試可用；遠端資料庫建好後由 #251 換成真的識別碼 |

`vars` 與 `d1_databases` 不會被 wrangler 的環境（`env.<名稱>`）繼承，加預覽環境時兩者都要在那個環境底下重寫。設定檔不放任何祕密。

## 資料表

`migrations/` 底下的 D1 migration（要改資料表就新增一個 migration，不改舊的）。

| 資料表 | 存什麼 |
|---|---|
| `subscriptions` | 推播訂閱：推播位址（唯一）、`p256dh`、`auth`、賽果摘要開關、今日賽事開關、組別、建立與更新時間。`id` 只增不重用 |
| `sent_matches` | 已發過的賽事，唯一鍵 `cup_id` 加 `match_group` 加 `game_id` |
| `sent_today_dates` | 今日賽事已發過的台北日期（`YYYY-MM-DD`），唯一鍵 |
| `notifications` | 每一則通知一筆：批次進度（`next_offset`，各則分開記，暫停一則再接著送不需要別的狀態）、開始送時定下來的三個版本內容（`content_men`、`content_women`、`content_both`）。送完時（`finished_at`）這一列就是發送紀錄：時間、哪一種、該送幾筆、成功幾筆、刪了幾筆、當時的推播訂閱總筆數、三個版本的通知文字 |
| `first_feed_read` | 發送服務第一次成功讀推播賽事檔的記號，最多一列；用 `INSERT OR IGNORE` 寫，改到一列的那次呼叫就是第一次 |

發送紀錄與「發過」的記號都不刪，不寫清理程式。

## 部署與綁定網域（#251 做）

這張票不部署。#251 需要：

1. `npx wrangler d1 create tvl-push`，把回傳的識別碼填進 `wrangler.jsonc` 的 `database_id`。
2. `npx wrangler d1 migrations apply DB --remote` 建表。
3. `npx wrangler deploy`。
4. 綁定 `push.vstat.tw`：`vstat.tw` 已經是 Cloudflare 上的 zone（ADR 0013），在 `wrangler.jsonc` 加 `"routes": [{ "pattern": "push.vstat.tw", "custom_domain": true }]` 再部署，或在 Cloudflare 後台 Worker 的 Domains & Routes 加 Custom Domain。Cloudflare 會自己建 DNS 紀錄與憑證；條件是 `push.vstat.tw` 不能已經有 CNAME 紀錄。依據：<https://developers.cloudflare.com/workers/configuration/routing/custom-domains/>。
