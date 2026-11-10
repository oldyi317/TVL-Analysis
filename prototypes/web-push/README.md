# tvl-push-proto（拋棄式 Web Push 實測原型，對應 issue #195）

一個 Cloudflare Worker：`public/` 提供測試頁，`/api/*` 由 `src/index.js` 處理；加密與簽章在 `src/webpush.js`（純 WebCrypto）。

**狀態：2026-11-10 已用 iPhone（iOS 27.2）實測完並拆除（Worker 與 KV 已刪）。** 結論在 issue #195 的結論留言；當天的原始紀錄在 `results/`（`log-final.json` 是 `/api/log` 的最後一次輸出，`tail-summary.txt` 是 `wrangler tail --format json` 的 CPU 時間彙整）。要重跑：`node scripts/gen-vapid.mjs` 產生新金鑰對，照下面「部署」做。

## 本機

```bash
npm install
node scripts/test-webpush.mjs      # RFC 8291 已知答案、JWT、粗略計時
npx wrangler dev                   # http://localhost:8787
curl "http://localhost:8787/cdn-cgi/local/scheduled?cron=*+*+*+*+*"   # 手動觸發一次 Cron
```

## 部署（登入 Cloudflare 後，依序）

```bash
npx wrangler login
# 1. KV：wrangler.jsonc 的 PROTO_KV 沒寫 id，deploy 時 wrangler 會自動建立（若失敗再手動：npx wrangler kv namespace create PROTO_KV，把 id 填進 wrangler.jsonc）
# 2. 私鑰 secret（公鑰已在 wrangler.jsonc 的 vars，須與私鑰成對；要換金鑰就跑 node scripts/gen-vapid.mjs，改 vars 的 VAPID_PUBLIC 並貼新私鑰）
npx wrangler secret put VAPID_PRIVATE      # 貼上 .dev.vars 裡 VAPID_PRIVATE= 後面的值
# 3. 部署
npx wrangler deploy
```

部署後用 iPhone Safari 開 `https://tvl-push-proto.<子網域>.workers.dev` →「分享」→「加入主畫面」→ 從主畫面圖示開啟 → 按「訂閱」。

## 看 CPU 時間

- 儀表板：Workers & Pages → 選 `tvl-push-proto` → Overview（Metrics），圖表名稱 **CPU Time per execution**（分位數）。來源：https://developers.cloudflare.com/workers/observability/metrics-and-analytics/
- 超限時：免費方案每次 HTTP 請求與每次 Cron 的 CPU 上限皆為 10 ms，超過回 Error 1102（Worker exceeded resource limits），分析資料中 outcome 為 `exceededCpu`。
- `npx wrangler tail --format json`：每筆事件頂層有 `cpuTime` 與 `wallTime`（毫秒，整數）與 `outcome`（2026-11-10 以 wrangler 4.159.0 實測）。
- `GET /api/bench?n=<則數>&jwt=once|each`：只做簽章加密、不發送，用來量一次呼叫發多則的 CPU 時間。

## 拆除

```bash
npx wrangler delete                         # 刪 Worker（會一併移除 Cron 與 secret）
npx wrangler kv namespace list              # 找出 PROTO_KV 的 id
npx wrangler kv namespace delete --namespace-id <id>
```

並在 iPhone 刪除主畫面圖示、到設定移除通知權限。
