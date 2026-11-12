# D1 認領實測（拋棄式原型）

「發送服務的剩餘參數」（#227）用的實測，2026-11-12 跑完。結論與數字在該票的結論留言。這個分支不開 PR、不合併。

- `src/index.js`：測試用的 Worker。HTTP 路由各測一種寫法，`scheduled` 每分鐘認領一次、每三次多等 75 秒。
- `scripts/run-tests.mjs`：`node scripts/run-tests.mjs <網址> <token>`，依序跑連續讀寫、條件式寫入、`RETURNING`、`batch()`、唯一鍵、並行搶同一批。
- `results/run-1.json`：上面那支的輸出。
- `results/cron-log.json`：定時呼叫 11 次的紀錄（`scheduled_time`、`started_at`、`finished_at` 是毫秒時間戳）。

部署時 token 用 `wrangler deploy --var TOKEN:<值>` 帶入。Worker `tvl-d1-proto` 與同名的 D1 資料庫測完已刪，`wrangler.jsonc` 裡的 `database_id` 已失效。
