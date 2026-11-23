# 比分服務

Cloudflare 每分鐘呼叫一次 `tvl-score`。排程使用事件的預定時間換算台北日期。整包每天啟動；精簡版只在當天有賽事或賽事檔無法讀取、解析時啟動。其他分鐘不發請求。服務不儲存狀態，HTTP 請求一律回 404。

| 設定 | 預設值 | 用途 |
| --- | --- | --- |
| `FULL_TIME` | `09:23` | 每日整包的台北時間 |
| `SLIM_TIMES` | `15:30,18:30,21:15` | 精簡版的台北時間，以半形逗號分隔，每項使用 HH:MM |
| `FEED_URL` | `https://vstat.tw/push/matches.json` | 網站已發佈的推播賽事檔 |
| `GITHUB_REPO` | `oldyi317/TVL-Analysis` | 要啟動 workflow 的擁有者與 repo |
| `GITHUB_WORKFLOW` | `daily-crawl.yml` | 在 main 分支上啟動的 workflow 檔名 |
| `DRY_RUN` | `true` | 布林值。啟用時送出 `dry_run: true`；關閉時沿用 workflow 的寫入設定 |
| `GITHUB_TOKEN` | Worker secret | 呼叫 GitHub workflow dispatch 的 fine-grained token，限目標 repo，Actions 權限為 write |

設定值放在 `wrangler.json` 的 `vars`；部署步驟、secret 怎麼產生與更換見 `docs/ops/daily-crawl.md`「比分服務」。GitHub 回非成功狀態時，每次排程最多嘗試三次，最後一次仍失敗就回報錯誤。

在本目錄用 Node.js 24 執行 `npm ci`、`npm run check`、`npm test`。
