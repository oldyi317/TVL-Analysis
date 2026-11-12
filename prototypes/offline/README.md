# 離線原型（拋棄式原型，不合併）

分支 `prototype/app-offline`，對應 issue #229。目的是在預覽站（GitHub Pages 子路徑 `/tvl-shell-proto`）上，讓使用者用 iPhone 主畫面的獨立視窗實測離線與換版。這個分支永遠不會合併。

## 在回答的六個問題

1. 空殼頁：離線冷啟動時，只靠 adapter-static 的 `fallback`（`shell.html`）加存好的程式檔與資料檔，能不能開出列表層級頁面、當季單場頁與球員頁。
2. 沒存的頁：離線點到沒看過的歷史單場頁／球員頁，會不會出站內說明頁而不是瀏覽器錯誤頁。
3. 3 秒：網路很慢時，等 3 秒才出現存的內容，手感如何（除錯面板可調等待上限與人為延遲）。
4. 30 分鐘：獨立視窗離開超過門檻自動換新、門檻內只跳「有新資料，點此更新」；iPhone 從背景回來時頁面有沒有被系統重載（看事件紀錄）。
5. 下載量：首次下載是否 gzip 後 5 MB 內；資料更新一次重抓是否 3 MB 內。
6. 圖示與名稱：四張圖示與「VStat 企排」在主畫面的樣子（manifest 與 apple-touch-icon，沒設 `apple-mobile-web-app-title`）。

## 建置

兩份 build（模擬每日更新），程式碼必須相同、只有資料不同。從 repo 根目錄：

```bash
git checkout 25e7e99 -- data/db/tvl_database.db        # v1：11-09 每日更新的前一個狀態
~/venvs/tvl/bin/python -m src.export.site_data
(cd web && BASE_PATH=/tvl-shell-proto npm run build)
cp -r web/build prototypes/offline/out/v1
git checkout HEAD -- data/db/tvl_database.db            # v2：目前 HEAD
~/venvs/tvl/bin/python -m src.export.site_data
(cd web && BASE_PATH=/tvl-shell-proto npm run build)
cp -r web/build prototypes/offline/out/v2
git status --short                                      # 不得有 data/db 的改動
```

圖示只在改原稿時重產：`node prototypes/offline/make-icons.mjs`（Playwright，輸出到 `web/static/icons/`）。

## 檢查

```bash
node prototypes/offline/check.mjs                 # 預設 out/v1、out/v2、base /tvl-shell-proto
node prototypes/offline/check.mjs --v1 <dir> --v2 <dir> --base /xxx --label v1 --skip-swap
node prototypes/offline/measure.mjs               # 吃 out/v1、out/v2 與 out/cached-urls-v1.json
bash prototypes/offline/measure-live.sh <站台網址> prototypes/offline/out/cached-urls-v1.json   # 部署後量實際傳輸量
node prototypes/offline/serve.mjs prototypes/offline/out/v1 /tvl-shell-proto 4180   # 手動起伺服器
```

`check.mjs` 在 Chromium（Pixel 7 尺寸）逐項印 ✓／✗。離線用 `context.setOffline(true)`；模擬背景用覆寫 `document.visibilityState` 再發 `visibilitychange`，「離開多久」靠真的等（門檻調成 10 秒）。快取清單寫到 `out/cached-urls-<label>.json`。

## 除錯面板（右下角「除錯」鈕）

顯示：頁面版本、接手的 SW 版本、等待中的新版、顯示模式、線上／離線、這次導覽怎麼回（網路／空殼頁（逾時）／空殼頁（離線））與毫秒、程式檔與資料檔快取數、當季預先下載進度、事件紀錄（最近 50 筆，新的在上；hidden 之後出現一筆不是自己觸發的「頁面載入」就是被系統重載了）。

設定存在 Cache `proto-config`（SW 被回收後仍在），改了立即生效：

- 導覽等待上限 1／2／3／5 秒（預設 3）：頁面載入請求等網路多久，逾時就回存好的空殼頁。
- 人為延遲導覽回應 0／2／5／10 秒（預設 0）：SW 先睡再去抓，用來在 iPhone 上模擬慢網路。
- 模擬離線 開／關：SW 把所有網路請求當成失敗。
- 自動換新門檻 30 分鐘／1 分鐘／10 秒（預設 30 分鐘）。
- 當成獨立視窗 開／關：讓 Chromium 與桌面也走獨立視窗的行為（預先下載、自動換新）。
- 清掉離線資料並註銷：清所有 Cache、註銷 SW、清事件紀錄後重載。

## 設計摘要

- SW（`web/src/service-worker.ts`）：安裝時預存全部程式檔、`shell.html`、隊徽／圖示／manifest／favicon。導覽先上網（逾時前原樣回傳，不存 HTML），逾時或失敗回空殼頁。程式檔等先用存的。`__data.json` 先上網、成功就存（Cache `data-v1`，不跟版本清掉），失敗用存的，都沒有回 503 JSON `{message:'offline-miss'}`。新 SW 不自己接手，等頁面送 `skipWaiting`。
- 頁面（`web/src/lib/proto/`）：自己註冊 SW（Kit 的 `serviceWorker.register` 關掉）；載入與回到前景時抓 `_app/version.json`（`no-store`）並 `registration.update()`；依「頁面是否已是新版／獨立視窗且離開超過門檻／其他」決定直接接手、自動換、或出提示條。
- `web/src/routes/+error.svelte`：收到 503 `offline-miss` 時顯示站內說明頁加返回鈕，其餘維持 Kit 預設錯誤頁樣子。
- 預存清單用 Kit 3 的 `$app/manifest`（`immutable`、`assets`）與 `$app/env`（`version`）；`$service-worker` 在 Kit 3 已移除。
- `web/tsconfig.json` 沒有排除 `src/service-worker`，`npm run check` 對 `service-worker.ts` 會報型別錯（屬原型邊界外，未改）。
