---
status: accepted
date: 2026-11-12
---

# 站台搬到自己持有的網域 vstat.tw，主機仍是 GitHub Pages

站台原本在 `https://oldyi317.github.io/TVL-Analysis/`。接下來要做成可安裝網站並加上推播（spec #231）。球迷把本站加到主畫面、登記推播訂閱時，瀏覽器認的是 origin：網址一換，裝過的要重裝，推播訂閱全部作廢。舊網址綁著 GitHub 帳號名、repo 名與 GitHub Pages 三件事，任何一個變了 origin 就跟著變。所以在任何可安裝或推播功能上線之前，先把站台搬到一個自己持有、換主機也不變的正式網址。選項：(a) 留在 `github.io` 子路徑；(b) 換到廠商給的免費根層級網址（GitHub 組織站 `<name>.github.io`、Cloudflare 的 `workers.dev`／`pages.dev`）；(c) 買網域，主機不換。決定 **(c)**。

- 正式網址是裸網域 `https://vstat.tw`。`www.vstat.tw` 只轉址到裸網域、不提供內容；兩者是不同的 origin，安裝與推播訂閱只認裸網域。
- 主機仍是 GitHub Pages，只換網址。站台內容在網域根目錄，`paths.base` 是空字串。站台用 Actions 部署，不加 CNAME 檔，網域填在 repo 的 Pages 設定。
- DNS 放 Cloudflare（免費方案），網站本體的紀錄不經 Cloudflare 代理，讓 GitHub 直接簽發與續期憑證。DNSSEC 不開。
- 註冊商是 PChome 買網址（Cloudflare 的註冊服務不賣 `.tw`）。

理由：(a) 除了綁死帳號名、repo 名與主機，Android 商店版的驗證檔只認網域根目錄，子路徑放不了；同帳號的其他專案站還會共用儲存配額與通知權限。(b) 免費、也在根目錄，但網址仍屬於廠商，之後換主機一樣要球迷重裝。頂級網域的取捨：`.tw` 優先、`.com` 備案，名稱「VStat 企排」定案時 `vstat.com` 已被註冊、`vstat.tw` 買得到；`.app`／`.dev` 的註冊局合約沒有漲價上限，不考慮（`.tw` 有 TWNIC 訂的每年 NT$800 上限）。

## Consequences

- 每年多一筆固定費用：`.tw` 在 PChome 的續用費單年 NT$720，多年期每年更低。
- 網域過期等於整站與所有推播訂閱一起失效：`.tw` 到期翌日停止解析，30 天內補繳可恢復，逾期釋出。PChome 後台沒有自動續約，所以推播上線前把網域續滿 5 年（#250），另有每週自動查到期日的 workflow，距到期少於 90 天就開 issue（#233）。註冊商帳號的 Email 不依賴這個網域。
- 舊網址 `oldyi317.github.io/TVL-Analysis/…` 的轉址沒有官方保證：實測其他掛了自訂網域的專案站，深層路徑都回 301 且保留路徑，但 GitHub 的文件沒有寫。沒轉也接受，不另外架轉址，別處貼過的連結不追。
- 搬家當天不做零停機：`paths.base` 是寫死的，不為一次性的搬家加切換機制。新版部署完成到 repo 填好網域、憑證簽發之間，站會短暫打不開，以一小時內為目標；退回的做法是拿掉 repo 的網域設定，再 revert 搬家的 commit 重新部署。
- `_github-pages-challenge-oldyi317` 那筆 TXT 紀錄要一直留著：它是 `oldyi317` 帳號的網域驗證，拿掉之後別人可以把這個網域掛到自己的 Pages 站。
- 正式站在網域根目錄，預覽站仍在子路徑：站內連結、manifest 與 service worker 的範圍一律從 `base` 推導，不因為正式站的 `base` 是空字串就寫死 `/`。
- 發送服務之後放在自己的子網域（`push.vstat.tw`，經 Cloudflare），不掛在網站的路徑下；網站本體的紀錄維持不經代理。
