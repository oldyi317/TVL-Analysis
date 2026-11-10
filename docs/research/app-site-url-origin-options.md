# 靜態站「專屬、長期不變 origin」選項研究（查證日 2026-11-10）

原文落檔：`scratchpad/raw/`（curl 抓 HTML、剝標籤後全文檢索；`*.html.txt` 為剝標籤版）。
標示：【事實】=讀到原文；【觀察】=本次 curl 實測；【推論】=我的判斷；【未確認】=沒讀到或無法核對。

---
## 1. GitHub Pages 專案站設自訂網域

### 1a 內容是否改由網域根目錄提供（base path 變空字串）
- 【事實】GitHub 文件的「Types of sites」表：專案站預設位置為 `http(s)://<owner>.github.io/<repositoryname>`。來源 https://docs.github.com/en/pages/getting-started-with-github-pages/about-github-pages（我抓回的是殼，WebFetch 版與搜尋結果一致；原文表格段落在 what-is-github-pages 頁抓到，見 raw/what-is.html.txt）。
- 【事實】自訂網域文件示範的 DNS 都是把網域指向 `<user>.github.io`，且明寫「CNAME 記錄要直接指向 `<user>.github.io`，不含 repository 名稱」。來源 https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site （2026-11-10）。
- 【觀察】上述 bootstrap 例（1b）：`twbs.github.io/bootstrap/docs/` 轉到 `getbootstrap.com/docs/`，即專案站內容在自訂網域是根路徑 `/docs/`，不是 `/bootstrap/docs/`。
- 【推論】專案站掛自訂網域後，內容在網域根層級，SvelteKit `paths.base` 應改為空字串。GitHub 文件我讀到的範圍內沒有一句直接寫「base path 變空」，是由上面三點推得。

### 1b 舊 github.io/<repo>/ 是否轉址（本題最關鍵）
- 【觀察，非官方保證】2026-11-10 對公開專案站唯讀實測，均回 `HTTP/2 301`、`server: GitHub.com`，且「保留路徑」：
  ```
  $ curl -sI https://twbs.github.io/bootstrap/docs/
  HTTP/2 301
  server: GitHub.com
  location: http://getbootstrap.com/docs/
  $ curl -sI https://twbs.github.io/bootstrap/
  HTTP/2 301
  location: http://getbootstrap.com/
  $ curl -sI https://jashkenas.github.io/underscore/
  HTTP/2 301
  location: http://underscorejs.org/
  $ curl -sI https://jashkenas.github.io/backbone/
  HTTP/2 301
  location: http://backbonejs.org/
  $ curl -sI https://hakimel.github.io/reveal.js/
  HTTP/2 301
  location: http://revealjs.com/
  ```
  另 vuejs/vue-router、reactjs/react-gh-pages 也回 301（未貼 location）；lodash/mocha 等回 404（可能未設自訂網域，不是反例）。
- 【觀察】`Location` 的 scheme 是 `http://`（再由新網域轉 https），這些專案多半是老舊、以分支發佈，不一定是 Actions artifact 部署。【未確認】Actions 部署（artifact）的專案站是否同行為；本次沒找到可確認為 artifact 部署的公開例子。
- 【事實】官方文件我檢索過的頁面（about-custom-domains、managing-a-custom-domain、troubleshooting-custom-domains、verifying、securing-with-HTTPS、github-pages-limits，全文已剝標籤檢索 "redirect"）只提到 apex↔www 的自動轉址；對「舊 github.io/<repo> 轉到自訂網域」我檢索的範圍內沒找到官方保證文字。限定範圍如上，不等於官方沒有寫。
- 【事實，旁證】GitHub 有 issue 指稱官方自訂網域文件對轉址描述不正確或過時（https://github.com/github/docs/issues/39087，我只看到標題，內容無法核對）。
- 【事實，旁證】第三方文章說會自動轉址（posit-dev great-docs recipe），Stack Overflow 鏡像說只有使用者站會自動轉、專案站要 JS 轉址——兩者互相矛盾，【觀察】結果支持前者。
- 證據等級：【觀察】5 個站一致（301＋保留路徑），無官方文件原文保證。不要把它當 SLA。

### 1c HTTPS 憑證與 DNS 紀錄
- 【事實】憑證：設定後 GitHub 檢查 DNS，成功即向 Let's Encrypt 申請 TLS 憑證並自動上傳到 Pages 的 TLS 終結伺服器。來源 https://docs.github.com/en/pages/getting-started-with-github-pages/securing-your-github-pages-site-with-https
- 【事實】若有 CAA 紀錄，至少要有一筆值為 `letsencrypt.org`。來源 https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/troubleshooting-custom-domains-and-github-pages
- 【事實】apex：A 記錄指向 185.199.108.153／.109.153／.110.153／.111.153；AAAA 為 2606:50c0:8000::153、8001、8002、8003；或用 ALIAS／ANAME 指向 `<user>.github.io`。www／子網域：CNAME 指向 `<user>.github.io`。「強烈建議不要用萬用字元 DNS（`*.example.com`）」（domain takeover 風險）。來源 managing-a-custom-domain 頁。
- 【事實】官方建議：用 apex 時「建議也設 www」；「We recommend always using a www subdomain, even if you also use an apex domain」。設好兩者的 DNS 後 GitHub 自動在 apex 與 www 之間轉址。來源 https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/about-custom-domains-and-github-pages

### 1d 網域驗證與 CNAME 位置
- 【事實】驗證：在帳號（或組織）設定的 Pages > Add a domain，依指示建立 DNS TXT 紀錄，驗證後該網域及其「直接子網域」都受保護，防止其他 GitHub 使用者把它拿去發佈自己的 Pages；TXT 要一直保留。驗證可防的情境含刪 repo、降級方案、解除連結但 DNS 仍指向 Pages。來源 https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/verifying-your-custom-domain-for-github-pages
- 【事實】Actions 自訂 workflow 部署：「no CNAME file is created, and any existing CNAME file is ignored and is not required」——自訂網域在 repo Settings > Pages 的 Custom domain 欄設定；分支發佈才會在 source 分支根目錄產生 CNAME 檔。來源 managing-a-custom-domain 頁（2026-11-10 抓到）。
- 【事實】網域在所有 Pages 站間必須唯一（troubleshooting 頁）。

### 1e DNS 放 Cloudflare（免費）
- 【事實，Cloudflare】已代理（proxied）紀錄的 DNS 查詢回 Cloudflare anycast IP，DNS only 回原值。來源 https://developers.cloudflare.com/dns/proxy-status/
- 【事實，Cloudflare 搜尋摘要】「too many redirects」排錯頁：加密模式為 Flexible 時 CF 以 HTTP 連源站，源站又導 HTTPS 就迴圈，需改 Full 以上。來源 https://developers.cloudflare.com/ssl/troubleshooting/too-many-redirects/ （只讀到搜尋摘要，原文本次未落檔，標【未核對原文】）。
- 【推論】GitHub Pages 源站會強制 HTTPS，所以 proxied 時要用 Full（非 Flexible）。
- 【事實】GitHub 官方文件（上列 6 頁全文檢索 "proxy/proxied/CDN"）沒有關於 Cloudflare 代理的指引；【推論】最穩的是 DNS only（灰雲），讓 GitHub 直接以 Let's Encrypt 發憑證，避免 HTTP-01 驗證被代理干擾。第三方部落格（medium、engineered.at）說先 DNS only 等憑證簽發完再開代理；這是第三方，【未核對官方】。
- 【推論】如果之後要用 Cloudflare Routes 擋 `<domain>/api/*`（見題 6），該主機名必須 proxied，與灰雲建議衝突，需 Full 模式並接受憑證續期風險。

---
## 2. GitHub Pages 用量限制與使用條款（現行文字）
來源 https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits （2026-11-10 抓取原文）：
- 【事實】發佈後站台 ≤ 1 GB；repo 建議 ≤ 1 GB；部署超過 10 分鐘逾時；軟性頻寬 100 GB／月；軟性 10 builds／小時，「若用自訂 GitHub Actions workflow 建置發佈則不適用此限制」；可能觸發 rate limit（回 429）。
- 【事實】用途限制：「not intended for or allowed to be used as a free web-hosting service to run your online business, e-commerce site, or any other website that is primarily directed at either facilitating commercial transactions or providing commercial software as a service (SaaS)」；不得用於傳送密碼、信用卡等敏感交易。
- 【推論】117 MB／4,080 檔、每天一次部署都在限額內；賽事統計站非交易／SaaS，不構成違規（未讀到對「推播」的限制，【未確認】）。
- 【事實】每個帳號只能有一個使用者或組織站（同頁）。

---
## 3. 免費根層級 origin 替代方案

### 3a 另開 GitHub 帳號／組織的 `<name>.github.io`
- 【事實】`<owner>.github.io` repo 的使用者／組織站在根路徑，每帳號一個（what-is / limits 頁）。
- 【事實】`actions/deploy-pages` README（https://raw.githubusercontent.com/actions/deploy-pages/main/README.md ）：job 需 `pages: write`、`id-token: write`，用 `github.token` 預設，部署目標為 `github-pages` environment；OIDC token 帶著「正在跑的 workflow 的 ref」並由 Pages deployment API 驗證。README 全文我檢索 repo/cross/another/different，沒有任何跨 repo 部署的說法或輸入參數（無 `repository` 之類）。
- 【推論】deploy-pages 綁定「執行 workflow 的 repo」自己的 Pages，不能直接部署到另一個 repo；要跨 repo 得把產物 push 到目標 repo 的分支，再讓目標 repo 用分支來源或自己的 workflow 發佈。【未確認】官方是否有明文禁止。
- 【事實】`peaceiris/actions-gh-pages`（第三方，非 GitHub 官方）支援 `external_repository: <user>/<repo>`，認證三擇一：`deploy_key`（SSH，需設）、`personal_token`（HTTPS，需設）；`github_token` 只對同 repo（README 表格 52–54 行、390–402 行）。來源 https://github.com/peaceiris/actions-gh-pages
- 【推論】做法＝原 repo workflow build → 用 deploy key（單 repo 最小權限，推薦）或 fine-grained PAT 推到 `<name>/<name>.github.io` 的某分支 → 目標 repo 以分支來源發佈（或在目標 repo 放 workflow 以 artifact 部署）。多一個要維護的憑證（deploy key 不過期，PAT 會過期）。
- 【風險，推論】`<name>.github.io` 的 origin 屬於 GitHub，改名帳號或搬家就換 origin；根目錄 `/.well-known/assetlinks.json` 可放（在該 repo 根目錄，且 Jekyll 預設不發佈 dot 目錄，要有 `.nojekyll` 或 artifact 部署＝不經 Jekyll，【推論】）。

### 3b Cloudflare Pages `<project>.pages.dev`
- 【事實】Free：500 builds／月（Git 整合建置）、1 build 同時、build 20 分鐘逾時；**每站最多 20,000 檔**（付費 100,000，需 `PAGES_WRANGLER_MAJOR_VERSION=4`）；**單檔 25 MiB**；每專案自訂網域 100。來源 https://developers.cloudflare.com/pages/platform/limits/ （頁面標示 Last updated Sep 5, 2026，2026-11-10 抓取）。
- 【事實】頻寬：我在該限額頁全文檢索 "bandwidth" 沒有命中，限額頁沒有頻寬上限；【未確認】是否有公平使用等其他條款。
- 【事實】Direct Upload：`npx wrangler pages deploy <BUILD_OUTPUT_DIRECTORY>`，產物在 `<PROJECT_NAME>.pages.dev`。官方有 GitHub Actions 範例（需 API Token 與 Account ID 當 secrets）。來源 https://developers.cloudflare.com/pages/get-started/direct-upload/ 、 https://developers.cloudflare.com/pages/how-to/use-direct-upload-with-continuous-integration/
- 【推論】117 MB／4,080 檔在限額內（<20,000 檔，單檔需確認 <25 MiB；總大小在 Pages 限額頁沒有上限，【未確認】）。Direct Upload 不走 Git 建置，【推論】不吃 500 builds／月。
- 【事實，定位】Pages 概覽頁：「Workers supports most Pages use cases… It is Cloudflare's primary platform for building applications. Start new projects with Workers.」；導覽列有「Migrate to Workers」。來源 https://developers.cloudflare.com/pages/ （Last updated Aug 25, 2026）、https://developers.cloudflare.com/workers/static-assets/migration-guides/migrate-from-pages/ 。
- 【推論】Pages 仍可用但官方已建議新專案改用 Workers Static Assets；Pages 的長期支援風險較高（未讀到停用日期，【未確認】）。

### 3c Workers Static Assets `<name>.<account>.workers.dev`
- 【事實】Workers Free：每個 Worker 版本 **20,000 檔**（Paid 100,000；Wrangler 需 v4）；**單檔 25 MiB**。來源 https://developers.cloudflare.com/workers/platform/limits/ （Last updated Oct 29, 2026）。
- 【事實】「Requests to static assets are free and unlimited」；只有打到 Worker 腳本的請求才算 Workers 計費。來源 https://developers.cloudflare.com/workers/static-assets/billing-and-limitations/ 、 https://developers.cloudflare.com/workers/platform/pricing/ 。
- 【事實】Free 方案 Worker 請求 100,000／日；若設 `run_worker_first` 讓某些路徑先進 Worker，該路徑每次請求都算，超限回 429，不會退回靜態資產（billing 頁）。
- 【推論】純靜態站＋另一個 Worker 發推播，靜態請求不吃 100k 額度；推播 Worker 的請求才算。
- 【事實】`workers.dev` 子網域：每個 Worker 自動建立，可停用；官方建議「production Workers 跑在 Routes 或 Custom Domain 上，而不是 workers.dev」（routing 頁 https://developers.cloudflare.com/workers/configuration/routing/ ）。
- 【未確認】workers.dev 帳號子網域更名後舊 origin 是否保留（會掉使用者的風險點）；本次未查到。

### 3d Public Suffix List
- 【事實】PSL 原始檔（https://publicsuffix.org/list/public_suffix_list.dat ，VERSION 2026-11-09_13-00-17_UTC，COMMIT ab73a956…）：
  - `// Cloudflare, Inc.` 區塊含 `pages.dev`、`workers.dev`、`r2.dev`（第 12766–12775 行）。
  - `// GitHub, Inc.` 區塊含 `github.io`、`githubusercontent.com`、`github.app`（第 13841–13846 行）。
- 【事實】PSL 說明：原用於讓瀏覽器拒絕在過寬網域設 cookie（如 `co.uk`），是跨廠商由 Mozilla 志工維護的後綴清單。來源 https://publicsuffix.org/learn/ 。
- 【事實】origin 以 scheme＋host＋port 為準，每個 `user.github.io` 本來就是不同 origin，儲存空間與 SW 彼此隔離。來源 MDN https://developer.mozilla.org/en-US/docs/Web/Security/Same-origin_policy 、 https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria 。
- 【推論】PSL 影響的是「site」（eTLD+1）層級行為：cookie 作用域、SameSite 判斷、瀏覽器把 `a.github.io` 與 `b.github.io` 視為不同 site、地址欄醒目顯示域名等。它不改變 origin 隔離本身；對本案（無 cookie、靠 origin 儲存與推播）沒有實際差異。

---
## 4. 同一 `<user>.github.io` 多專案站共用什麼
- 【事實】MDN：瀏覽器「per origin」管理資料；配額與驅逐準則適用於整個 origin，即使該 origin 跑多個網站，如 `https://example.com/site1/` 與 `/site2/`；涵蓋 localStorage、IndexedDB、Cache API、OPFS 等。來源 MDN Storage quotas and eviction criteria（上列）。
- 【事實】MDN：兩 URL 同 origin 的條件是 protocol、port、host 相同（路徑不計）。來源 Same-origin policy（上列）。所以 `<user>.github.io/a/` 與 `/b/` 可互相讀寫對方的 localStorage、IndexedDB、Cache（Cache 名稱可撞名）。
- 【事實】Service worker scope：預設為 SW 腳本所在目錄；scope 應是同目錄或更深；需更廣 scope 要靠 `Service-Worker-Allowed` HTTP 標頭。來源 MDN https://developer.mozilla.org/en-US/docs/Web/API/ServiceWorkerContainer/register 、 https://developer.mozilla.org/en-US/docs/Web/API/ServiceWorkerRegistration/scope 。
- 【推論】專案站的 SW 放 `/<repo>/sw.js`，scope 最多 `/<repo>/`（GitHub Pages 無法自訂標頭），所以不會攔截同 origin 其他專案，但同 origin 的其他專案站也能註冊自己的 SW 與讀你的儲存。
- 【事實＋推論】通知權限：MDN Notifications 指南說明需使用者授予「站台／應用」權限（https://developer.mozilla.org/en-US/docs/Web/API/Notifications_API/Using_the_Notifications_API ），MDN 本頁沒有明寫「以 origin 為單位」；【推論】依 Permissions 規格權限為 origin 範圍，故同 origin 的所有專案共用一次授權。【未核對規格原文】。
- 【推論】對本案：最終網址沒變之前不能說「路徑隔離就安全」；共用 origin＝共用配額與權限，換 origin 則全部歸零。

---
## 5. 通知來源顯示（主機名稱會不會出現）
- iOS／iPadOS 主畫面 web app：
  - 【事實】WebKit：加入主畫面的 web app 可在使用者互動後請求權限，iOS 會提示；允許後在「通知設定」可「per web app」管理；通知與其他 app 一樣出現在鎖定畫面、通知中心。來源 https://webkit.org/blog/13878/web-push-for-web-apps-on-ios-and-ipados/ （2026-11-10 抓取）。
  - 【事實】同文：使用者加入主畫面時可改名；iOS 以 Manifest `id` 識別 web app（用於跨裝置同步專注模式）。
  - 【未確認】提示與通知是否顯示 `<user>.github.io` 主機名稱：WebKit 文章與我的兩次搜尋都沒有講到；要靠實機測試，無法核對。
- Android Chrome：
  - 【事實，搜尋摘要】Chromium 2023 年底變更：已安裝 web app 的通知改顯示 web app 自己的名稱與圖示，而不是 Chrome 的；非安裝的分頁式網站仍可能顯示 Chrome 標示。我只讀到搜尋摘要（來源 https://mspoweruser.com/google-is-simplifying-how-pwas-for-chrome-work-adding-better-notification-system ，二手），【未核對 Chromium 原文】。
  - 【未確認】權限提示是否列出主機名稱（常見實作會顯示網站 origin，但我沒讀到官方原文）；通知小字是否顯示主機名稱同樣未確認。
- 【推論】想避免外露 `github.io`，用自訂網域是最直接的做法；但這要等實機驗證才能確定「顯示與否」。

---
## 6. 自訂網域 DNS 在 Cloudflare 免費方案時，把子網域或路徑交給 Worker
- 子網域（如 `push.<domain>`）：
  - 【事實】Workers Custom Domains：需 active Cloudflare zone＋Worker；Cloudflare 自動建 DNS 紀錄並發憑證；整個網域／子網域的所有路徑都指向該 Worker；「不能在已有 CNAME 紀錄的主機名上建 Custom Domain」。來源 https://developers.cloudflare.com/workers/configuration/routing/custom-domains/ （Last updated Sep 29, 2026）。
  - 【推論】`push.<domain>` 與 apex/www 指向 GitHub Pages 並存沒有衝突（不同主機名）。但推播送出端是 Worker 呼叫推播服務，瀏覽器不需要打 `push.<domain>`，除非要用來收訂閱；若網頁在 `<domain>` 而 API 在 `push.<domain>`，是跨 origin，要 CORS。
- 路徑（如 `<domain>/api/*`）：
  - 【事實】Workers Routes：需 active zone、Worker、且該網域／子網域要有「proxied（橘雲）」DNS 紀錄；沒有 DNS 紀錄會 `ERR_NAME_NOT_RESOLVED`。來源 https://developers.cloudflare.com/workers/configuration/routing/routes/ （Last updated Oct 18, 2026）。
  - 【推論】指向 GitHub Pages 的 apex／www 要改為橘雲才能用路徑路由，需 SSL 模式 Full（見 1e）。
- 免費方案：
  - 【事實】Workers limits 頁「Routes and domains」列出 routes per zone 1,000、custom domains per zone 100，未區分 Free／Paid；Workers Free 為 100,000 請求／日。來源 https://developers.cloudflare.com/workers/platform/limits/ 。【推論】Free 可用，路由進入 Worker 的請求才計入 100k。【未確認】我沒查到「免費方案不可用 Routes」的限制，亦無明文保證可用。

---
## 還不知道什麼（彙整）
1. Actions 部署的專案站，舊 github.io/<repo> 301 轉自訂網域是否有官方保證（只有 5 個舊站的觀察，且 Location 為 http）。
2. 設自訂網域後舊 URL 的 Service worker／已安裝 PWA、推播訂閱不會跟著搬——origin 改變就掉；官方沒有搬遷機制（推論）。
3. Cloudflare Pages 頻寬／總大小是否有限制（限額頁沒有，需官方確認）。
4. Pages 的停用時程。
5. workers.dev 子網域更名後舊 URL 行為。
6. 權限提示與通知是否顯示主機名稱（iOS、Android 都未確認，需實機）。
7. Cloudflare 橘雲＋GitHub Pages 的憑證續期在 Full 模式下的實際穩定度（只有第三方說法）。
8. 網域年費（未查證；一般 .com 約每年 10–15 美元，屬常識性估計，非本次查證）。

---
## 比較表
| 項目 | A 留在子路徑 `<user>.github.io/<repo>/` | B 自訂網域＋GitHub Pages | C1 GitHub 另開帳號 `<name>.github.io` | C2 Cloudflare（Workers Static Assets，Pages 次選） |
|---|---|---|---|---|
| 年費 | 0 | 網域年費（未查證，約 10–15 美元估計）；GitHub／Cloudflare DNS 免費 | 0 | 0（Free 方案） |
| 是否擁有 origin | 否（GitHub 擁有 `github.io`，且與同帳號專案共用 origin） | 是（網域自己持有，之後可換主機） | 否（`<name>.github.io` 屬 GitHub，改帳號名會變） | 否（`workers.dev`/`pages.dev` 屬 Cloudflare） |
| TWA assetlinks.json | 不可（根目錄 `/.well-known/` 屬使用者站，不屬專案站；推論） | 可（網域根目錄；Pages 站內放 `.well-known/` 即可） | 可（站在根目錄；注意 Jekyll 不發佈 dot 目錄，artifact 部署則無此問題，推論） | 可（根路徑，需放進靜態資產） |
| 部署流程要改什麼 | 不用改；PWA 要處理 base `/<repo>` | `paths.base` 改空字串、在 repo Settings 設 Custom domain、DNS 設 A/AAAA 或 CNAME、TXT 驗證；deploy-pages 不用改 | 新帳號建 repo、deploy key／PAT、改成跨 repo 推送或兩段 workflow、base 改空字串 | 改 workflow 用 wrangler（API Token＋Account ID）、base 改空字串、`wrangler.jsonc` assets 設定 |
| 主要風險 | 換 origin 會掉安裝與推播授權；與同帳號其他站共用權限與儲存；無 TWA 驗證檔位置 | 網域續費與忘記續費；Cloudflare 橘雲＋GitHub 憑證互動；舊網址轉址無官方保證（觀察 301） | 憑證管理；deploy-pages 不能跨 repo（推論）；origin 不屬於自己 | Pages 官方已不建議新專案；workers.dev 子網域變更風險；未確認的主機名顯示 |

一句話總結：只有「自訂網域＋GitHub Pages」能同時做到擁有 origin、TWA 驗證檔可放、部署流程改動最小（base 改空字串＋Settings 填網域＋DNS），代價是每年網域費；免費根層級方案（GitHub 組織站或 Cloudflare）技術上都可行但 origin 不是自己的，之後搬家一樣掉使用者。
