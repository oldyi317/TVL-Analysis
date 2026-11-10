# 套殼技術與 github.io 子路徑站台的相容性
查證日期：2026-11-10　對應票：#185（Part of #182）

版本基準（npm registry 查得，2026-11-10）：@capacitor/core 8.6.0（官方文件頁籤列 v8 與 v9，v8 為預設顯示）、@sveltejs/kit 3.1.1、@sveltejs/adapter-static 4.0.1、@bubblewrap/cli 1.27.0。

## 結論
1. 事實：Digital Asset Links（DAL）檔只認 `https://<host>/.well-known/assetlinks.json`，在網域根目錄；官方沒有任何「子路徑」版本。專案站 `<user>.github.io/<repo>/` 的 origin 是 `<user>.github.io`，驗證檔得放在 `<user>.github.io/.well-known/`，那由 `<user>.github.io` 使用者站 repo 控制，不是專案站 repo。
2. 推論：技術上可行的 github.io 路徑是「另建 `<user>.github.io` 使用者站 repo 放驗證檔」（Bubblewrap 的 `startUrl` 可以是相對於 host 的子路徑）；但若站台之後要搬自訂網域，在有人安裝前搬最省事，因為 DAL 綁 host，換網域等於換一個 app 身分。
3. 事實：Capacitor 官方明寫 `server.url` 是給 live reload、「not intended for use in production」；上架的標準做法是 `webDir` 內嵌（來源見 S12）。內嵌 app 的資料更新靠送審（Capacitor 官方的即時更新方案是 Appflow，付費服務）。
4. 事實：iOS 沒有 TWA 對應物；PWABuilder 的 iOS 路線就是 WKWebView 套殼 Xcode 專案；Apple 審查指南 4.2 要求 app 不得只是「repackaged website」。
5. 觀察（非官方保證）：以三個第三方專案站為樣本實測，設自訂網域後 `<user>.github.io/<repo>/…` 會 301 轉到自訂網域且保留後續路徑與 query；樣本的 Location 為 http 開頭。官方文件沒有承諾這個行為，本專案未實測。

## 逐題回答

### Q1 Android TWA：DAL 驗證檔放哪、專案站子路徑能否通過
- 事實：Chrome 官方整合指南寫「The AssetLinks file must be under /.well-known/assetlinks.json, at the root of the domain, as that's only the place Chrome will look for it.」來源 S1。Digital Asset Links 官方文件稱該路徑是「statement list 的官方名稱與位置」，其他位置或名稱無效。來源 S3。
- 事實：Android App Links 文件寫系統對 intent filter 的每個 hostname 查 `https://hostname/.well-known/assetlinks.json`（只用 hostname，無路徑）。來源 S2。
- 事實：Bubblewrap `twa-manifest.json` 欄位：`host` =「The origin that will be opened in the Trusted Web Activity」，`startUrl` =「The start path for the TWA. Must be relative to the domain.」來源 S5。即驗證單位是 origin（host），startUrl 可以是 host 下的路徑。
- 事實：多來源 TWA 文件寫「each domain to be validated needs its own assetlinks.json」（搜尋摘要轉述；原頁 multi-origin 為 S6，我讀到「origins being opened ... validated using Digital Asset Links」，未逐字核對「each domain」那句）。
- 推論（官方文件沒有直接談 github.io 專案站）：專案站 `oldyi317.github.io/TVL-Analysis/` 的 origin 是 `oldyi317.github.io`，DAL 必須在 `oldyi317.github.io/.well-known/assetlinks.json`。GitHub 文件說使用者站來源檔必須放在名為 `<owner>.github.io` 的 repo（S9），專案站只服務其 repo 子路徑；因此專案站 repo 本身無法提供該檔。實測：`https://twbs.github.io/.well-known/assetlinks.json` 回 404（該帳號有使用者站但沒放此檔，只證明路徑由使用者站控制，不是證明可行）。
- 推論：所以「需不需要使用者站 repo 或自訂網域」的答案是：二選一，必須有一個能控制某 origin 根目錄的東西。使用者站 repo 可行但 origin 變成共用的 `oldyi317.github.io`（該帳號所有專案站共用一個 assetlinks.json，每個 app 的指紋要併在同一檔）；自訂網域最乾淨。
- 未確認：「origin 驗證通過後，startUrl 為子路徑、且 Chrome 對 scope 外導覽的行為」沒有找到官方明文針對子路徑的說明；Bubblewrap 的 `fullScopeUrl` 說明「僅 Meta Quest 使用」（S5）。需實機（Android）驗證，擁有者沒有 Android 實機，可用模擬器。
- 事實：驗證失敗時 app 以 Custom Tab 顯示（有網址列），不會直接壞掉（S4 Quick Start：「If verification fails, the app opens your site in a Custom Tab with browser UI」，經 WebFetch 摘要，原頁未用 curl 逐字核對）。
- 事實：Play 上架時簽章指紋：啟用 Play App Signing 時，上傳金鑰與實際簽章金鑰指紋不同，assetlinks 要放對（S10 Chrome Android 概念頁；搜尋結果也有社群文章印證，僅當線索）。
- 事實：assetlinks 更新延遲：Android 15+ 系統背景定期重驗，最多 7 天傳播；Android 14 以下只在安裝/更新 app 時重驗（S2）。意味著 DAL 一旦出錯或換網域，已安裝使用者最多要等 7 天或更新 app。
- 事實：TWA 內容由使用者的瀏覽器渲染、網頁「must be accessible and useful in the browser first」，且需符合 Add to Home Screen（installability）要求，需要 Web App Manifest（S7）。Lighthouse PWA 類別已移除，改在 DevTools 查（搜尋摘要，來源為 Chrome 官方部落格，未開原文，僅當線索）。Bubblewrap `build` 預設會驗證 Quality Criteria，可用 `--skipPwaValidation` 跳過（S5）。現站沒有 manifest、icon（題目背景），TWA 路線這是必要新增項。
- 事實：PWABuilder 的 Android 打包也要求同樣的 `.well-known/assetlinks.json` 部署在「PWA 的 URL」網域（CloudAPK 說明 S11：`https://example.com/.well-known/assetlinks.json`），PWABuilder 與 Bubblewrap 在這點上沒有放寬。PWABuilder 官方 docs 站是 SPA（curl 與 WebFetch 都只拿到「Loading documentation...」），其 docs 原文「無法核對」。

### Q2 Capacitor：內嵌 vs `server.url`
- 版本：Capacitor 文件 v8（npm latest 8.6.0）。
- 內嵌：`webDir`「should contain the final index.html of your app」（S12）。更新機制：內容隨二進位包出貨；Capacitor 官方稱「realtime updates ... as long as those changes do not require binary updates」並推 Appflow live deploy（S13）。Appflow 是 Ionic 的付費 Mobile DevOps 平台（S13 稱「free to get started」，費率與目前是否仍提供服務未確認）。未使用 Appflow 時，資料要更新就得重 build、重送審。
- `server.url`：S12 寫「Load an external URL in the Web View... intended for use with live-reload servers」、「This is not intended for use in production.」`server.cleartext`、`server.allowNavigation` 也同樣標示非生產用途。Live Reload 指南也是開發情境（S14）。
- 離線行為：內嵌 = 離線可用（檔案在 app 內）。`server.url` = Capacitor 文件沒有給離線機制；殼只是載入遠端頁，離線行為取決於網頁自己的 service worker/快取，官方沒有承諾（未確認，屬推論）。
- 官方對線上載入的立場：事實＝文件標「not intended for production」。Apple 方面事實＝審查指南 2.5.2 要求 app「self-contained in their bundles」，不得「download, install, or execute code which introduces or changes features or functionality」（S15）；Capacitor 自己的說法是 Apple 與 Google「explicitly allow web content updates」（S13）。推論：載入遠端 HTML/JS 且功能不變，審查風險比內嵌高，且 4.2 minimum functionality 的風險同時存在；實際審查結果取決於審查員，查不到官方判例。
- 事實：Capacitor 本機服務的 scheme：Android 預設 `https`（`androidScheme`），iOS 預設 `capacitor`（`iosScheme`）；「Changing this value from anything other than http or https can result in your application unable to resolve routing」（S12）。

### Q3 SvelteKit 靜態站配 Capacitor 的已知限制
- `paths.base`：S16 定義為「root-relative path that must start, but not end with /」，並寫「you need to prepend all your root-relative links with the base value」。事實推論：Capacitor 內嵌時 app 從 `https://localhost/`（Android）或 `capacitor://localhost/`（iOS）的根提供檔案，推論目前的 `paths.base = '/<repo>'` 內嵌版必須改成空字串；也就是需要兩種 build（GitHub Pages 用 base，內嵌用空 base）。Capacitor 文件沒有明文講 `localhost` 根（我讀到的是 scheme 設定，S12），所以「根目錄服務」是推論。
- `relative: false`：目前設定下，輸出的資源引用為根絕對路徑（`/<repo>/_app/...`）。（SvelteKit 配置頁找到 `paths.base` 與 `paths.assets` 說明；`paths.relative` 該頁的逐字內容未讀到，「無法核對」。）
- `trailingSlash = 'always'`：S17（adapter-static）寫在主機不會對 `/a` 提供 `/a.html` 時需設 `trailingSlash: 'always'`。內嵌版 WebView 的靜態伺服（Capacitor 內建 localhost 伺服）是否能把 `/a/` 解析成 `/a/index.html`：Capacitor 文件未找到說明，「未確認」，需實測。
- SPA fallback：S18 寫 `fallback` 頁讓未預渲染的 URL 回該頁，並明說這種模式「large negative performance impact」，但註明對「mobile app that wraps a locally-served SPA」影響較小。現站全預渲染、沒有 fallback 頁，所以內嵌時深連結或重新整理可能 404；加 `fallback: '200.html'` 之類是官方允許的退路（S17、S18）。
- 預渲染路由：S17 要求根 layout `prerender = true` 且不要 `ssr = false`；站台目前符合。
- 資料大小：JSON 約 22 MB、產物約 23 MB 被打進 app（題目背景事實）。商店包大小上限未查，「未確認」。
- 未確認：Capacitor 官方文件頁沒有 SvelteKit 專屬指南（我抓的 environment-setup 與 web 頁未見；其餘頁未全查）。

### Q4 iOS 有沒有 TWA 對應物
- 事實：沒有（在我查到的官方來源裡）。Capacitor iOS「uses WKWebView, not the deprecated UIWebView」（S19）；PWABuilder iOS 平台「generates an iOS app that loads your PWA in a WKWebView」，輸出 Xcode 專案、需在 Mac 編譯上架，且 README 稱「iOS has very limited support for PWAs」並轉為社群維護（S20）。「沒有 TWA 對應物」是否定宣稱，無原文逐字佐證，標推論。
- 事實：iOS 使用者本來就能把網站「Add to Home Screen」，有 manifest 且 `display: standalone/fullscreen` 的網站成為 Home Screen web app（S21 WebKit 部落格，2023-02）。
- 風險事實：App Store 審查指南 4.2：「Your app should include features, content, and UI that elevate it beyond a repackaged website.」4.2.2：除目錄外 app 不應主要是「web clippings」或「a collection of links」；4.2.6：以商業模板或 app 生成服務做成的 app 會被拒，除非由內容提供者直接提交（S15）。推論：純 WKWebView 套站有被拒風險；加原生功能（推播、小工具等）可降低，但這是推論，沒有官方保證。

### Q5 GitHub Pages 換自訂網域：費用、步驟、舊網址轉址
- 費用：GitHub 官方 Pages 文件沒有提到自訂網域本身收費（S9、S22 均未見費用條目；Pages 在公開 repo 的 Free 方案可用，S23）。網域註冊費是註冊商收取，金額「未確認」（沒查註冊商價格）。Pages 限制：站台 ≤ 1 GB、軟頻寬 100 GB/月、部署逾時 10 分鐘（S23）；23 MB 站台遠低於上限，但頻寬用量需留意。
- 步驟（事實，S9、S22）：在 DNS 設 apex（A/ALIAS/ANAME）或 `www` 子網域（CNAME 指到 `<user>.github.io`）；GitHub 建議一律用 `www` 子網域，必要時也設 apex，兩者之間會自動轉址；在 repo 的 Pages 設定填網域；官方指南另有網域驗證與 HTTPS 步驟（未全逐頁讀）。用 Actions 自訂 workflow 部署時，自訂網域要在設定頁設，不是靠 CNAME 檔（S24 對 custom workflow 的說明僅確認「可用 Actions 部署」，CNAME 檔細節「無法核對」）。
- 舊網址轉址：官方文件只寫「apex 與 www 之間自動轉址」，沒寫 `github.io/<repo>` 轉到自訂網域（S9、S22）。實測（2026-11-10，curl -I，第三方站台當樣本）：
  - `https://twbs.github.io/bootstrap/docs/5.3/getting-started/introduction/` → 301 → `http://getbootstrap.com/docs/5.3/getting-started/introduction/`
  - `https://vuejs.github.io/vue-router/guide/?x=1` → 301 → `http://router.vuejs.org/guide/?x=1`
  - `https://reactjs.github.io/react-gh-pages/` → 301 → `http://reactcommunity.org/react-gh-pages/`
  推論：專案站設自訂網域後，舊 github.io 路徑會 301 並保留深層路徑與 query（根路徑到根路徑）；樣本的 Location 為 http 開頭，疑似是這些站設定的歷史遺留或 Pages 對舊網域的行為，最終是否升級為 https 未再追。官方文件沒有此行為保證，不應當作承諾。
- 事實（S9）：使用者/組織站設自訂網域時，同帳號底下沒有自訂網域的專案站會改服務在 `<自訂網域>/<repo>`。因此若走「使用者站 repo + 自訂網域」，專案站會自動變成自訂網域的子路徑，但這會讓 `paths.base` 維持 `/<repo>`；若專案站自己設網域，則站台在網域根，`paths.base` 需改成空字串（推論）。

## 來源列表（皆查證於 2026-11-10；curl 原始頁落檔於 scratchpad/r/）
- S1 Chrome：TWA 整合指南 https://developer.chrome.com/docs/android/trusted-web-activity/integration-guide
- S2 Android：驗證 Android App Links https://developer.android.com/training/app-links/verify-android-applinks
- S3 Google：Digital Asset Links 入門 https://developers.google.com/digital-asset-links/v1/getting-started
- S4 Chrome：TWA Quick Start https://developer.chrome.com/docs/android/trusted-web-activity/quick-start
- S5 Bubblewrap CLI README（原始檔，v1.27.0 時期 main 分支）https://raw.githubusercontent.com/GoogleChromeLabs/bubblewrap/main/packages/cli/README.md
- S6 Chrome：Multi-origin TWA https://developer.chrome.com/docs/android/trusted-web-activity/multi-origin
- S7 Android：TWA 概觀 https://developer.android.com/develop/ui/views/layout/webapps/trusted-web-activities
- S8 Chrome：Android concepts for web devs（Bubblewrap 與 DAL 段）https://developer.chrome.com/docs/android/trusted-web-activity/android-for-web-devs
- S9 GitHub Docs：自訂網域概觀 https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/about-custom-domains-and-github-pages
- S10 同 S8（Play App Signing 與指紋段）
- S11 PWABuilder CloudAPK 說明 https://raw.githubusercontent.com/pwa-builder/CloudAPK/master/Next-steps-unsigned.md
- S12 Capacitor config（v8）https://capacitorjs.com/docs/config
- S13 Capacitor Deploying and Updating https://capacitorjs.com/docs/guides/deploying-updates
- S14 Capacitor Live Reload https://capacitorjs.com/docs/guides/live-reload
- S15 Apple App Store Review Guidelines https://developer.apple.com/app-store/review/guidelines/
- S16 SvelteKit configuration https://svelte.dev/docs/kit/configuration
- S17 SvelteKit adapter-static https://svelte.dev/docs/kit/adapter-static
- S18 SvelteKit single-page apps https://svelte.dev/docs/kit/single-page-apps
- S19 Capacitor iOS https://capacitorjs.com/docs/ios
- S20 PWABuilder iOS README https://raw.githubusercontent.com/pwa-builder/pwabuilder-ios/main/README.md
- S21 WebKit：Web Push for Web Apps on iOS https://webkit.org/blog/13878/web-push-for-web-apps-on-ios-and-ipados/
- S22 GitHub Docs：管理自訂網域 https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site
- S23 GitHub Docs：Pages 限制 https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits
- S24 GitHub Docs：自訂 workflow https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages
- S25 GitHub Docs：關於 Pages（使用者站 vs 專案站表）https://docs.github.com/en/pages/getting-started-with-github-pages/about-github-pages
- 線索（非第一手，未作為事實依據）：Medium「Android App Links: deploy assetlinks.json in minutes」；DEV Community TWA 實作文；Chromium blog TWA quality criteria update（goo.gle/twa-quality-criteria-update）。
- 實測請求：`curl -I` 對上述三個第三方 github.io 專案站與 `twbs.github.io/.well-known/assetlinks.json`（HTTP 狀態與 Location 標頭）。

## 還不知道什麼
- 專案站子路徑 startUrl 在 origin 通過 DAL 後，Chrome TWA 對 scope 外導覽的實際行為（官方無明文），需 Android 實機/模擬器驗證。
- Capacitor 內建 localhost 伺服對 `trailingSlash: 'always'` 預渲染目錄（`/a/` → `/a/index.html`）的處理；`paths.base` 設空字串後內嵌版是否完整運作。
- `paths.relative` 在 SvelteKit 3.1.1 的逐字行為（該頁抓取未讀到）。
- Appflow 目前是否仍運作與定價（只讀到「free to get started」）。
- Capacitor `server.url` 在 App Store / Google Play 的實際審查通過率；沒有官方判例。
- 商店包大小上限與 23 MB 資料內嵌的實際影響；Apple 與 Google 的帳號年費（本次未查，超出票面問題）。
- 自訂網域的註冊費與續約費（依註冊商）；GitHub 轉址是否保證長期存在（官方文件未承諾）。
- PWABuilder 官方 docs（SPA 頁）原文，無法核對。

## 對決策的含意（皆為推論）
1. 若要走 Android TWA，站台一定要有「自己能控制根目錄」的 origin：自訂網域，或 `oldyi317.github.io` 使用者站 repo。安裝量上來之後才換網域，代價是 DAL 重新驗證（最多 7 天傳播）與 app 更新；所以「有人安裝之前先搬」較划算。
2. 自訂網域搬家的成本主要是 `paths.base` 與 DNS：專案站自己設網域後站台在根，base 可改空字串；舊 github.io 網址在第三方樣本上會 301 到新網域（本專案未實測），既有書籤與搜尋引擎連結可能不會立刻失效，但官方未承諾。
3. TWA 是「殼載入線上網址」，資料每日更新不必送審，這是它對本站最合的地方；Capacitor 內嵌路線則每次資料變動都要重送審，與「每天早上更新」衝突，除非 app 啟動時另外 fetch 線上 JSON（站台目前沒有 runtime fetch，需要新增）。
4. Capacitor 的 `server.url` 路線官方不建議用於生產，且在 iOS 上同時踩到 2.5.2 與 4.2 的審查疑慮；iOS 若要上架，較穩的是內嵌加一些原生功能，否則 4.2 風險高。是否值得做 iOS 商店版，建議先看 Android 版的安裝反應。
5. 不管走哪條路，現站缺的 manifest、icon、installability（TWA 的前置條件）都要補；這件事對 iOS「加入主畫面」與 Android Chrome 安裝也都有益，可先做、不綁定決策。
6. 內嵌 Capacitor 要維護兩種 build（base 與空 base）、且 22 MB JSON 進包；若決定走 TWA 為主，這些成本可全部避開。
