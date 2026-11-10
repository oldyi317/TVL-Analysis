# 可安裝網站（PWA）在 iOS 與 Android 的安裝、推播與離線能力

查證日期：2026-11-10　對應票：#183（Part of #182）
標記說明：【事實】= 讀到原文；【推論】= 由事實推出或無原文佐證；【未確認】= 沒查到或無法核對。

## 結論（≤8 行）
1. iPhone 做得到「安裝 + 離線 + 推播」，不經商店、不需 Apple 開發者帳號；條件是 iOS/iPadOS 16.4 以上，且推播要由「已加到主畫面的網站」在使用者點擊時請求（WebKit 官方說法）。
2. iPhone 安裝只能使用者手動（分享選單「加到主畫面」），站台無法彈出安裝提示；Android Chrome 可用 beforeinstallprompt 自訂安裝按鈕。
3. iOS 26 起任何加到主畫面的網站預設以 web app 開啟，安裝不再需要 manifest（Safari 26.0 官方文章）。
4. 23 MB 快取：WebKit 官方給的單一來源配額為磁碟 15%–60%（視 app 而定），Chrome 為 60%；量級上遠低於上限，可行。但仍屬 best-effort，系統儲存壓力時可被 LRU 清除。
5. 子路徑 `/<repo>/` 可用：service worker 預設 scope 是腳本所在目錄，放在 `/<repo>/` 下即可涵蓋整個子路徑，無需特殊標頭。
6. 識別綁定 origin（含 manifest id 的解析基準）；換網域即視為不同 app，已安裝者與推播訂閱不會自動遷移（部分為推論，見 Q3）。

## 逐題回答

### Q1 安裝流程與站台能否主動提示
- iOS（Safari）【事實】：使用者點分享鈕 → 「加到主畫面」。WebKit 原文：「users of Safari on iOS and iPadOS could do this by tapping the Share button to open the Share menu, and then tapping 'Add to Home Screen'」。（來源 S1，2026-11-10）
- iOS 26 / iPadOS 26【事實】：「By default, every website added to the Home Screen opens as a web app」，使用者可在加入時關閉「Open as Web App」；原文「there are now zero requirements for 'installability' in Safari」。（S2）
- iOS 站台能否主動提示安裝：我讀到的 WebKit 文章只描述使用者手動流程，沒有任何站台觸發安裝的 API；beforeinstallprompt 為 Chromium 系事件。結論「iOS 站台不能主動彈出安裝提示」為【推論】（無原文明寫「不支援」）。實務上只能在頁面放教學文字。
- Android Chrome【事實】：符合安裝條件時瀏覽器觸發 beforeinstallprompt，站台可保存事件並在自己的按鈕中呼叫 prompt()（S3 web.dev、S4 MDN）。條件（Chromium 系含 Chrome/Samsung Internet/Edge）：manifest 含 name 或 short_name、192px 與 512px 圖示、start_url、display 與/或 display_override、prefer_related_applications 非 true；且需 HTTPS（S5 MDN）。web.dev 註明該事件已移到獨立 incubator 規格，Chrome 團隊「沒有移除或棄用計畫」（S3）。
- 各平台差異：Android 有自動提示與自訂提示；iOS 純手動。Android 需 manifest 才合格，iOS 26 不需。
- 瀏覽器自身的安裝 UI 因瀏覽器與平台而異（S5 MDN）。

### Q2 Web Push
- iOS 最低版本【事實】：iOS/iPadOS 16.4（2023-02-16 beta 公告：「iOS and iPadOS 16.4 beta 1, and with it comes support for Web Push… for Home Screen web apps」）。（S1）
- 是否必須先安裝：原文「A web app that has been added to the Home Screen can request permission to receive push notifications as long as that request is in response to direct user interaction — such as tapping on a 'subscribe' button」（S1）。Apple 文件頁（developer.apple.com）我抓到的 HTML 無法檢索到對應文字【無法核對】。「Safari 分頁內不能推播」屬【推論】；iOS 26 後此規則是否改變【未確認】（Safari 26.0 文章只在 Web Inspector 段提到 Web Push 事件，未談規則變更，S2）。
- 權限請求限制【事實】：需明確使用者手勢；訂閱必須 userVisibleOnly: true，且每則推播都必須顯示通知；違反會使訂閱被撤銷（S6 Meet Web Push）。不需 Apple Developer Program；若自管推播伺服器需放行 *.push.apple.com（S1）。使用者在系統通知設定逐 app 管理，並整合專注模式（S1）。
- 徽章【事實】：iOS 16.4 起支援 Badging API（S1）。
- Declarative Web Push【事實】：iOS 18.4 / iPadOS 18.4 可測；免 service worker 即可請求訂閱並顯示通知，訊息格式含 `"navigate"`（點擊後開啟的 URL）、`app_badge` 等；向下相容既有 Web Push（S7）。
- 點擊開到指定頁：標準 Web Push 在 service worker 的 notificationclick 事件中呼叫 clients.openWindow(url)（S8 MDN，範例為 `openWindow("/")`）；iOS 上此行為我沒有逐版本的 iOS 原文，標為【未確認】（Declarative 版本的 navigate 欄位為官方設計，S7）。
- Android Chrome：標準 Push API + Notifications API；MDN 的 Notification.requestPermission 範例以點擊事件觸發（S9）。「Android Chrome 不必安裝即可收推播」我沒讀到明確原文，【未確認】。Android 最低 Chrome 版本【未確認】。
- Android 權限回收【推論，來源為二手報導】：2025-10 報導稱 Chrome（Android 與桌面）會自動撤銷低互動網站的通知權限，且已安裝的 web app 排除；Chrome 官方原文未能取得（S10 為搜尋結果，僅線索）。

### Q3 安裝與推播訂閱綁什麼
- manifest id【事實】：MDN：id 必須與 start_url 同源；未指定或無效時以 start_url 為準；解析基準是 start_url 的 origin（S11）。Chrome：加了 id 之後，start_url 與 manifest 路徑可改，「if and only if their particular origin stays the same」，瀏覽器以 id 認定同一個 PWA（S12，Chrome 96 起）。
- Chromium 內部文件【事實】：App ID 由 ManifestId 雜湊而來；Origin 為「app 鎖定的網站來源」（S13）。
- iOS【事實】：iOS 16.4 起支援 manifest id，用於跨裝置同步專注模式設定；並與使用者可改的名稱合併來唯一識別 web app，允許同一網站多份安裝（S1）。
- 推播訂閱：由 ServiceWorkerRegistration 建立 PushSubscription（S7），service worker 註冊以 origin + scope 為界。
- 換網址/子路徑後的後果【推論】：(a) 換 origin（例如從 github.io 子路徑搬到自訂網域）＝新 origin，視為新 app；舊安裝與舊訂閱不會自動帶過去，需使用者重新安裝並重新同意推播。(b) 同 origin 換子路徑：service worker scope 與 start_url 改變，舊 scope 的 SW 與訂閱不涵蓋新路徑；在 Chrome 若有穩定 id 可保住安裝識別，但訂閱仍屬舊 registration。以上均無官方「遷移行為」原文，建議一開始就固定最終網址與 id。
- 同一 github.io 網域下所有專案站共用 origin（`https://<user>.github.io`）：【推論】儲存與權限的 origin 層級隔離只到網域，不到子路徑；其他專案站理論上可讀同 origin 的儲存。未查原文，【未確認】。

### Q4 離線儲存配額與清除規則
- WebKit（Safari 17.0 起）【事實】：單一來源配額，瀏覽器 app 為磁碟 60%，其他 app 15%；整體配額 80% / 20%；主畫面 web app 與在瀏覽器開啟時享有相同配額；超額寫入拋 QuotaExceededError（S14 WebKit storage 文章）。Safari 17 不再提示使用者加配額（同文）。
- 清除規則【事實】：在超出整體配額、系統儲存壓力、或「久未互動」時可被逐 origin 清除，採 LRU，最後使用時間＝最近一次使用者互動或儲存操作；預設 best-effort；persist() 的授予依啟發式，如「是否以主畫面 web app 開啟」（S14）。
- ITP 7 天上限【事實】：Safari 對腳本可寫儲存設 7 天（Safari 使用日）上限（S15）；但「主畫面 web app 的第一方網域不受 ITP 7 天上限約束，且資料與 Safari 隔離」（S16 WebKit tracking-prevention 頁）。意即：在 Safari 分頁使用的離線快取可能被 7 天規則清掉，安裝為主畫面後不受此規則。
- Chrome【事實】：瀏覽器最多用磁碟 80%，單一來源最多 60%（S17 web.dev）。MDN 稱 Chrome 團隊研究顯示資料「very rarely deleted」，常造訪者被清除機率很低；LRU 僅在儲存壓力下啟動，且略過已 persist() 的來源（S18 MDN）。
- 23 MB 可行性【推論】：遠低於各家配額下限；主要風險不是配額而是 (1) 首次安裝下載 23 MB 的流量與時間，(2) 每天更新資料時整包重抓成本，宜做分檔版本化快取、僅更新變動的 JSON，(3) iOS best-effort 清除。
- 備註：web.dev 稱 Safari 約 1GB 且會提示（S17），與 WebKit 官方文章（S14，Safari 17 起 60%）不一致；WebKit 官方優先，S17 該段為作者自述「找不到官方文件」。

### Q5 子路徑對 scope 的限制
- service worker【事實】：預設 scope 為腳本所在目錄；scope 應是同目錄或更深；需更廣須由 HTTP 標頭 Service-Worker-Allowed 放行（S19 MDN register）。需 HTTPS（S20 MDN）。
- manifest scope【事實】：若未指定或 start_url 不在 scope 內，有效 scope 取 start_url 去掉檔名/查詢/片段；超出 scope 的導覽會出現瀏覽器 UI（S21 MDN）。
- 推論：站台掛 `/<repo>/`，只要 SW 檔放在 `/<repo>/` 底下、manifest 的 start_url 與 scope 都設 `/<repo>/`，即不需任何標頭，也不會越界。GitHub Pages 能否自訂回應標頭【未確認】（我只讀到大小與頻寬限制：站台 ≤1 GB、軟上限 100 GB/月，S22；沒讀到標頭相關原文），因此方案不應依賴 Service-Worker-Allowed。
- 絕對路徑陷阱【推論】：manifest 與 SW 內的 start_url、圖示、快取清單若寫成 `/xxx` 會跑出子路徑，需帶 base。

### 其他與決策相關的事實
- EU：iOS 17.4 beta 曾在歐盟停用主畫面 web app，後撤回（2024-03-01，皆為二手報導 S23；Apple 原文未取得，【無法核對】）。擁有者若不在歐盟可忽略，但顯示此能力有政策風險。
- iOS 無自動更新提示機制等細節【未確認】。

## 來源列表（皆於 2026-11-10 讀取）
- S1 https://webkit.org/blog/13878/web-push-for-web-apps-on-ios-and-ipados/
- S2 https://webkit.org/blog/17333/webkit-features-in-safari-26-0/
- S3 https://web.dev/articles/customize-install
- S4 https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/How_to/Trigger_install_prompt
- S5 https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Guides/Making_PWAs_installable
- S6 https://webkit.org/blog/12945/meet-web-push/
- S7 https://webkit.org/blog/16535/meet-declarative-web-push/
- S8 https://developer.mozilla.org/en-US/docs/Web/API/ServiceWorkerGlobalScope/notificationclick_event
- S9 https://developer.mozilla.org/en-US/docs/Web/API/Notification/requestPermission_static
- S10（僅線索）搜尋結果：Android Authority 等 2025-10 報導，Chrome 官方原文未取得
- S11 https://developer.mozilla.org/en-US/docs/Web/Manifest/id
- S12 https://developer.chrome.com/blog/pwa-manifest-id
- S13 https://chromium.googlesource.com/chromium/src/+/b5470ac602dd94de3fa37f35c8a2a3649c6d013d/components/webapps/docs/identifiers.md
- S14 https://webkit.org/blog/14403/updates-to-storage-policy/
- S15 https://webkit.org/blog/10218/full-third-party-cookie-blocking-and-more/
- S16 https://webkit.org/tracking-prevention/
- S17 https://web.dev/articles/storage-for-the-web
- S18 https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria
- S19 https://developer.mozilla.org/en-US/docs/Web/API/ServiceWorkerContainer/register
- S20 https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API/Using_Service_Workers
- S21 https://developer.mozilla.org/en-US/docs/Web/Manifest/scope
- S22 https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits
- S23（二手）MacRumors 2024-03-01 等，經搜尋結果取得
- 原始抓取檔：scratchpad/wayfinder/raw/（push、storage、ios26、itp、cid 等 .txt）

## 還不知道什麼
- iOS 26 / 26.x 上 Web Push 是否仍限主畫面 web app（只有 2023 年原文）。
- iOS 上 notificationclick + openWindow 指定頁的實測行為與版本差異。
- Android Chrome 不安裝是否可收推播、最低版本；Chrome 自動撤銷通知權限的官方規則與時限。
- GitHub Pages 是否能設定自訂回應標頭。
- 換 origin／子路徑時，各平台對已安裝 app 與既有訂閱的實際行為（需實測）。
- 同 github.io 網域下專案站的儲存隔離程度。
- iOS 主畫面 web app 長期不開啟時，快取被清除的實際門檻（官方只說「久未互動」，無天數）。
- Apple 原文對歐盟 web app 政策的現況。

## 對決策的含意（皆為推論）
1. 「只做可安裝網站」在 iPhone 上技術可行（安裝 + 離線 + 推播），不需開發者帳號與年費；代價是使用者必須自己加到主畫面並在站內點擊同意推播，轉換率與觸達會低於商店 app。
2. 推播設計宜以「使用者主動訂閱」為前提：訂閱按鈕放在站內，依球隊/比賽粒度分流；因 iOS 要求每則推播必顯示通知，不能做靜默同步。
3. 離線範圍：23 MB 技術上放得下，但 iOS 為 best-effort，建議把「核心殼 + 最近資料」列為必快取，歷史資料按需快取，並做更新時只拉差異。
4. 網址要先定案：換 origin 會讓安裝與訂閱全部歸零；若預期將來換自訂網域，宜儘早搬、並固定 manifest id。
5. 推播需要伺服器端發送（Web Push 協定）。「不自管常駐伺服器」與每日更新的限制下，可由排程（如 GitHub Actions）發送，但訂閱資料要存在某處；此為架構問題，本研究未涵蓋。
6. 只有在需要上架曝光、或需要 iOS 上無法由 Web 達成的能力時，才需考慮商店版；現有證據下主要差距是「安裝動線與曝光」，不是「能力」。
