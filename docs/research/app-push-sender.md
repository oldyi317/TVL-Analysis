# 不自管伺服器的推播發送方案研究

查證日期：2026-11-10　對應票：#186（Part of #182）

## 結論

1. 有一條零固定費用、無常駐機器、同時涵蓋 iPhone 與 Android 的路：可安裝網站（PWA）加標準 Web Push（VAPID），發送端用 Cloudflare Workers（免費方案）加 KV。iOS 需 16.4 以上且使用者須先「加入主畫面」；不需要 Apple 或 Google 開發者帳號（Apple 官方明說）。
2. Cloudflare 免費方案的硬限制是每次呼叫 50 個子請求，所以單次呼叫最多直接送 50 則 Web Push；人數到數百以上要分批（多次觸發或佇列）或改走 FCM 主題。分批做法為推論，未實測。
3. FCM 主題能省掉「自己存訂閱」的發送端迴圈，但瀏覽器端沒有官方文件確認的用戶端訂閱主題 API，訂閱仍得由後端呼叫 topic subscription server API（需 OAuth 與 service account），所以不真正省掉伺服器元件。套殼 iOS 要 APNs 金鑰，需付費 Apple 開發者計畫（官方頁未直接斷言，第三方與 Apple 論壇佐證）。
4. 「開賽提醒」要完全在裝置端排程，只有套殼 app 的 Local Notifications 做得到；可安裝網站沒有可用的等價 API（Chrome 的 Notification Triggers 官方文件寫明開發已終止）。代價是 iOS 套殼要裝到實機仍需開發者身分（免費 Apple ID 僅能 Xcode 側載，據第三方約 7 天到期）。
5. GitHub Actions 排程可當發送端（最短 5 分鐘、高載時會延遲、公開 repo 60 天無活動會被停用），但訂閱資料仍需有地方存且不可放公開 repo。

## 方案比較表

| 方案 | iOS 可安裝網站 | Android 可安裝網站 | iOS 套殼 | Android 套殼 | 需要的帳號與金鑰 | 要保存的使用者資料 | 誰、何時觸發發送 | 免費額度 | 已知限制 |
|---|---|---|---|---|---|---|---|---|---|
| A. Web Push + Cloudflare Workers + KV + Cron | 是（iOS 16.4+，須加入主畫面，須使用者手勢） | 是（Push API 為 Baseline） | 不適用 | 不適用 | Cloudflare 帳號；自產 VAPID 金鑰對（存為 Worker secret）；無 Apple/Google 開發者帳號 | 每位訂閱者的 PushSubscription（endpoint、keys）；可加隊伍偏好 | Workers Cron Trigger（UTC）依賽程或每日觸發 | Workers 10 萬請求/日；KV 讀 10 萬、寫 1,000/日；Cron 5 個/帳號 | 免費方案每次 50 子請求、CPU 10 ms；加密 CPU 是否夠用未確認；iOS 須 userVisibleOnly，不可靜默推 |
| B. Web Push + GitHub Actions 排程發送 | 同 A | 同 A | 不適用 | 不適用 | VAPID 私鑰存 Actions secret；訂閱仍需存放處（KV 或其他） | 同 A | Actions schedule | 公開 repo 的標準 hosted runner 免費 | 最短 5 分鐘、可能延遲、60 天無活動停用；不可用筆電 runner |
| C. FCM（網頁端） | 依賴 Web Push，細節未確認 | 是 | 不適用 | 不適用 | Firebase 專案、VAPID、service account | FCM 權杖（Installation ID）；主題訂閱由後端完成 | 後端呼叫 FCM v1 API 發主題訊息 | FCM 標示無費用、未列用量上限 | JS SDK 無用戶端訂閱主題；仍需有後端呼叫 server API |
| D. FCM + Capacitor 套殼 | 不適用 | 不適用 | 需 APNs 驗證金鑰（付費 Apple 計畫，推論） | 需 google-services.json | Firebase、Apple 開發者計畫（iOS）、Play 主控台 US$25 一次性（上架時） | FCM/APNs 權杖 | 後端或 Actions 呼叫 FCM | 同 C | Capacitor 推播外掛文件未提主題 API；iOS 不支援靜默推 |
| E. Capacitor Local Notifications | 不適用 | 不適用 | 是（本機排程，免伺服器） | 是（Android 12+ 需 SCHEDULE_EXACT_ALARM 才準時） | 側載可免帳號；上架才要開發者帳號 | 無伺服器資料；只存賽程與使用者的隊伍偏好於裝置 | App 開啟時依賽程排程；賽程變動需 app 再同步 | 無 | 排程只在 app 取得最新賽程時更新；iOS 待排通知上限與重開機持續性文件未載（未確認） |
| F. 可安裝網站的裝置端排程 | 無 | 僅 Chrome 實驗旗標 | 不適用 | 不適用 | 無 | 無 | 無 | 無 | Notification Triggers 開發已終止，不可作為產品依賴 |

## 逐題回答

### Q1 Web Push（VAPID）配 Cloudflare Workers + KV/D1 + 排程

- 訂閱存哪（事實）：瀏覽器訂閱後得到 subscription，須傳到伺服器保存才能日後發送；iOS 與 Chrome 皆須使用者手勢與明確授權。KV 免費：讀 100,000/日、寫 1,000/日、刪 1,000/日、列舉 1,000/日、儲存 1 GB；每日 00:00 UTC 重置，超過則該類操作失敗（來源 2）。D1 免費：讀 500 萬列/日、寫 10 萬列/日、儲存 5 GB（來源 3）。寫入只發生在訂閱與退訂，1,000 次/日對數千人規模夠用（推論）。
- 誰在排定時間發送（事實）：Workers Cron Trigger。Cron 以 UTC 執行；設定變更最長 15 分鐘生效；官方頁未給準時保證，只說跑在較不忙的機器（來源 4）。免費方案每帳號 5 個 Cron Trigger，每次 Cron 的 CPU 上限 10 ms（來源 1）。Cron 觸發時間可自由選，因此「依賽程當天開賽前」與「每日 09:23 後」都可排（推論）。
- 免費額度上限（事實）：請求 100,000/日（00:00 UTC 重置）；子請求 50/請求；CPU 10 ms；等待網路（fetch、KV）不計 CPU（來源 1）。
- 關鍵風險（推論）：每則 Web Push 是一次對推播服務的 fetch，算一個子請求，故單次呼叫最多約 50 則。數百到數千訂閱者需分批；官方限額頁沒有針對此的建議。Web Push 的 VAPID 簽章與 aes128gcm 加密的 CPU 耗時在免費 10 ms 是否足夠，找不到實測來源，標為未確認（搜尋僅見 Workers 相容函式庫 web-push-neo、web-push-browser，無耗時數據）。
- 不需要的帳號（事實）：Apple 明說發 Web Push 不需加入 Apple Developer Program（來源 6、7）。

### Q2 Firebase Cloud Messaging

- 費用（事實）：Firebase 價格頁將 Cloud Messaging 列為無費用產品，未列用量上限（來源 9）。
- 主題能否免去自存訂閱：主題限額為每個實例最多 2,000 個主題、批次訂閱每批最多 1,000 實例、訂閱速率 3,000 QPS/專案、同時 fanout 1,000（來源 10）。主題訂閱可在「伺服器端（topic subscription server API 或 Admin SDK）或用戶端 API」進行（來源 11），但英文文件未針對網頁指明用戶端 API；網頁 SDK 入門頁完全未提主題（來源 12）。搜尋結果的 Stack Overflow 回答稱 JS SDK 無 subscribeToTopic，屬第二手，標推論。結論：網頁端仍須把權杖送後端，由後端以 OAuth 呼叫 server API 訂閱；FCM 只是在「發送時」省掉逐一送的迴圈，並不省掉「存權杖」或「後端元件」（推論）。主題訊息「為吞吐量而非延遲最佳化」（來源 10），不保證準點。
- 發送端認證（事實）：FCM v1 發送需 service account JSON 或其衍生的短期 OAuth token（來源 13）。這份金鑰要有安全存放處（Worker secret 或 Actions secret）。
- iOS 套殼（事實）：Firebase iOS 文件要求上傳 APNs 驗證金鑰，「至少需要開發或正式其中一把」（來源 14）。是否須付費 Apple 開發者計畫：Firebase 頁未明說；Apple 論壇與 Expo、Flavio Copes 等第三方稱免費 Personal Team 不能開 Push Notifications 能力（搜尋結果，未讀 Apple 原文，標推論），價格約 US$99/年，官方價格頁未讀。
- iOS 可安裝網站走 FCM：FCM 網頁文件只說支援 Push API 的瀏覽器並連到支援矩陣，未點名 Safari/iOS（來源 12）；故 iOS PWA 經 FCM 的支援度「未確認」，直接走標準 Web Push 較有依據。

### Q3 套殼 app 的本機排程通知，與可安裝網站的對應能力

- 套殼（事實）：Capacitor Local Notifications 設計目標是「不經伺服器、在裝置上排程通知」；`schedule()` 支援 `at` 指定日期時間與重複（來源 15）。Android 12 起排程不精準，除非在 AndroidManifest 宣告 `SCHEDULE_EXACT_ALARM`；使用者仍可在設定關閉精準通知，關閉時 app 會重啟且以精準鬧鐘排的通知會被刪除（來源 15）。Android 13 需要通知權限請求。iOS 待排通知數量上限、重開機後是否保留：文件頁未載，標未確認（我只確認頁面全文檢索後無相關字樣，且曾以 curl 抓原始 HTML 剝標籤檢索 "reboot|boot|persist" 皆無命中）。
- 因此開賽提醒在套殼 app 內可完全不靠發送服務，但前提是 app 手上有最新賽程：賽程資料每日 09:23 更新，app 需在開啟時（或背景同步，後者未查證）重抓賽程並重排；使用者很久不開 app，賽程改期就不會反映（推論）。
- 可安裝網站（事實）：Chrome 的 Notification Triggers（showTrigger/TimestampTrigger）官方頁警告「開發已終止」，原因是無法保證跨平台一致與可靠體驗；目前狀態表「Origin trial Complete、Launch Not started」，僅能用實驗旗標在本機試（來源 16）。MDN 的 Periodic Background Sync 頁指出 Chrome 僅授予已安裝並獨立啟動的 web app，且為非 Baseline（來源 17），Safari/Firefox 支援在該頁文字中未載（無法核對表格內容）。結論：可安裝網站沒有可依賴的裝置端排程；開賽提醒需靠 Web Push 由外部觸發（推論）。

### Q4 從 GitHub Actions 排程直接發送

- 事實（來源 5）：schedule 最短間隔 5 分鐘；高載時可能延遲；僅在預設分支執行；公開 repo 在 60 天無活動後排程自動停用。
- 費用（事實，來源 18）：公開 repo 使用標準 GitHub-hosted runner 免費；自架 runner 免費。
- 推論：目前每日更新的 runner 在筆電，關機就不跑，故發送應另開用 GitHub-hosted runner 的 workflow，不能掛筆電 runner。VAPID 私鑰或 FCM service account 存於 Actions secrets；fork 的 PR 取不到 secrets（此點未查官方頁）。準點度不足以做「開賽前 30 分鐘」之類精準提醒，延遲量級官方未給，標未確認。訂閱資料不可放公開 repo（endpoint 洩漏等於可被濫發，推論），所以仍需 KV 或等價儲存，意味著仍要 Cloudflare Worker 接收訂閱。

### Q5 每日 09:23 更新對兩種通知的適用性（推論為主）

| 通知 | A Workers Cron | B Actions | C/D FCM | E 本機排程 | F |
|---|---|---|---|---|---|
| 開賽提醒 | 可。賽程在站內資料，Worker 可讀發布的靜態資料或自存賽程；Cron 觸發時間可多個，但免費帳號只有 5 個 Cron Trigger，需一個高頻（例如每 5–15 分鐘）檢查「近期是否有開賽」 | 可但準點度弱 | 可，但主題訊息偏吞吐量，準點弱 | 最合適，完全不需伺服器，但依賴 app 取得最新賽程 | 不可 |
| 隔日早上賽果摘要 | 可。09:23 更新後觸發，需排在爬蟲與部署完成之後；爬蟲筆電不保證在線，故 Worker 應讀「已發布的資料」而不是假設更新必到 | 可，可在爬蟲 workflow 完成後直接串接，時序最簡單 | 可 | 不可（賽果在裝置上不知道，除非 app 開啟時抓） | 不可 |

重點（推論）：賽果摘要必須有發送端；開賽提醒才有可能零伺服器，但僅限套殼。

## 來源列表（皆於 2026-11-10 查證）

1. Cloudflare Workers Limits：https://developers.cloudflare.com/workers/platform/limits/（頁面標示更新日 2026-10-29；免費請求 100,000/日、CPU 10 ms、子請求 50、Cron Trigger 5、單次 Cron CPU 10 ms；curl 原始 HTML 核對）
2. Cloudflare KV Pricing：https://developers.cloudflare.com/kv/platform/pricing/
3. Cloudflare D1 Pricing：https://developers.cloudflare.com/d1/platform/pricing/
4. Cloudflare Cron Triggers：https://developers.cloudflare.com/workers/configuration/cron-triggers/
5. GitHub Actions 觸發事件：https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows
6. WebKit 官方部落格 Web Push for Web Apps on iOS and iPadOS：https://webkit.org/blog/13878/web-push-for-web-apps-on-ios-and-ipados/（iOS/iPadOS 16.4、須加入主畫面、直接使用者互動、免 Apple Developer Program；curl 全文核對）
7. WebKit Meet Web Push：https://webkit.org/blog/12945/meet-web-push/（須使用者手勢、userVisibleOnly=true、違反則訂閱被撤銷；不提供靜默背景執行）
8. MDN Push API：https://developer.mozilla.org/en-US/docs/Web/API/Push_API（Baseline，自 2023-03 起各瀏覽器可用）
9. Firebase 價格：https://firebase.google.com/pricing
10. FCM 主題訊息：https://firebase.google.com/docs/cloud-messaging/topic-messaging
11. FCM 管理主題訂閱：https://firebase.google.com/docs/cloud-messaging/manage-topic-subscriptions
12. FCM 網頁入門：https://firebase.google.com/docs/cloud-messaging/web/get-started
13. FCM v1 API 發送與授權：https://firebase.google.com/docs/cloud-messaging/send/v1-api
14. FCM iOS 用戶端設定：https://firebase.google.com/docs/cloud-messaging/ios/client
15. Capacitor Local Notifications：https://capacitorjs.com/docs/apis/local-notifications
16. Chrome Notification Triggers API：https://developer.chrome.com/docs/web-platform/notification-triggers
17. MDN Periodic Background Sync：https://developer.mozilla.org/en-US/docs/Web/API/Web_Periodic_Background_Synchronization_API
18. GitHub Actions 計費：https://docs.github.com/en/billing/concepts/product-billing/github-actions
19. Capacitor Push Notifications：https://capacitorjs.com/docs/apis/push-notifications（iOS 須開 Push 能力；Android 用 FCM 與 google-services.json；iOS 不支援靜默推；頁面未提主題）
20. Google Play 主控台註冊：https://support.google.com/googleplay/android-developer/answer/6112435（US$25 一次性）
21. Apple 開發者計畫比較：https://developer.apple.com/support/compare-memberships/（抓到的文字缺勾選欄位，無法核對各項目歸屬）
- 線索（第二手，僅供推論）：搜尋彙整之 Stack Overflow 與 Apple 論壇討論（FCM JS 無用戶端主題訂閱、Personal Team 不能啟用 Push、側載約 7 天到期）；web-push-neo、web-push-browser 為 Workers 相容 Web Push 函式庫。

## 還不知道什麼

- Cloudflare 免費 10 ms CPU 是否足夠完成一則 VAPID 簽章加 aes128gcm 加密（無實測來源）。
- 超過 50 子請求的分批做法與 Cloudflare Queues 的免費額度（本次未查 Queues 頁）。
- Workers Cron 的實際準點度（官方頁無保證）、GitHub schedule 的實際延遲量級。
- iOS 可安裝網站經 FCM 的支援度（FCM 頁未點名 Safari）；FCM 網頁端能否在用戶端直接訂閱主題（官方英文頁未明示）。
- Apple 官方是否明文要求付費計畫才能在 iOS 套殼用遠端推播（只有第二手佐證）；US$99 價格與 NT$ 換算未取官方頁。
- Capacitor Local Notifications 在 iOS 的待排通知上限、兩平台重開機後是否保留（文件頁未載）。
- 套殼 app 背景更新賽程的可行性（未查 Background Runner 等外掛）。
- Web Push 在 Android 已安裝 PWA 的送達可靠度與省電機制影響（未查）。
- 商店審核對「只是網站套殼」的接受度（未查，不在本票範圍）。

## 對決策的含意（推論）

1. 若要零費用且不開任何開發者帳號，Web Push（PWA）加 Cloudflare Worker 是唯一涵蓋 iOS 與 Android 的路；商店版 app 在 iOS 上多出付費 Apple 帳號，Android 上架多出 US$25。
2. 需要保存的使用者資料可壓到最小：一筆 PushSubscription 加可選的隊伍偏好，放 KV；寫入量遠低於免費額度。相對地要新增的執行期元件是「一個 Worker（接收訂閱＋Cron 發送）」與「VAPID 私鑰 secret」。
3. 「開賽提醒」若要完全不靠發送服務，只有套殼加 Local Notifications；但這會綁死商店版與裝置端賽程同步，代價是賽程改期可能不即時。若接受 Worker，則開賽提醒與賽果摘要共用同一發送路徑，架構較單純。
4. 通知粒度要與 50 子請求限制一起看：以「依隊伍分群」發送或限定訂閱規模，會比逐人個別化內容容易落在免費額度內；人數成長到數百以上需改分批或升級（Paid 方案是否算固定費用，需另行評估，本票未查價格）。
5. 賽果摘要不要掛在筆電 runner 上：發送端應在雲端且讀「已發布的資料」，並接受爬蟲晚到時摘要延後或略過的行為。
6. 在做決定前先做兩個小驗證：一則 Web Push 在免費 Worker 的 CPU 耗時、iPhone 加入主畫面後實收測試；這兩項能直接消除本報告最大的未確認點。
