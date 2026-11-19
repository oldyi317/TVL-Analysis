---
status: accepted
date: 2026-11-19
---

# 網站維持純靜態，只為推播新增一個發送服務；App Store 不做，Google Play 延後

本站自 ADR 0001 起是純靜態網站，`web/PRODUCT.md` 寫的是「沒有執行期 API、沒有登入、沒有使用者輸入」。spec #231 把本站做成可安裝、可離線、會推播的可安裝網站，其中推播得有一個地方保管推播訂閱，在球迷沒開著本站時把通知送出去，靜態主機做不到。決定：**只為推播新增一個發送服務，其餘不變**，仍然沒有帳號、登入、社群。同一時間要定的還有商店版 app 做不做（#193），選項：(a) 只做可安裝網站；(b) 可安裝網站加商店版 app；(c) 分階段。決定 **(c)**：第一階段只做可安裝網站，**App Store 任何階段都不做，Google Play 延後**。

- `web/PRODUCT.md` 的定位從「純靜態、沒有執行期 API」改成「只為推播新增一個發送服務，仍然沒有帳號、登入、社群」。定稿文字在 `web/PRODUCT.md` 的 `## Positioning`。
- 第一階段只做可安裝網站，iPhone 與 Android 同一期上線，沒有任何一項等商店版 app。
- App Store 不做。iPhone 球迷一律用可安裝網站。
- Google Play 是沒有期限的第二階段。要做時用 TWA（殼只負責開啟正式網址），不用 Capacitor，另開地圖。沒有日期或數字門檻，使用者想做時啟動，啟動前四項都要成立：
  1. 第一階段已在正式網址上線。
  2. 商標查證有結果，且不擋隊徽與名稱。
  3. 查清楚 Google 商店頁不會公開顯示法定姓名。
  4. 手上有一支 Android 10 以上的實體手機，也找得到 12 位願意連續 14 天參與封測的人。
- 商店頁會公開顯示法定姓名的那一家商店就不上。若商店審查對「企排」有疑慮，商店頁的顯示名稱可單獨退為 `VStat`。

發送服務的理由：站上有新賽果、當天有比賽時球迷不會知道，只能自己記得回來看，這是 spec #231 要解決的三件事之一。新增的東西只有它：網站仍是唯一的前端，發送服務只讀站上本來就公開的推播賽事檔，自己決定什麼時候發。推播用不到身分：推播位址本身就是憑證，每台裝置各自訂閱、不同步，不需要知道球迷是誰；帳號、登入與社群都在 spec #231 的 Out of Scope。

商店版 app 的取捨（#193 的結論留言，2026-11-10）。主目標三項（可安裝、可離線、會推播）都不需要商店版 app，它唯一的用途是次要目標「走一次商店上架流程」。App Store 出局有三個理由，各自都夠：

- 個人帳號的賣家名稱就是法定姓名，會公開顯示在商店頁。repo 的授權刻意用別名署名，方向一直是少曝光；本站又是用了聯賽名稱與隊徽的非官方站，實名會讓商標爭議直接找到本人。
- 套殼裡標準 Web Push 不能用。Apple 工程師 2024-07 在開發者論壇明說「Web Push Notifications will not work in apps with a WKWebView」（<https://developer.apple.com/forums/thread/760767>，2026-11-10 查）；Apple 文件列的支援情境只有主畫面的可安裝網站與 macOS 的 Safari（<https://developer.apple.com/documentation/usernotifications/sending-web-push-notifications-in-web-apps-and-browsers>）。要推播就得寫原生程式碼、發送服務另接 Apple 的通道（這一句是推論），違反「網站是唯一前端」。iOS 17 以後的套殼有沒有開放，找不到任何官方說已開放，「到現在仍然不行」也是推論。
- 年費約 NT$3,245，加 `.tw` 網域 NT$720 合計約 NT$3,965，貼著每年 NT$4,000 的預算上限；過審門檻沒有官方說法，可能付了錢還被退件。

Google Play 選 TWA 的理由：殼載入線上站，每日資料更新不用重新送審，沒有第二份前端要維護；驗證檔要放網域根目錄，搬到正式網址後放得了（ADR 0013）。Capacitor 內嵌檔案，每次資料更新都要送審。第四項前提來自 Google Play 的裝置驗證要求：「any non-rooted physical Android mobile device that runs at least the Android 10 operating system」（<https://support.google.com/googleplay/android-developer/answer/14316361>，2026-11-10 查）。

## Consequences

- 網站本身仍是純靜態：所有頁面建置時產生，放在 GitHub Pages（ADR 0001、ADR 0013）。
- 唯一的執行期服務是放在 Cloudflare Workers 的發送服務（程式在 `push/`，正式環境 `https://push.vstat.tw`）。它的狀態怎麼存、怎麼不重發見 ADR 0014；部署由使用者手動做，流程在 `docs/ops/push-service.md`。
- 推播由球迷自己開啟，預設不發；站台不主動跳系統的通知權限詢問。
- 推播訂閱不綁任何身分。關於球迷，發送服務存的只有推播位址、加密用的兩把金鑰、兩種通知各自要不要、選的組別，以及建立與更新時間（`push/migrations/0001_init.sql` 的 `subscriptions`）。
- iPhone 球迷只有可安裝網站一條路；iOS 16.4 以前沒有 Web Push，不另外設計。
- 第二階段啟動之前，Android 球迷同樣只有可安裝網站。
