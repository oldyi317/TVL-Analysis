# 商店對套殼 app 的審查規則與開發者帳號條件

查證日期：2026-11-10　對應票：#184（隸屬 #182）
原始抓取檔：`/tmp/claude-1000/-mnt-d-TVL-Analysis/f87b07e7-dc5c-43c4-8551-487ca46f3d57/scratchpad/wayfinder/raw/`（apple_*.txt、g_*.txt）

## 結論

1. 事實：Apple 4.2 要求 app「elevate it beyond a repackaged website」，且 4.2.2 排除「web clippings」「collection of links」；官方沒有給「做到什麼算夠」的清單。純 WebView 套殼是高風險，這點有條文佐證；「加推播／離線就會過」只有開發者論壇與部落格（二手線索），查無官方保證。
2. 事實：Google 對「只提供網站 webview」的限制是針對「未經網站擁有者許可」的情況；本站擁有者與 app 開發者同一人，不落入該條。但「Limited Functionality and Content」仍適用。審查風險 Google 低於 Apple（推論）。
3. 事實：Google 個人帳號（2023-11-13 後建立）上架正式版前必須「closed test，至少 12 位測試者連續 14 天」，且必須用「實體 Android 10+ 手機」做裝置驗證。擁有者沒有 Android 實機，這是硬阻礙（需借／購一支手機，費用未確認）。
4. 事實：費用 Apple US$99／年（當地幣別價格於註冊時顯示，台灣 NT$ 數字未確認）；Google 一次性 US$25。兩者個人與組織帳號費用相同。
5. 事實：Apple 個人帳號的賣家名稱就是法定姓名，不能用別名；Google 個人帳號的「開發者名稱」可與法定姓名不同，但官方頁面同時寫「Google 會顯示你的法定姓名、國家、開發者 email」。個人帳號要完全不露實名，兩邊都做不到（Google 部分有一處條文矛盾，見逐題 4）。

## 逐題回答

### 1. Apple：對「只是包網站」的 app 的最低功能要求

原文（App Review Guidelines，頁面標示 Last Updated: October 16, 2026）：

- 4.2 Minimum Functionality：「Your app should include features, content, and UI that elevate it beyond a repackaged website. If your app is not particularly useful, unique, or "app-like," it doesn't belong on the App Store. If your App doesn't provide some sort of lasting entertainment value or adequate utility, it may not be accepted.」
- 4.2.2：「Other than catalogs, apps shouldn't primarily be marketing materials, advertisements, web clippings, content aggregators, or a collection of links.」
- 4.2.6：「Apps created from a commercialized template or app generation service will be rejected unless they are submitted directly by the provider of the app's content.」（自己是內容提供者、自己送審，所以不受此限；若用第三方「網站轉 app」生成服務，必須由內容提供者本人送。）
- 4.3(b)：「Don't submit apps that are indistinguishable from what's already widely available.」（排球數據站不屬於文中列的氾濫類別，但「體育賽事比分」類 app 很多；是否被認定相似為推論，未確認。）
- 2.1 清單：「Update your contact information in case App Review needs to reach you.」「Include detailed explanations of non-obvious features ... in the App Review notes」。
- 1.5 Developer Information：「Make sure your app and its Support URL include an easy way to contact you」。

說明（推論，標明）：條文只給原則，沒有列舉「哪些原生功能算夠」。官方也未明說「WebView 不行」，只說不能是 repackaged website。本站是「內容型、預渲染、每日更新」，正好落在「content aggregator／web clipping」的灰區邊緣；「球員與球隊資料庫」若被視為自有內容而非聚合，風險較低，但查無 Apple 官方判例佐證。

哪些做法常被退（二手線索，非官方）：
- Apple Developer Forums 一則討論（https://developer.apple.com/forums/thread/806726，查證 2026-11-10）：開發者主畫面、選單為原生，但 2–3 個內頁仍載入網頁，遭 4.2 退件 10 次；審查員回覆內容是對 4.2 條文的複製貼上，並附左側選單截圖稱內容不足。Apple 員工的官方回覆只有：建議預約「one-on-one App Review consultation」（Meet with Apple）。社群回覆建議把頁面改成原生畫面、增加實質內容。該串未顯示最終是否過審。
- 另一則（origin-devforums.apple.com/forums/thread/835821）WebFetch 回 HTTP 502，無法核對。搜尋摘要稱有人詳列原生功能於審查備註後仍遭 4.2.2 通用退件，此點未能核對原文。
- 多家廠商部落格與論壇一致建議：加入推播、離線、分享、原生導覽、相機等，並於審查備註說明。皆非 Apple 文件，只當線索。

被視為足夠的原生功能：Apple 官方未列舉，未確認。

### 2. Google Play：個人帳號要求與 TWA／WebView 政策

個人帳號測試要求（https://support.google.com/googleplay/android-developer/answer/14151465，2026-11-10）：
- 「Developers with personal accounts created after November 13, 2023, must run a closed test for their app with a minimum of 12 testers who have been opted in continuously for at least 14 days.」
- 滿足後才能在 Dashboard 申請 production access，要回答三段問卷（About your closed test／About your app/game／About your production readiness）；並須摘要測試回饋。
- 「Open testing ... becomes available after you gain production access.」

裝置驗證（https://support.google.com/googleplay/android-developer/answer/14316361）：
- 「developers with new personal accounts will be required to verify that they have access to a real Android mobile device using the Play Console mobile app before they can make their app available on Google Play.」
- 「You can use any non-rooted physical Android mobile device that runs at least the Android 10 operating system.」驗證不到一分鐘，同一支裝置可驗多個帳號，日後可換裝置。
- 事實推導：擁有者無 Android 實機，上架 Google Play 前必須取得一支（借用即可，因為只需通過一次驗證；是否能用別人的手機登入自己的帳號，條文未禁止，但「未確認是否有其他限制」）。

TWA 與 WebView 政策：
- 官方 TWA 概述（https://developer.android.com/develop/ui/views/layout/webapps/trusted-web-activities，頁面標示 Last updated 2026-02-26）：內容由使用者瀏覽器渲染、全螢幕，「the app and the site it opens are expected to come from the same developer. (This is verified using Digital Asset Links.)」「Web content must be accessible and useful in the browser first.」需 Chrome 72+，並「need to meet the same Add to Home Screen requirements」。此頁未談 Play 上架審查。
- Spam 政策（https://support.google.com/googleplay/android-developer/answer/9899034）：「We don't allow apps whose primary purpose is to drive affiliate traffic to a website or provide a webview of a website without permission from the website owner or administrator.」範例違規是未經許可包 Google Shopping。說明（推論）：自家網站套殼不是此條要禁止的對象，因為有擁有者許可。
- Functionality 政策（https://support.google.com/googleplay/android-developer/answer/9898783）：「We do not allow apps that only have limited functionality and content.」範例：「Apps that are static without app-specific functionalities, for example, text only or PDF file apps」。說明（推論）：資料動態更新的數據站較不像靜態 app，但此為推論，Google 未列「網站套殼」的接受標準。
- Repetitive Content：「Creating multiple apps with highly similar functionality, content, and user experience」不允許。若考慮男女隊、各賽季分開上架會碰到，單一 app 即可避免。
- 目標 API：自 2026-08-31 起，新 app 與更新必須 target Android 16（API 36）以上（https://support.google.com/googleplay/android-developer/answer/11926878）。
- 另有「Android developer verification」計畫（2026 年底起先在巴西、新加坡、印尼、泰國對認證裝置上的 app 要求開發者註冊，之後全球推行）：來自搜尋摘要指向 https://support.google.com/googleplay/android-developer/answer/16471116，本人未讀原文，「無法核對」，也未確認對台灣 Play 上架的影響。

### 3. 費用與帳號類型差別

Apple（https://developer.apple.com/programs/enroll/ ，另 https://developer.apple.com/support/purchase-activation/ ，2026-11-10）：
- 「The Apple Developer Program is 99 USD per membership year. Prices may vary by region and are listed in local currency during the enrollment process.」台灣的 NT$ 實價未確認（搜尋與台灣頁面皆未顯示）。
- 費用豁免僅限非營利、認可教育機構、政府單位；個人不適用。
- 個人（含 sole proprietor/single person business）：需 Apple Account 開雙重驗證、達當地法定成年；用法定姓名；審核確認「Legal name」「Email, phone, and address. P.O. boxes are not accepted.」
- 組織：需法律實體（不收 DBA、商號、分支）、D-U-N-S 號碼、與組織網域相關的工作 email、公開可用的組織網站。
- 另：條文指出可用 Apple Developer app 註冊（擁有者有 iPhone，可行；細節未讀）。

Google（https://support.google.com/googleplay/android-developer/answer/6112435，2026-11-10）：
- 「There is a US$25 one-time registration fee.」不接受預付卡；可能要求「valid government ID and a credit card, both under your legal name」，資料無效則費用不退。
- 滿 18 歲才可註冊。
- 個人 vs 組織（https://support.google.com/googleplay/android-developer/answer/13634885）：功能相同、皆可收款。差別：組織需 D-U-N-S；金融、健康（醫療）、VPN、政府類 app 必須用組織帳號；個人帳號有上述 closed test 與裝置驗證要求。排球數據站不屬必須用組織帳號的類別。

總費用（推論）：首年固定費 US$99 + US$25 = US$124，之後每年 US$99。以約 NT$30 兌 1 美元估（匯率為推論，未查證）約 NT$3,700，貼近擁有者設定的年度固定費用上限 NT$4,000（交辦時給的條件）；若需另購 Android 實機，會超出。

### 4. 商店頁面公開的開發者資訊

Apple（個人帳號）：
- 「Your name will be displayed as the seller name of your apps on the App Store.」且 Apple 說明（https://developer.apple.com/help/account/membership/program-enrollment/）：個人須以個人身分加入，「your legal name will appear as the seller」；「Using an alias, nickname, or company name as your first or last name will cause a delay in the approval」。結論：個人帳號賣家名稱＝法定姓名，不能用別名。要顯示別名只能走組織帳號（需法律實體＋D-U-N-S，且組織名稱即法律實體名稱，不收 DBA／商號）。
- 歐盟 DSA（https://developer.apple.com/help/app-store-connect/manage-compliance-information/manage-european-union-digital-services-act-trader-requirements/）：「Articles 30 and 31 of the Digital Services Act (DSA) require Apple to verify and display trader contact information for all traders distributing apps on the App Store in the European Union (EU). This includes an address, phone number, and email address」。非 trader 則「consumers in the EU will be informed that consumer rights ... won't apply」。Apple 稱判斷 trader 由開發者自評；「if you're a hobbyist and you developed your app with no intention of commercializing it, you may not be considered a trader」。本站不收費、無廣告，擁有者可能是 non-trader（推論；Apple 與 Google 都明言自己無法替你判定，法律問題未確認）。
- 非歐盟地區（含台灣）頁面是否顯示地址／電話：Apple 頁面只說明 trader 規定；非 trader 時未見其公開地址電話的說法，但「無法核對」是否完全不顯示。標未確認。
- App 內與 Support URL 仍須有聯絡方式（1.5），可用專用 email，不必是個人資訊。

Google（個人帳號，https://support.google.com/googleplay/android-developer/answer/13628312）：
- 需提供：Developer name（「can be different from your legal name」，可隨時更改）、法定姓名、法定地址、給 Google 的聯絡 email 與電話（皆需驗證）、公開的 developer email。
- 原文：「Google will display your legal name, your country (as per your legal address), and developer email address on Google Play. If you decide to monetize on Google Play then Google will display your full address.」→ 個人帳號公開法定姓名、國家、developer email；不收費且不上架付費內容時不公開完整地址。
- 條文矛盾：同頁寫「Developer name; can be different from your legal name」又寫「display your legal name」。實際商店頁畫面上法定姓名顯示在哪個位置，本次未能核對，標未確認；不應假設用暱稱就能完全不露實名。
- 歐盟／英國 DSA（https://support.google.com/googleplay/android-developer/answer/17594752）：所有帳號須在 2027-02-01 前宣告 trader 或 non-trader；新帳號建立時即須宣告。若宣告 trader：「If you use a personal developer account, you must provide a contact phone number to be displayed on Google Play. Be aware that your address will also be publicly displayed.」並從 2027-03-01 起 D-U-N-S、服務條款連結等顯示。non-trader：搜尋摘要稱僅顯示 email 與國家（本人於該頁未讀到此句，標推論），且會向歐盟／英國用戶標示消費者權利不適用。「If you don't distribute apps in the EEA or UK, you can declare as a non-trader.」誤報 trader 身分可能遭下架或停權。
- 組織帳號：公開顯示法定名稱、法定地址、developer email、developer phone。
- 另（來源 https://support.google.com/googleplay/android-developer/answer/10841920）：韓國個人帳號須另顯示電話；不適用台灣。

個人帳號能否改用別名：
- Apple：不能（賣家名稱＝法定姓名）。
- Google：開發者名稱可不同，但法定姓名仍會被顯示於頁面（條文）；是否因此完全揭露，未確認。

### 5. 上架必備項目

隱私權政策：
- Apple 5.1.1(i)：「All apps must include a link to their privacy policy in the App Store Connect metadata field and within the app in an easily accessible manner.」App Store Connect 說明：「You're required to provide a privacy policy URL for your iOS app platform」，並須填 App Privacy 資料處理說明。
- Google：User Data 政策 Key Considerations：「Have a valid privacy policy in both the app's store listing and within the app itself.」資料安全表單：「Even developers with apps that do not collect any user data must complete this form and provide a link to their privacy policy. In this case, the completed form and privacy policy can indicate that no user data is collected or shared.」（https://support.google.com/googleplay/android-developer/answer/10787469）內部測試軌道不需，封閉測試起需要。
- 事實／推論：本站無帳號、無後端，可寫「不蒐集」，但若使用任何分析或第三方 SDK／字型 CDN，須如實申報（推論）。

年齡分級：
- Apple：App Store Connect 問卷產生分級；「An Unrated app can't be published on the App Store.」（https://developer.apple.com/help/app-store-connect/reference/app-information/age-ratings-values-and-definitions/）問卷含「Unrestricted Web Access」等能力欄位；套殼若允許自由瀏覽任意網頁可能影響分級（推論，需據實填）。
- Google：IARC 等評級機構依問卷結果核發（https://support.google.com/googleplay/android-developer/answer/9859655），須在 Play Console 填問卷。

截圖規格：
- Apple（https://developer.apple.com/help/app-store-connect/reference/app-information/screenshot-specifications/）：每個裝置尺寸至少 1 張、最多 10 張，jpeg／jpg／png，不可含 alpha；必填 iPhone 6.x 吋「Dynamic Island（medium display）」級，本次只讀到 1179 x 2556 pixels 這一組。**未確認**：Apple 必填尺寸的完整清單沒有讀全（官方頁的尺寸表未完整取得），送審前須回官方頁核對；若 UI 跨尺寸一致，只需提供最高解析度，其餘自動縮放；支援 iPad 才需 iPad 13 吋。2.3.3：「Screenshots should show the app in use, and not merely the title art, login page, or splash screen.」
- Google（https://support.google.com/googleplay/android-developer/answer/9866151）：App icon 512×512、上限 1024KB；Feature graphic 1024×500（必填）；至少 2 張跨裝置類型截圖，JPEG 或 24 位元 PNG（無 alpha），最小邊 320px、最大邊 3840px、長邊不超過短邊 2 倍；每種裝置最多 8 張；要符合推薦版位需至少 4 張、短邊 ≥1080px（9:16 或 16:9）。

其他上架必備（事實，來自上列頁面）：Apple 需 Support URL（1.5）、審查備註；Google 需 target API 36（2026-08-31 起）、trader 宣告。

## 來源列表（皆於 2026-11-10 讀取）

第一手：
1. https://developer.apple.com/app-store/review/guidelines/ （4.2、4.2.2、4.2.6、4.3、1.5、2.1、2.3.3、5.1.1）
2. https://developer.apple.com/programs/enroll/ （個人／組織條件、US$99）
3. https://developer.apple.com/support/purchase-activation/ （US$99，in local currency）
4. https://developer.apple.com/help/account/membership/program-enrollment/ （個人賣家名稱＝法定姓名）
5. https://developer.apple.com/help/app-store-connect/manage-compliance-information/manage-european-union-digital-services-act-trader-requirements/
6. https://developer.apple.com/help/app-store-connect/reference/app-information/screenshot-specifications/
7. https://developer.apple.com/help/app-store-connect/reference/app-information/age-ratings-values-and-definitions/
8. https://developer.apple.com/help/app-store-connect/manage-app-information/manage-app-privacy/
9. https://support.google.com/googleplay/android-developer/answer/14151465 （個人帳號測試要求）
10. https://support.google.com/googleplay/android-developer/answer/14316361 （裝置驗證）
11. https://support.google.com/googleplay/android-developer/answer/6112435 （US$25、18 歲）
12. https://support.google.com/googleplay/android-developer/answer/13634885 （帳號類型）
13. https://support.google.com/googleplay/android-developer/answer/13628312 、 .../10840893 、 .../10841920 （顯示資訊、驗證）
14. https://support.google.com/googleplay/android-developer/answer/17594752 （trader 宣告）
15. https://support.google.com/googleplay/android-developer/answer/9899034 （Spam／Webviews）、 .../9898783 （Functionality）
16. https://support.google.com/googleplay/android-developer/answer/10787469 （Data safety）、 .../10144311 （User Data）、 .../9859655 （內容分級）、 .../9866151 （商店素材）、 .../11926878 （target API）
17. https://developer.android.com/develop/ui/views/layout/webapps/trusted-web-activities

二手線索（只當線索）：
18. https://developer.apple.com/forums/thread/806726 （開發者論壇：WebView app 4.2 退件）
19. 搜尋結果摘要（WebSearch，非原文）：TWA／Play 條文摘要、廠商部落格關於 4.2 的建議；已標明處皆非官方。

## 還不知道什麼

- Apple 對「內容型網站套殼」的實際過審率與「夠用」的原生功能清單：官方未列，只有二手經驗。
- 台灣地區 Apple 年費的 NT$ 實價與 Google 費用的實際扣款金額（匯率）。
- Google 個人帳號商店頁上法定姓名的實際顯示位置，及「developer name 可不同」與「顯示法定姓名」的真實界線。
- Apple 非歐盟地區、非 trader 個人帳號是否完全不顯示地址電話：無法核對。
- 擁有者是否構成歐盟／英國法下的 trader（本站免費、無廣告）：兩家都表示無法替開發者判定，法律意見未取得。
- Google Android developer verification 計畫對台灣個人開發者的影響：未讀原文。
- 借用他人 Android 手機做裝置驗證是否有其他限制；12 位測試者的招募難度（需找滿 12 人且連續 14 天）。
- Apple 審查時程、退件後申訴的時間成本：未查。

## 對決策的含意（皆為推論）

1. Google 的硬門檻是「Android 實機＋12 人 14 天封閉測試」，與審查嚴不嚴無關；擁有者沒有 Android 手機，若要走 Google Play，要先解決裝置與測試者來源。若只做 iOS，這兩項都不用。
2. Apple 的風險在 4.2：純套殼有被退件的實質可能，補原生功能的成本（推播、離線、原生導覽與畫面）可能比做 PWA 本身大；且退件次數與時間不可預測（論壇有 10 次退件案例，僅為個案）。
3. 隱私方面，Apple 個人帳號賣家名稱必為法定姓名，無法用暱稱；Google 開發者名稱可用暱稱，但法定姓名與國家仍會公開。兩邊個人帳號都無法完全避開實名曝光；若實名曝光是否決條件，商店版需改為組織帳號（需法律實體，成本與可行性另案評估），否則 PWA 才能維持暱稱署名。
4. 固定費用 US$124 首年（約貼近擁有者設定的 NT$4,000 上限，匯率為估計）還不含購買 Android 手機與可能的法律實體成本；預算上雙商店並行很緊。
5. 若仍要做商店版，較合理的順序是 iOS 先（擁有者有 iPhone、無額外硬體），但前提是願意承擔 4.2 風險並在審查備註與功能上補強；Google Play 因裝置與 12 人測試門檻，成本較高且對內容站收益不明。
6. 在決定前可先低成本驗證兩件事：預約 Apple 的 one-on-one App Review consultation（官方建議的管道）詢問本站類型是否可能過審；以及確認 Taiwan 區 Apple 年費實價。
