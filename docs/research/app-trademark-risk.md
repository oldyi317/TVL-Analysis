# 非官方體育數據 app 的商標與資料來源風險

查證日期：2026-11-10　｜　對應票：#187（Part of #182）
**非法律意見。** 本文只整理條文、商店政策與公開先例；凡涉及法律性質的判斷一律標「推論」。

## 結論

1. 兩大商店的原文政策都明文禁止未獲許可使用第三方商標與標誌，Google 條文甚至把「college and professional sports team logos」列為著作權違規常見例。這是事實。
2. 商店審查不看「非官方」聲明有沒有寫：Apple 論壇多位開發者的轉述，一致認為單靠免責聲明不足以代替授權證明；Apple 官方文件無此說法，屬社群說法，不是官方立場。
3. 找到的公開先例（MLB 2008、MLB 2014、Apple 退件訊息）都是「要求移除隊徽／改名」，不是封殺球員姓名與比分。MLB 自己說過球員姓名與數據是公有領域，問題在隊徽與「讓人誤以為官方」。這些先例都非台灣、非排球。
4. 台灣法：比分等純事實不受保護的推論有條文支持（著作權法第 7、10-1 條）；台灣著作權法沒有獨立的資料庫權，編輯著作須有創作性（智財法院 2025 判決，二手轉述）。但隊徽與「TVL」商標是否註冊，我查不到，未確認。
5. 最大不確定：官網「使用者條款」未見對資料再利用的限制（經 WebFetch 摘要層回報，我自己的 curl 被 Cloudflare 擋，無法核對原文）。

## 逐題回答

### Q1 商店政策原文

**Apple App Review Guidelines**（https://developer.apple.com/app-store/review/guidelines/ ，2026-11-10 curl 全文抓取、剝標籤後檢索）

- 5.2 Intellectual Property：「Make sure your app only includes content that you created or that you have a license to use. Your app may be removed if you've stepped over the line and used content without permission.」
- 5.2.1：「Don't use protected third-party material such as trademarks, copyrighted works, or patented ideas in your app without permission, and don't include misleading, false, or copycat representations, names, or metadata in your app bundle or developer name. Apps should be submitted by the person or legal entity that owns or has licensed the intellectual property and other relevant rights.」
  說明：未經許可不得使用商標等受保護素材；名稱與中繼資料不得有誤導或仿冒。
- 5.2.2：「If your app uses, accesses, monetizes access to, or displays content from a third-party service, ensure that you are specifically permitted to do so under the service's terms of use. Authorization must be provided upon request.」
  說明：顯示第三方服務內容時，須確保服務條款明確允許，且被要求時要能出示授權。此條對「整理自官網」的內容有關（是否適用屬推論，官網是否算 "third-party service" 無官方解釋）。
- 4.1(c)：「You cannot use another developer's icon, brand, or product name in your app's icon or name, without approval from the developer.」
- 4.1(b)：「Submitting apps which impersonate other apps or services is considered a violation of the Developer Code of Conduct and may result in removal from the Apple Developer Program.」
- 2.3.7：「...don't try to pack any of your metadata with trademarked terms, popular app names, pricing information, or other irrelevant phrases just to game the system. App names must be limited to 30 characters.」
- 全文檢索未見 "unofficial" 字樣與任何「非官方聲明可免責」條款（此為我對該頁剝標籤後全文 grep 的結果，限於該次抓到的版本；否定性宣稱屬推論）。

**Google Play**

- 智慧財產權政策（https://support.google.com/googleplay/android-developer/answer/9888072 ，2026-11-10 curl 全文）：
  - 「We don't allow apps or developer accounts that infringe on the intellectual property rights of others (including trademark, copyright, patent, trade secret, and other proprietary rights).」
  - 著作權違規常見例含「College and professional sports team logos.」
  - 「Trademark infringement is improper or unauthorized use of an identical or similar trademark in a way that is likely to cause confusion as to the source of that product. If your app uses another party's trademarks in a way likely to cause confusion, your app may be suspended.」
  - 「If you have written documentation proving that you have permission to use a third party's intellectual property in your app or store listing (such as brand names, logos and graphic assets), contact the Google Play team in advance of your submission...」
  - 商標權人流程：「we encourage you to reach out to the developer directly... If you are unable to reach a resolution with the developer, please submit a trademark complaint through this form.」
  說明：Google 先鼓勵權利人直接聯絡開發者，再走商標申訴表單。
- 冒充政策 Impersonation（https://support.google.com/googleplay/android-developer/answer/9888374 ，curl 全文）：
  - 「Don't imply that your app is related to or authorized by someone that it isn't. Be careful not to use app icons, descriptions, titles, or in-app elements that could mislead users about your app's relationship to someone else or another app.」
  - 違規例：「Apps whose icons and titles are falsely implying a relationship with another company / developer / entity / organization」「Apps that falsely claim to be the official app of an established entity. Titles like "Justin Bieber Official" are not allowed without the necessary permissions or rights.」
- 欺騙行為／誤導聲稱（https://support.google.com/googleplay/android-developer/answer/9888077 ，curl 全文）：「Get all necessary permissions and rights before claiming to be an "official" app.」「Don't falsely imply an affiliation or relationship with another company...」
- 中繼資料（https://support.google.com/googleplay/android-developer/answer/9898842）：標題 30 字元內；Key Considerations 有「Don't use a celebrity's name or a brand's logo without permission.」

### Q2 名稱／圖示含 TVL、企業排球聯賽的風險；商店對「非官方」字樣的要求

- 事實：Apple 4.1(c) 禁止未經同意把「another developer's」品牌放進名稱或圖示（條文字面是 developer，是否涵蓋聯賽主辦單位是推論）。5.2.1 與 2.3.7 禁止在名稱與中繼資料放商標詞。Google 冒充政策把「icons and titles falsely implying a relationship」列為違規。
- 事實：兩家政策原文都沒有「加上 unofficial 就可使用」的條款；Google 的原文是「not use ... in a way that could mislead」，重點在是否誤導。
- 公開案例：Kodular 論壇一則 2019 年 7 月的 Google Play 下架貼文，app 為「RDVV University」，套件名含 unofficial，通知引冒充政策。Google 通知另稱無法驗證列表中圖文的「ownership or permission」，並建議可用新套件名與新 app 名稱提交合規版本。社群成員解讀是 listing 未揭露與品牌的實際關係；這是社群解讀，非 Google 立場（https://community.kodular.io/t/app-suspended-from-playstore/29878 ，經 WebFetch 摘要，2026-11-10）。
- 社群說法（Apple 論壇，無 Apple 員工回覆）：NBA 案中有人說「A simple disclaimer is no substitute」，也有人認為名稱含 NBA 風險高、「Machine for NBA players」型的描述性用法或許較安全，但該人明言需諮詢律師（https://developer.apple.com/forums/thread/130430）。
- 推論：「TVL」是聯賽縮寫；名稱含 TVL 或「企業排球」，Apple 審查可能以 5.2.1／4.1 要求證明授權，Google 可能以冒充／商標條款處理。「企業排球」是否因過於描述性而識別性低，涉及商標法第 29 條（描述性標識不得註冊，條文見下），但仍屬推論，且不等於可隨意使用。

### Q3 app 內顯示隊徽 vs 掛在網站上

- 事實：商店政策是針對「app 內與 listing 的內容」。Google 原文列「College and professional sports team logos」；Apple 的退件訊息（2021 年前後論壇貼文，原文見 https://developer.apple.com/forums/thread/699048）為：
  「Guideline 4.1 - Design - Copycats ... Specifically, your app includes content that resembles one or multiple third-party sports teams and/or leagues. You may attach documentary evidence in the App Review Information section in App Store Connect. ... Alternatively, please remove the third-party content from your app and its metadata.」
  另一則 2017 年貼文同樣是 4.1 退件，要求移除 icon、截圖、預覽中的第三方標誌；回覆者說可在 review notes 附授權證明（https://developer.apple.com/forums/thread/74658 ，無 Apple 員工回覆）。
- 推論：網站與 app 差別在於「商店是事前審查＋可單方下架的守門人」；網站沒有商店審查，只有權利人直接寄函或向主機商檢舉。
- 推論：差異是風險「觸發點」不同（審查員主動要求證明、權利人可向商店單一表單檢舉），法律上侵權與否不因載體而不同。上商店等於把「是否取得隊徽使用許可」變成被要求提出的問題。
- 事實：Apple 論壇上有人（無 Apple 員工）指出「其他 app 也這樣做」不被接受，且曾通過不保證下次通過（699048 回覆）。
- 常見降低風險做法（來自上述先例與社群，非官方保證）：移除隊徽改用隊名文字／城市名／隊色（718427 回覆者建議）；取得球隊或主辦單位書面授權並在 review notes 附上（74658、699048、Google 政策原文鼓勵事前聯絡 Google Play）。

### Q4 資料整理自官網的台灣法律性質

- 條文（全國法規資料庫，https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=J0070017 ，著作權法，頁面修正日期民國 111 年 6 月 15 日，2026-11-10 查）：
  - 第 7 條：「就資料之選擇及編排具有創作性者為編輯著作，以獨立之著作保護之。編輯著作之保護，對其所收編著作之著作權不生影響。」
  - 第 10-1 條：「依本法取得之著作權，其保護僅及於該著作之表達，而不及於其所表達之思想、程序、製程、系統、操作方法、概念、原理、發現。」
  - 第 9 條第 1 項第 4 款：「單純為傳達事實之新聞報導所作成之語文著作」不得為著作權之標的（此款針對新聞報導，對比分表格是否適用屬推論）。
  - 第 65 條：合理使用四要件（目的與性質含商業或非營利、著作性質、質量比例、對潛在市場影響）。
  - 全文檢索：該法條文無「資料庫」專章或專屬權利（頁面內「資料庫」字樣僅出現在網站導覽與頁尾，非條文）。推論：台灣著作權法沒有歐盟式 sui generis 資料庫權。
- 公開先例（二手，Lee and Li 2025-05-29 電子報轉述）：智慧財產及商業法院 113 年度刑智上易字第 35 號刑事判決（2025-01-16）維持不起訴／駁回，轉述認定彙整政府與公開網站資料的資料庫「不具原創性」、「辛勤勞動本身不足以構成著作權保護」。這是房地產資料的案例，不是體育資料；我只讀到律所摘要，未讀判決原文（https://www.leeandli.com/TW/Newsletters/7448.htm ，經 WebFetch 摘要）。
- 學界資料另引最高法院 91 年台上字第 940 號（中古車行情表缺乏創作性）——來自搜尋摘要的轉述，我未開原文，標「待核對」。
- 主管機關說明：我搜尋「智慧財產局 比賽結果／賽程表／事實性資料」未找到智慧財產局針對體育比分的解釋函。找到的最接近一則是 114.10.22 電子郵件字第 1141022 號（比賽現場播放歌曲的公開演出問題），與資料保護無關。結論：未找到。
- 公平交易法（https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=J0150002 ，頁面修正日期民國 106 年 6 月 14 日）：
  - 第 22 條第 1 項第 2 款：「事業就其營業所提供之商品或服務，不得有下列行為：...以著名之他人姓名、商號或公司名稱、標章或其他表示他人營業、服務之表徵，於同一或類似之服務為相同或近似之使用，致與他人營業或服務之設施或活動混淆者。」第 2 項：已依法註冊取得商標權者不適用本條。
  - 第 25 條：「除本法另有規定者外，事業亦不得為其他足以影響交易秩序之欺罔或顯失公平之行為。」
  - 推論：適用對象是「事業」；不收費、無廣告的個人愛好者是否為「事業」，我沒有找到說明，未確認。另，若直接大量重製他人整理的內容「搭便車」，第 25 條是學者提到的補充保護路徑（來自搜尋摘要轉述，未讀原文）。
- 官網條款：WebFetch（摘要層）回報 https://tvl.ctvba.org.tw/ 頁尾為「© 2025 Top Volleyball League 企業排球聯賽 | 隱私權政策」，連到 /policy-user；該頁「no clauses on reuse, scraping, or redistribution」、無商標聲明。我用 curl 抓同站被 Cloudflare 擋（回 "Sorry, you have been blocked"），所以沒有原文可交叉核對，只能寫「無法核對」，不能寫「原文未見」。

### Q5 其他非官方體育數據 app 的先例

| 先例 | 內容 | 來源（皆 2026-11-10 讀取） |
|---|---|---|
| MLB 與 Baseball 參考 app（2008-07-17） | MLB Advanced Media 來信要求移除隊徽（含捷徑小圖示），理由為商標，並擔心使用者誤以為官方；MLB 表示球員名與數據屬公有領域。開發者表示會移除。更新是否上架文章未證實。 | https://www.macworld.com/article/191564/baseball-2.html |
| MLB 球迷 podcast 下架（2014-05-08） | MLB 向 Apple 提交含約二十幾個 podcast 的商標通知，Apple 先隱藏；文章稱 MLB 慣例是通知對方改名並持續溝通，且節目預計重新上架。 | https://techcrunch.com/2014/05/08/mlb-fan-podcasts-removed-from-itunes-to-return-shortly/amp/ |
| 世界盃 2018 app 退件 | 退件訊息：「your app still includes content that resembles FIFA. ... remove any terms that are a FIFA's trademark.」結果未證實。 | https://developer.apple.com/forums/thread/107074 |
| 足球賽果 app，5.2.1（約 2022 年） | 資料來自付費 API（api-football）；Apple 員工只回覆「讀 API 提供者條款中關於 Data/Logos/images 的部分」；開發者移除世界盃標誌並重新提交，結果未證實。 | https://developer.apple.com/forums/thread/718427 |
| 足球 app，4.1 退件 | 見 Q3 原文；先前版本曾通過，後來被更嚴格的審查員退件。 | https://developer.apple.com/forums/thread/699048 |
| 手球隊 app，4.1 退件（2017-03） | 要求移除 icon／截圖中的第三方標誌；社群說附授權信。 | https://developer.apple.com/forums/thread/74658 |
| Google Play 大學 app 下架（2019-07） | 冒充政策；見 Q2。 | https://community.kodular.io/t/app-suspended-from-playstore/29878 |

- 搜尋範圍：英文「unofficial fan app removed App Store trademark league」「Google Play suspended unofficial football league」、中文「台灣 運動 非官方 App 下架 商標 隊徽 侵權」。
- **台灣體育聯盟（企業排球、CPBL、P.LEAGUE+）對非官方 app 要求下架或改名的公開案例：未找到。** 我另在搜尋結果看到泰國 CTH 控告 Apple 英超轉播 app 的報導（https://www.mobileworldlive.com/apple/thai-broadcaster-sues-apple-sport-app/），那是轉播權案，與數據 app 性質不同，只列為線索，未讀內文。
- 以上先例多為論壇與舊新聞；個案結果多半未證實，且沒有一個是「僅用比分與球員資料、未用隊徽與聯盟名稱」而被下架。這個「沒有」是我未找到，不是不存在。

## 來源列表（2026-11-10 查）

第一手：
1. Apple App Review Guidelines https://developer.apple.com/app-store/review/guidelines/
2. Google Play 智慧財產權政策 https://support.google.com/googleplay/android-developer/answer/9888072
3. Google Play 冒充政策 https://support.google.com/googleplay/android-developer/answer/9888374
4. Google Play 欺騙行為政策 https://support.google.com/googleplay/android-developer/answer/9888077
5. Google Play 中繼資料政策 https://support.google.com/googleplay/android-developer/answer/9898842
6. 著作權法 https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=J0070017
7. 商標法 https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=J0070001 （頁面日期民國 112 年 5 月 24 日；第 18、29、30、36、68、70 條）
8. 公平交易法 https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=J0150002
9. TVL 官網 https://tvl.ctvba.org.tw/ 與 /policy-user（僅 WebFetch 摘要）

二手／線索：
10. Lee and Li 電子報 https://www.leeandli.com/TW/Newsletters/7448.htm
11. Apple 開發者論壇 5 則（718427、107074、130430、699048、74658，見上表）
12. Macworld 2008、TechCrunch 2014、Kodular 論壇 2019（見上表）

商標法條文節錄（原文，供對照）：
- 第 30 條第 1 項第 5 款：「相同或近似於國際跨政府組織或國內外著名且具公益性機構之徽章、旗幟、其他徽記、縮寫或名稱，有致公眾誤認誤信之虞者」不得註冊（與是否適用於排協屬推論）。
- 第 36 條第 1 項第 1 款：「以符合商業交易習慣之誠實信用方法，表示自己之姓名、名稱，或其商品或服務之名稱、...特性、用途、產地或其他有關商品或服務本身之說明，非作為商標使用者」不受商標權效力拘束；第 2 款：表示商品或服務使用目的而有使用他人商標之必要者，但有致混淆誤認之虞者不適用。
- 第 68 條：未得同意，於同一或類似商品或服務使用相同或近似於註冊商標之商標，有致相關消費者混淆誤認之虞者，為侵害商標權。

## 還不知道什麼

- TVL／企業排球聯賽／各隊隊徽是否在台灣註冊商標：未確認。經濟部智慧財產局商標檢索系統（tmsearch.tipo.gov.tw）curl 連線失敗（HTTP 000），一般搜尋也沒有相關結果。要靠人工在瀏覽器查。
- 隊徽的著作權歸屬（球隊、企業、設計者）：未確認。
- 官網 /policy-user 的原文有無資料再利用限制：無法核對（原文被 Cloudflare 擋，僅有摘要層回報）。
- Apple 審查實務上對「獨立開發者、免費、無廣告、非官方聲明」的容忍度：無官方資料，只有論壇轉述。
- 排協或球隊對外部開發者使用隊徽的授權慣例與意願：未找到公開資訊；不得代擁有者聯絡，需擁有者自行決定。
- 智慧財產局對體育比分／賽程資料的解釋函：未找到。
- 公平交易法第 22、25 條對非營利個人 app 的適用：未確認。

## 對決策的含意（以下皆為推論）

1. 風險集中在「隊徽」與「名稱／圖示含聯賽縮寫」，不在比分、球員姓名等事實資料。先例與 MLB 自述都把爭點放在標誌與讓人誤認為官方。
2. 「非官方」聲明有助於降低「誤認」的疑慮，但沒有任何商店政策原文承認它可免責，且論壇共識是不能取代授權證明。應視為輔助，不當作保險。
3. 風險最低的路線是 app 內不放官方隊徽（改隊名文字、隊色或自製識別），名稱用描述性字樣且不含 TVL 縮寫與「官方」字樣，圖示自製。這同時讓商店審查的「請出示授權」問題不會出現。
4. 若想保留隊徽，唯一有商店原文支持的做法是取得書面授權後附在 review notes（Apple）或事前聯絡 Google Play。此舉需由擁有者自行聯絡排協或球隊，且要先查商標與著作權歸屬。
5. 事後被檢舉的情境，兩家商店流程都是通知開發者並可能先下架；Google 條文明示商標權人宜先聯絡開發者。實際上是否先通知取決於權利人與平台，無台灣案例可佐證。
6. 在商標檢索與官網條款原文兩項補齊之前，無法判斷「上商店」整體風險；建議先補這兩項（可由擁有者在瀏覽器完成），再決定名稱、圖示與是否上架。
