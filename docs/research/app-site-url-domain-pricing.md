# 個人註冊自訂網域（非商業體育數據站）：年費與個資曝光調查

查證日期：2026-11-10（所有「查證日」未另註者皆為此日）。
原始抓取檔在 `scratchpad/raw/`（API JSON、剝標籤全文）。

## 0. 先講結論

- 事實：所有 gTLD 的「續約價」才是年費重點；首年促銷（.xyz US$2、.club US$4 等）續約會跳到 US$14–25，不能拿來估年費。
- 事實：.tw 系列由 TWNIC 訂「管理費上限」：.tw／.com.tw／.net.tw／.org.tw／.game.tw／.ebiz.tw／.club.tw 每年 NT$800，.idv.tw 每年 NT$400；受理註冊機構不得超收（TWNIC 規章第十二條）。
- 事實：Cloudflare Registrar 不支援 .tw 系列（TLD 政策頁全文無 .tw）。
- 事實：.tw 系列 2023-06-01 起註冊人聯絡資訊「預設不揭露」。我實際查了 TWNIC RDAP，樣本中的個人 .idv.tw 姓名、地址、電話、Email 都顯示 "Redacted for Privacy"。TWNIC 公告與各註冊商價目表沒看到隱私保護的收費項，但註冊商結帳頁沒有逐家驗證，是否加錢未確認（見 4.1）。
- 事實：.com 每年可漲 7%（4 年區間內），今年 2026-11-01 剛漲過；新 gTLD 的 ICANN 基本合約沒有價格上限，續約漲價要提前 180 天通知（獨立驗證指出同條 (i) 款另有只需 30 天通知的例外，本報告未復核原文）。
- 預算（推論）：US$99 ≈ NT$3,245（匯率 32.78），NT$4,000 扣掉後網域剩約 NT$755/年。US$99（Apple 開發者年費）與每年 NT$4,000 上限是委託方給的前提，不是本報告查證的事實。續約價在這之下的選項很多（見第 8 節表）。

## 匯率

- 1 USD = 32.78 TWD：open.er-api.com，更新時間 2026-11-10 00:02 UTC。https://open.er-api.com/v6/latest/USD
- 交叉核對 32.85（fawazahmed0 currency-api，2026-11-09）。https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json
- 1 EUR ≈ 36.7 TWD：EUR/USD 1.1193（ECB 參考匯率，經 frankfurter.dev，2026-11-09）× 32.78。https://api.frankfurter.dev/v1/latest?base=EUR&symbols=USD
- 台灣銀行牌告匯率頁有機器人驗證，抓不到，未使用。
- 以下台幣換算皆為四捨五入，非任何銀行實際成交價。

## 1. 各 TLD 首年價與續約價

標示說明：「官方」＝註冊商自己的價目 API／頁；「第三方」＝比價站。價格不含稅（除非另註）。

### 1.1 Porkbun（官方公開 API，2026-11-10 即時抓取）

來源：`POST https://api.porkbun.com/api/json/v3/pricing/get`（無需金鑰）。

| TLD | 首年 USD | 續約 USD | 續約 TWD |
|---|---|---|---|
| .com | 11.81 | 11.81 | 387 |
| .net | 12.52 | 12.52 | 410 |
| .org | 7.98 | 11.84 | 388 |
| .app | 8.75 | 14.93 | 489 |
| .dev | 8.75 | 12.87 | 422 |
| .tw | 11.89 | 17.99 | 590 |
| .club | 4.12 | 15.96 | 523 |
| .xyz | 2.04 | 14.21 | 466 |
| .com.tw / .idv.tw | 不販售（API 無此項） | | |
| .club.tw / .game.tw / .ebiz.tw | 17.99 | 17.99 | 590 |

- 首年價是 API 的 `registration` 欄位，部分另有優惠碼（AWESOMENESS 首年折 US$1，僅 1 次）。
- 疑點：.com 11.81 是否已反映 2026-11-01 的批發漲價，未確認（Porkbun 加成不明）。

### 1.2 Cloudflare Registrar（第三方，官方價目需登入）

官方價目在 dashboard（需登入）；`domains.cloudflare.com` 對機器人回 403。以下取自 cfdomainpricing.com 的 JSON（`https://cfdomainpricing.com/prices.json`，各項 updatedAt 2026-10-09 至 10-13，抓取 2026-11-10）。這是第三方，非 Cloudflare 官方。

| TLD | 首年 USD | 續約 USD | 續約 TWD |
|---|---|---|---|
| .com | 10.46 | 10.46（見下，推估已漲） | 343（推估現價 366） |
| .net | 11.86 | 11.86 | 389 |
| .org | 8.50 | 11.20 | 367 |
| .app | 8.20 | 14.20 | 465 |
| .dev | 8.20 | 12.20 | 400 |
| .club | 15.20 | 15.20 | 498 |
| .xyz | 12.30 | 11.20 | 367 |
| .tw 系列 | 不支援 | | |

- .com 價格是 2026-11-01 Verisign 漲價「之前」的資料。推論：Cloudflare 宣稱按批發成本販售，批發價 US$10.26→10.97、ICANN 費 0.20，故現價應為 US$11.17（約 NT$366）。這是推論；Openprovider 的文章說註冊商成本從 10.46 變 11.17（https://www.openprovider.com/blog/com-price-increase-domain-resellers ，2026-11-05），我沒找到 Cloudflare 自己的公告。
- .xyz 首年 12.30 高於續約 11.20，與其他 TLD 的常態相反，可能是資料抓取時點不一致，未確認。
- Cloudflare 的「按註冊局定價續約」說法有官方文件：「your domain will only renew at the list price set by the registry」。https://developers.cloudflare.com/registrar/account-options/renew-domains/ （文件更新日 2026-04-24）。
- 第二來源（.com）：costgoat 比價頁顯示 Cloudflare .com US$10.46，頁面更新 2026-10-20（經 WebSearch 摘要取得，未直接讀原頁）。

### 1.3 Namecheap（官方頁 403；僅有第三方）

官方價目頁 `namecheap.com/domains/registration/gtld/<tld>/` 對 curl 與 WebFetch 均回 HTTP 403，無法核對官方價。以下取自 domainoffer.net（比價站，頁面標示 Prices Updated: Nov 10, 2026 04:16），例如 https://domainoffer.net/tld/com/namecheap 。

| TLD | 首年 USD（含促銷碼） | 續約 USD | 續約 TWD |
|---|---|---|---|
| .com | 6.48 | 15.98 | 524 |
| .net | 11.68 | 14.86 | 487 |
| .org | 7.24 | 15.18 | 498 |
| .app | 10.78 | 22.98 | 753 |
| .dev | 10.78 | 20.98 | 688 |
| .club | 4.98 | 24.98 | 819 |
| .xyz | 2.00 | 21.48 | 704 |
| .tw | 該站無此頁（404）；Namecheap 不在 TWNIC 受理註冊機構名單 | 未確認 | |

- 首年價含新客優惠碼 WELCOME30（僅部分副檔名適用），續約是牌價。
- 第二來源：WebSearch 摘要引用同站另一份 2026-07 資料，.com 續約 14.78、首年 6.79，與上表 15.98 不同，顯示 Namecheap 牌價在變動，需以結帳頁為準。

### 1.4 補充：Dynadot（官方頁，第二註冊商來源；Dynadot 是 TWNIC 受理註冊機構）

來源：https://www.dynadot.com/domain/pricing （2026-11-10 抓取）。

- .com 11.62／11.62、.net 12.52／12.52、.org 7.99／11.64、.xyz 1.99／14.45（首年／續約，USD）。.app、.dev、.club 該頁未列，未確認。
- .tw：https://www.dynadot.com/domain/tw 顯示 €16.45（首年、續約、轉入皆同）。頁面標「EN - USD」卻顯示歐元，可能依地區顯示，約 NT$604。

### 1.5 .tw 系列：TWNIC 上限與各受理註冊機構實價（新台幣）

**TWNIC 官方上限**（網域名稱註冊管理業務規章須公告事項，2023-12-11 最近修訂）：.com.tw／.net.tw／.org.tw／.game.tw／.ebiz.tw／.club.tw 與泛用型（.tw、.台灣、中文.tw 等）管理費 NT$800/年；.idv.tw NT$400/年；移轉、取消暫不收費。https://www.twnic.tw/dnservice/policy/contents.php?id=4&lang=zh-tw

**註冊商實價**

| 註冊商 | .tw／.com.tw（1 年） | .idv.tw（1 年） | 來源與備註 |
|---|---|---|---|
| 中華電信 HiNet | 新申請 800／續用 800 | 新申請 400／續用 400 | https://domain.hinet.net/#/dnprice （SPA，我抓其價格元件 JS `assets/dnprice-cXhU0I-y.js` 解析）。.com 880、.net 1,380、.org 930 元；多年期有折。含稅與否未確認 |
| PChome 買網址 | 新申請 800／續用 720（續購九折） | 新申請 400／續用 360 | https://myname.pchomebiz.com.tw/shopping/price_tw.htm 。多年期：idv.tw 3 年續 972、5 年 1,530、10 年 2,880（≈每年 324／306／288）；頁面另有促銷字樣，截圖日期不明。含稅與否未確認 |
| Gandi | .tw 首年 €22.00、續約 €22.00（不含 VAT 20%；≈NT$807，含法國 VAT €26.4≈NT$969） | 未列 | https://www.gandi.net/en/domain/tld/tw 。Gandi 台灣地區頁面可能顯示 TWD，我只抓到歐元版；WebSearch 摘要稱台灣地區顯示 744，未直接驗證 |
| Dynadot | .tw €16.45（≈NT$604） | 未確認 | 見 1.4 |
| Porkbun | .tw 首年 US$11.89／續約 US$17.99（≈NT$590） | 不販售 | 見 1.1 |
| 網路中文 Net-Chinese | 未確認 | 未確認 | 官網為 Nuxt SPA，抓到的 HTML 沒有價格；未找到可抓的價格 API |
| Cloudflare／Namecheap | 不支援／不在 TWNIC 名單 | 同 | 見第 2 題 |

**TWNIC 受理註冊機構名單**（2026-11-10 讀取原頁）：https://www.twnic.tw/dnservice/register/registrar.php?lang=zh-tw 列有協志聯合、中華電信、網路中文、PChome、新世紀資通、台灣固網、WebNIC、中華國際通訊、GANDI SAS、捕夢網、SEAAREA、Key-Systems、Ascio、Dynadot、GNAME、遠振資訊、匯智資訊、TWNIC 自身（rs.twnic.tw）。名單中沒有 Cloudflare、Namecheap；Porkbun 也未列（但它賣 .tw，推論可能經上述某家上游轉售，未確認）。

## 2. Cloudflare Registrar 支援哪些 TLD？

- .tw 系列：不支援。事實依據：下載 https://www.cloudflare.com/tld-policies/ 原始 HTML，剝標籤得 24,958 字全文，檢索 ` tw `、`.tw`、`idv`、`taipei`、`Taiwan` 皆無命中；同頁有 .com（Verisign）、.net、.org（PIR）、.app、.dev（Google Registry）、.xyz（XYZ）、.club（GoDaddy）。
- 你列的 10 個 TLD 中，Cloudflare 支援 .com、.net、.org、.app、.dev、.club、.xyz；不支援 .tw、.com.tw、.idv.tw。
- 交叉：cfdomainpricing.com 的 430 個 TLD 清單中沒有任何 `.tw` 結尾項。
- 查證日 2026-11-10。

## 3. 個人註冊 .tw／.idv.tw／.com.tw 的資格

TWNIC 官方（https://www.twnic.tw/dnservice/qa.php Q2、Q7、Q8；及 https://www.twnic.tw/dnservice/policy/contents.php?id=4&lang=zh-tw ）：

- .idv.tw：「凡自然人均可申請」。
- .tw（泛用型）、.game.tw、.ebiz.tw、.club.tw：不限制申請人資格。
- .com.tw：限依法設立登記的公司、行號、有限合夥、其他營利團體組織，或「執行業務專業人士，完成稅籍登記者」；外國公司依本國法登記者亦可。一般個人不符。
- 2003 年起不做事前文件審查，採事後檢舉審查，不符資格可被刪除（Q7）。
- 申請不需台灣身分證或統編的字樣，TWNIC 頁面沒有「需身分證」的要求；但註冊商可更嚴：

註冊商自訂規則（事實）：

- HiNet 價目頁：.idv.tw「限具有中華民國國籍之國民」；.tw/.台灣「依法登記之國內外公司、商號或自然人均可申請」。https://domain.hinet.net/#/dnprice （同上 JS 檔）
- PChome 價目頁：idv.tw「申請資格限具有中華民國國籍之國民」；.tw 泛用型「國內外公司、商號、法人或自然人均可申請」。https://myname.pchomebiz.com.tw/shopping/price_tw.htm
- 與 TWNIC 規章（自然人即可）不一致，推論：這是註冊商自加限制。外籍個人申請 .idv.tw 可能需換註冊商，我沒有找到外國註冊商賣 .idv.tw 的證據（Gandi 頁只列 .tw 及 .game/.ebiz/.club.tw、.com.tw 規則，未列 .idv.tw），未確認。
- 外國註冊商可否辦：.tw、.club.tw 等不限資格者可以。Gandi：「.TW domain names are open to anyone」，法人才需提供登記號碼（https://www.gandi.net/en/domain/tld/tw ）。Gandi、Dynadot、Key-Systems、Ascio、GNAME 在 TWNIC 名單內。

給本案的推論：經營者若是持有中華民國國籍的個人，.idv.tw（NT$360–400/年）資格沒問題；.com.tw 不適合。

## 4. WHOIS／RDAP 個資曝光

### 4.1 .tw 系列

事實，官方公告：TWNIC 公告（2023-04-21）「網域名稱申請同意書條文修訂暨 .tw/.台灣網域名稱之 WHOIS 查詢結果揭露方式調整公告」：自 2023-06-01 起 .tw／.台灣的 WHOIS「聯絡資訊（Contact）將預設為不揭露」，註冊人可向受理註冊機構辦理揭露與否的異動；2018 年的 GDPR 遮蔽措施同時廢止。https://www.twnic.tw/dnservice/policy/contents.php?id=20&lang=zh-tw （我讀了原頁全文）

事實，申請同意書：TWNIC 得提供聯絡資訊揭露選項，供註冊人選擇是否顯示。https://www.twnic.tw/dnservice/policy/contents.php?id=6&lang=zh-tw （2025-10-16 修訂版）

事實，Gandi 的轉述：個人註冊人與其他聯絡人「all information is not disclosed」；法人註冊人仍公開組織名稱、國別、Email；註冊人可經註冊商要求揭露。https://www.gandi.net/en/domain/tld/tw 。同頁另有一句「Individual registrant, admin and tech contacts: only name and email is disclosed」，應是指選擇揭露後的狀態，原文未明說，未確認。

事實，我的實測（2026-11-10）：用 IANA 登錄的 RDAP 端點 `https://ccrdap.twnic.tw/tw/domain/<域名>`（IANA bootstrap：https://data.iana.org/rdap/dns.json ）查兩個樣本，只看欄位型態、未記錄任何人名：

- 一個個人性質的 .idv.tw 樣本：registrant、administrative、technical 三個聯絡人的 `fn`、`adr`、`tel`、`email` 欄位值全為 "Redacted for Privacy"；另有一個 `contact-uri`（網頁聯絡表單連結）。註冊商名稱（公司）有顯示。
- 一個公司 .com.tw 樣本：registrant 的 `org` 與 Email 有顯示，`fn`、地址、tel 為 Redacted；admin／tech 全 Redacted。與 Gandi 描述的「法人仍揭露組織名稱與 Email」一致。
- 限制：樣本只有 2 個；port 43 的 whois.twnic.net.tw 我用 nc 連線，5 秒內沒有回應內容，未取得 WHOIS 文字版。我不知道該樣本註冊人是否曾主動選擇不揭露，但官方預設值與 Gandi 說明一致。

費用：TWNIC 公告與各註冊商價目均未見額外隱私保護費；HiNet、PChome 價目表只列管理費。對「是否加錢」我沒有看到任何收費項，但註冊商端的實際結帳頁無法逐家驗證，未確認。

### 4.2 gTLD 對照

- Cloudflare：事實，官方文件「Cloudflare Registrar offers personal data redaction on WHOIS for free」，姓名、Email、郵寄地址顯示 "Data Redacted"；州／省與國家依 ICANN 規定仍顯示；若註冊局不允許則不遮（"if permitted by the registry"）。第三方可透過表單轉信。https://developers.cloudflare.com/registrar/account-options/whois-redaction/ （文件更新 2026-10-24）。另 https://www.cloudflare.com/products/registrar/ 寫「Free WHOIS redaction」。
- Porkbun：事實，KB：WHOIS privacy 預設開啟、免費，不顯示真實聯絡資訊也不送給註冊局；少數副檔名（例如 .us）不允許，結帳時會警告。https://kb.porkbun.com/article/97-how-to-configure-whois-privacy-service-porkbun 。官方 FAQ：每個 Porkbun 網域免費含 WHOIS Privacy。https://porkbun.com/about/porkbun-faq 。Porkbun 對 .tw 的隱私說明，我沒在這些頁面讀到，未確認；但 .tw 註冊局端預設就不揭露（4.1）。
- Namecheap：官方頁 403，無法直接讀。WebSearch 摘要引用 Namecheap 自家頁面稱「免費終身 domain privacy，隨新註冊與轉入」（https://www.namecheap.com/security/domain-privacy-service/ ，未直接讀原頁），比價站也寫 "free WHOIS privacy included"（https://domainoffer.net/tld/com/namecheap ）。2021 年 Namecheap 部落格曾寫「只要仍是 Namecheap 客戶就免費」，措辭不同，未確認現行條款。
- 對照結論（推論）：.tw 預設遮蔽的是註冊局層級；gTLD 的遮蔽是註冊商提供的服務，換註冊商、轉出或關閉服務就可能曝光。三家 gTLD 註冊商預設都遮蔽姓名；Cloudflare 仍會顯示州／國家。

## 5. 續約漲價風險

### 5.1 .com

- 事實，Verisign 10-Q（2020）：2020-03-27 Verisign 與 ICANN 簽署 Third .com Amendment，允許 .com 每年最多漲 7%，限「每個六年期的最後四年」；首個六年期自 2018-10-26 起算。https://www.sec.gov/Archives/edgar/data/1014473/000101447320000019/vrsn-2020331x10q.htm （我下載全文檢索 "7%" 命中）
- 事實，漲價落地：批發價 2026-11-01 04:00 UTC 起由 US$10.26 漲至 US$10.97（約 +6.9%），加 ICANN 費 US$0.20 後註冊商成本 US$10.46→11.17。https://www.openprovider.com/blog/com-price-increase-domain-resellers （2026-11-05）。WebSearch 另列 tldes.com、realtimeregister.com、osir.com 等多篇一致，未逐篇讀原文。
- 推論：六年期自 2024-10-26 起算的話，2026-11 是第二個六年期第三年，之後 2027、2028、2029 還可各漲最多 7%。來源只有第三方文章（WebSearch 摘要：「up to 7% three more times: 2027, 2028, 2029」），我沒讀到 ICANN 合約原文第 7.3 節，未確認；依 7% 複利估，2029 年批發價最高約 10.97×1.07³≈13.44（costgoat 估 13.44，nameocean 估 13.45）。
- 可鎖價：ICANN 基本合約規定註冊局須讓註冊商以現價一次續約最多 10 年（見 5.2 原文）。Openprovider 文章也提到可在漲價前一次續 9–10 年鎖價。

### 5.2 新 gTLD（.app、.dev、.xyz、.club）

- 事實，ICANN 新 gTLD 基本註冊局合約（2017-07-31 版，99 頁，我下載 PDF 轉文字）Section 2.10：註冊價格上漲須提前書面通知註冊商，首次註冊 30 天，續約 180 天；註冊局須提供以「漲價前現價」續約 1–10 年的選項；續約價須對所有同期續約一致（uniform renewal pricing）。全文沒有任何價格上限（%）條款。https://newgtlds.icann.org/sites/default/files/agreements/agreement-approved-31jul17-en.pdf
- 推論：.app、.dev（Google Registry）、.xyz（XYZ.COM）、.club（GoDaddy Registry）使用此基本合約，故無法定漲幅上限；我沒有逐個讀各 TLD 現行合約，是否有個別修訂，未確認。實務上註冊商續約價已高於首年價（見第 1 節，.club／.xyz 續約是首年 3–7 倍）。
- .org：WebSearch 摘要稱 2019 年 ICANN 續約時刪除了 .org／.biz／.info 原有每年 10% 上限（ICANN 公共意見文件 mm.icann.org/pipermail/comments-org-renewal-18mar19/ ），我沒讀原文，標為第二手。
- .net：Verisign 10-Q（2020）寫 .net 每年可漲最多 10%（至 2023-06-30 合約期）；WebSearch 摘要稱 Verisign 8-K（2026-08-27）宣布 .net 批發價 US$10.91→12.00，2027-03-01 生效。8-K 原文 SEC 網站擋機器人，未能直接讀，標為第二手，現行 .net 合約是否仍有 10% 上限，未確認。

### 5.3 .tw

- 事實：.tw 價格由 TWNIC 費率委員會訂上限（規章第十一條，「維持收支平衡為原則」），註冊商不得超收（第十二條）。https://www.twnic.tw/dnservice/policy/contents.php?id=5&lang=zh-tw
- 事實：上限 NT$800/.idv.tw NT$400 的修訂紀錄中，最近一次修訂是 2023-12-11；但國際註冊商（Gandi、Dynadot、Porkbun）定價並非 TWD，隨匯率變動，且高於 TWNIC 上限折算值時（Gandi €22≈NT$807）的對應關係我沒查清楚，未確認。
- 推論：本土註冊商（HiNet／PChome）的 .tw 系列價格有上限保護，漲價風險最低；但上限本身可由 TWNIC 修訂。

## 6. 過期沒續約的後果

### 6.1 gTLD（Cloudflare 官方時程）

來源：https://developers.cloudflare.com/registrar/faq/ （「What happens when a domain expires」）

| 天數（自到期日起） | 狀態 |
|---|---|
| Day 0 | 到期日 |
| Day 1–30 | 寬限期，網域照常解析（不再自動續約重試） |
| Day 31–40 | 暫停期，解析到 Cloudflare 的暫停頁 |
| Day 41–70 | 贖回期（Redemption），不再解析；可能可以還原，另收 restore 費，金額依 TLD，頁面未列 |
| Day 71–75 | Pending Delete |
| 約 Day 75 之後 | 註冊局釋出，任何人可註冊 |

- Cloudflare 自己的說法：「Domain names should be released after a period of 75 days, although the exact deletion timeline is ultimately determined by the domain's registry.」
- Cloudflare 自動續約約在到期前 30 天首次嘗試，失敗會再試，但「no guarantee」。https://developers.cloudflare.com/registrar/account-options/renew-domains/
- ICANN 層級（第二來源，經 WebSearch 摘要，未讀原文）：Expired Registration Recovery Policy 規定註冊商須在到期前約一個月與一週各通知兩次，並允許到期後續約；註冊局須提供 30 天 Redemption Grace Period。https://www.icann.org/en/contracted-parties/consensus-policies/expired-registration-recovery-policy/expired-registration-recovery-policy-21-02-2024-en
- 其他註冊商（Porkbun、Namecheap）的過期時程：未確認，我沒有抓到各家頁面（Porkbun KB 的 expiry 文章 404；Namecheap 403）。
- HiNet 價目頁（官方）：「國際域名進入贖回期，贖回費用 5,000元（含續用一年費用）」。https://domain.hinet.net/#/dnprice

### 6.2 .tw 系列

- 事實，TWNIC「受理註冊機構執行凍結暨刪除網域名稱規範公告」（2017-07-25）：到期未繳續用管理費者，自到期日翌日中午 12:00 凍結（遇週末順延至週一）；自到期日後第 33 天中午 12:00 刪除。原文：「如到期凍結30日仍未繳費，執行刪除日為…」。https://www.twnic.tw/dnservice/policy/contents.php?id=19&lang=zh-tw 。此公告仍列於 TWNIC 現行公告清單，沒有標示廢止，但 2017 年之後是否有調整，我無法核對。
- 事實，Gandi：.tw 到期即停用，到期後 30 天內可續約；「Deletion process: No quarantine period」（無隔離期）。https://www.gandi.net/en/domain/tld/tw
- 事實：.tw 為先申請先發給（TWNIC Q4）。推論：刪除後立刻可被他人註冊，沒有看到贖回期，因此 .tw 的容錯窗口（約 33 天）比 gTLD（約 75 天）短。
- 被搶註的風險（推論）：過期域名釋出後，搶註者可立即註冊；PWA 的 origin 綁域名，若落入他人手中，使用者的安裝與推播訂閱會對著他人控制的網站。TWNIC 爭議處理辦法只在「與申訴人商標、姓名、事業名稱相同或近似且惡意」時才可申訴（Q11），一般情況救不回。
- 降風險做法（推論）：開自動續約並確認付款卡有效；一次續多年（HiNet 可到 10 年、gTLD 註冊局最多 10 年）；續約提醒寄到不依賴該域名的 Email。

## 7. 一個 PWA 相關的備註（推論，不在原問題內）

- 這份調查沒有查 GitHub Pages 對自訂網域的設定限制，也沒查 .app／.dev 的 HSTS preload 規則對測試環境的影響；兩項皆未確認，若要採用請另外查。

## 8. 摘要表：年續約費由低到高

匯率 32.78 TWD/USD、36.7 TWD/EUR。標「第三方」者官方價未能取得。WHOIS 欄指個人註冊時姓名是否預設遮蔽。

| # | 續約 TWD | TLD | 註冊商 | 續約原價 | WHOIS 是否遮蔽姓名 | 備註 |
|---|---|---|---|---|---|---|
| 1 | 360 | .idv.tw | PChome | NT$360（1 年，續購九折）；3/5/10 年約 324/306/288 每年 | 是（TWNIC 預設不揭露；樣本實測 Redacted） | 限中華民國國籍者（PChome 規定） |
| 2 | 366（列價 343） | .com | Cloudflare（第三方） | US$11.17 推估（列價 10.46，漲價前） | 是（免費遮蔽） | 2026-11-01 已漲 |
| 3 | 367 | .org | Cloudflare（第三方） | US$11.20 | 是 | |
| 4 | 367 | .xyz | Cloudflare（第三方） | US$11.20 | 是 | 首年 12.30 與續約反常 |
| 5 | 381 | .com | Dynadot | US$11.62 | 未確認 | |
| 6 | 382 | .org | Dynadot | US$11.64 | 未確認 | |
| 7 | 387 | .com | Porkbun | US$11.81 | 是（預設免費） | 是否含 11-01 漲價未確認 |
| 8 | 388 | .org | Porkbun | US$11.84 | 是 | |
| 9 | 389 | .net | Cloudflare（第三方） | US$11.86 | 是 | 2027-03 批發價將漲（第二手） |
| 10 | 400 | .idv.tw | HiNet | NT$400 | 是 | 限中華民國國籍者 |
| 11 | 400 | .dev | Cloudflare（第三方） | US$12.20 | 是 | |
| 12 | 410 | .net | Porkbun／Dynadot | US$12.52 | Porkbun 是；Dynadot 未確認 | |
| 13 | 422 | .dev | Porkbun | US$12.87 | 是 | |
| 14 | 465 | .app | Cloudflare（第三方） | US$14.20 | 是 | |
| 15 | 466 | .xyz | Porkbun | US$14.21 | 是 | 首年 US$2.04 |
| 16 | 489 | .app | Porkbun | US$14.93 | 是 | |
| 17 | 498 | .club | Cloudflare（第三方） | US$15.20 | 是 | |
| 18 | 523 | .club | Porkbun | US$15.96 | 是 | |
| 19 | 524 | .com | Namecheap（第三方） | US$15.98 | 是（依其頁面，未讀原頁） | 首年促銷 6.48 |
| 20 | 590 | .tw | Porkbun | US$17.99 | .tw 註冊局預設不揭露；Porkbun 對 .tw 的說明未確認 | 首年 11.89 |
| 21 | 604 | .tw | Dynadot | €16.45 | 同上（TWNIC 預設） | |
| 22 | 720 | .tw／.com.tw | PChome | NT$720 | 同上；個人註冊人姓名遮蔽 | .com.tw 需公司 |
| 23 | 800 | .tw／.com.tw | HiNet | NT$800 | 同上 | |
| 24 | 807 | .tw | Gandi | €22（不含 VAT） | 個人全遮蔽（Gandi 轉述） | |

未列入表：Namecheap .net／.org／.dev／.xyz／.app／.club（續約 NT$487–819，皆高於上列同 TLD）；.tw 在 Cloudflare、Namecheap 不可註冊。

## 9. 還不知道什麼

1. Cloudflare 與 Namecheap 的官方牌價：Cloudflare 價目在登入後的 dashboard、Namecheap 官網對機器人 403；表中 Cloudflare、Namecheap 皆為第三方資料。
2. Cloudflare .com 在 2026-11-01 漲價後的實際價格：推估 US$11.17，沒讀到 Cloudflare 官方公告或實際價目。Porkbun 的 .com 11.81 是否已含漲價，亦未確認。
3. .com 2027–2029 還能漲多少：只有第三方說法與 7% 公式推算，沒讀到 ICANN .com 註冊局合約第 7.3 節原文。
4. .app／.dev／.xyz／.club 現行個別合約有無修訂價格條款；.org 的 10% 上限刪除與 .net 目前合約上限、2026-08 Verisign 8-K 的 .net 漲價：只有第二手。
5. 網路中文（Net-Chinese）的 .tw 價格；HiNet、PChome 價格含不含 5% 營業稅；Gandi 的 .idv.tw 有沒有賣、台灣區顯示的 TWD 價格。
6. 外國註冊商可否辦 .idv.tw（個人、非 ROC 國籍者）：沒找到證據。
7. .tw 的 WHOIS 文字版（port 43）實測內容；RDAP 樣本只有 2 個，註冊商端是否收取「揭露／不揭露」異動費未確認。
8. TWNIC 2017 年凍結／刪除時程（凍結次日、刪除第 33 天）現行是否仍適用：公告未標廢止，但沒有更新版可對照；.tw 是否有贖回期：未見官方說明。
9. Porkbun、Namecheap 的過期後時程與 restore 費用；Cloudflare 各 TLD 的 restore 費用金額。
10. Gandi 稱「Late renewal 30 天」的費用是否另計：頁面列 €22.00，與一般續約同價，我不確定是否另有罰金。
11. GitHub Pages 自訂網域與 PWA 搬家的技術細節：不在本次範圍。

## 10. 來源清單（查證日皆為 2026-11-10，除另註）

- Porkbun 價目 API：https://api.porkbun.com/api/json/v3/pricing/get
- Porkbun WHOIS Privacy KB：https://kb.porkbun.com/article/97-how-to-configure-whois-privacy-service-porkbun
- Porkbun FAQ：https://porkbun.com/about/porkbun-faq
- Cloudflare TLD 政策：https://www.cloudflare.com/tld-policies/
- Cloudflare WHOIS redaction：https://developers.cloudflare.com/registrar/account-options/whois-redaction/
- Cloudflare 續約：https://developers.cloudflare.com/registrar/account-options/renew-domains/
- Cloudflare 過期時程 FAQ：https://developers.cloudflare.com/registrar/faq/
- Cloudflare 價目（第三方）：https://cfdomainpricing.com/prices.json
- Namecheap 價目（第三方）：https://domainoffer.net/tld/com/namecheap（同站 net／org／app／dev／club／xyz）
- Dynadot：https://www.dynadot.com/domain/pricing、https://www.dynadot.com/domain/tw
- Gandi .tw：https://www.gandi.net/en/domain/tld/tw
- HiNet 價目：https://domain.hinet.net/#/dnprice
- PChome 價目：https://myname.pchomebiz.com.tw/shopping/price_tw.htm
- TWNIC 受理註冊機構：https://www.twnic.tw/dnservice/register/registrar.php?lang=zh-tw
- TWNIC Q&A：https://www.twnic.tw/dnservice/qa.php
- TWNIC 認識 .tw：https://www.twnic.tw/dnservice/
- TWNIC 須公告事項（收費上限、申請條件）：https://www.twnic.tw/dnservice/policy/contents.php?id=4&lang=zh-tw
- TWNIC 註冊管理業務規章：https://www.twnic.tw/dnservice/policy/contents.php?id=5&lang=zh-tw
- TWNIC 申請同意書：https://www.twnic.tw/dnservice/policy/contents.php?id=6&lang=zh-tw
- TWNIC WHOIS 揭露調整公告：https://www.twnic.tw/dnservice/policy/contents.php?id=20&lang=zh-tw
- TWNIC 凍結暨刪除規範：https://www.twnic.tw/dnservice/policy/contents.php?id=19&lang=zh-tw
- .tw RDAP：https://ccrdap.twnic.tw/tw/domain/<域名>；IANA bootstrap https://data.iana.org/rdap/dns.json；IANA .tw 頁 https://www.iana.org/domains/root/db/tw.html
- Verisign 10-Q（Third .com Amendment）：https://www.sec.gov/Archives/edgar/data/1014473/000101447320000019/vrsn-2020331x10q.htm
- Openprovider（.com 2026-11-01 漲價）：https://www.openprovider.com/blog/com-price-increase-domain-resellers
- ICANN 新 gTLD 基本合約：https://newgtlds.icann.org/sites/default/files/agreements/agreement-approved-31jul17-en.pdf
- ICANN ERRP（第二手）：https://www.icann.org/en/contracted-parties/consensus-policies/expired-registration-recovery-policy/expired-registration-recovery-policy-21-02-2024-en
- 匯率：https://open.er-api.com/v6/latest/USD、https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json、https://api.frankfurter.dev/v1/latest
