# 獨立核對：兩份研究報告的 10 條決策依賴事實（查證日 2026-11-10）

被核對報告：`scratchpad/domain-pricing.md`、`scratchpad/origin-options.md`（唯讀，未修改）。

## 結論

- 10 條中 **10 條與來源原文相符**（第 2、3 條附分組解讀註記；第 6 條附時效註記；第 5 條 .tw 揭露為預設值，非強制）。
- 找到的不符點（報告寫法與來源不一致）：**0 條**。
- 另列 **內部矛盾／不完整 4 項**、**截斷 2 處**（見後段），不影響 10 條的判定，但其中「不用另外加錢」一句超出證據，應修正。
- 判定：**完成**。

## 核對方法

- 來源優先：對 1、2、4、5、6、7、8、9 於 2026-11-10 以 `curl` 重新抓取原始 HTML（存於 `scratchpad/vtools/fresh/`），以本地腳本剝標籤後全文檢索引文；另與報告落地檔 `scratchpad/raw/` 比對（檔案大小與引文一致，TWNIC 與 GitHub、Cloudflare 頁面相同）。
- 條 3（HiNet）：HiNet 價目頁為 SPA，沿用報告落地的價格元件 JS（`raw/hinet_price.js`，2026-11-10 抓取）解析，未另行重抓。
- 條 10：本次實跑 `curl -sI https://twbs.github.io/bootstrap/docs/`（見下）。
- 搜尋引文時使用 `python3 -I`，未對下載內容執行任何程式。
- 未以 WebFetch 摘要代替原文；所有引文均為頁面原字。

## 十條判定

### 1. TWNIC 管理費上限：.tw 泛用型與 .com.tw 每年 NT$800、.idv.tw 每年 NT$400 — **相符**

來源：`https://www.twnic.tw/dnservice/policy/contents.php?id=4&lang=zh-tw`（網域名稱註冊管理業務規章須公告事項，112 年 12 月 11 日修訂）。
原文：「收費上限 … com.tw net.tw org.tw game.tw ebiz.tw club.tw 管理費800元/年 idv.tw 管理費400元/年 泛用型網域名稱 管理費800元/年」。
報告寫法與原文一致。

### 2. PChome .tw 一年價：新申請 NT$800、續用 NT$720；.idv.tw 新申請 400、續用 360 — **相符（附分組註記）**

來源：`https://myname.pchomebiz.com.tw/shopping/price_tw.htm`（2026-11-10 重抓）。
原文（頁面以列群組呈現，抽出文字後為）：「.com.tw 新申請 $800 續用費 $720 轉入費 $720 … .net.tw .org.tw .game.tw 泛用型中文.tw 泛用型中文.台灣 泛用型英文.tw .idv.tw 新申請 $400 續用費 $360 轉入費 $360」。
註記：泛用型 .tw 與 .com.tw 同列於 800／720 的價格群組，頁面文字中 .tw 沒有獨立一行，報告以「.tw 800／720」採用此群組解讀，判定相符；續用 720 為續購九折，與頁首「新註冊 / 續購九折」一致。

### 3. HiNet .tw 一年價 NT$800、.idv.tw NT$400 — **相符**

來源：`https://domain.hinet.net/#/dnprice`，價格元件 JS（`raw/hinet_price.js`，2026-11-10）。
原文：`.com.tw` 所在 rowspan=4 的價格格為「新申請 800元 續用 800元 轉入 800元」，該群組涵蓋 `.net.tw`、`.org.tw`、`.tw/.台灣（中/英）`；`.idv.tw` 為「新申請 400元 續用 400元 轉入 400元」。
限制：未重新抓取此 SPA（見方法）。

### 4. Cloudflare Registrar 不支援 .tw 系列（TLD 政策頁沒有 .tw）— **相符**

來源：`https://www.cloudflare.com/tld-policies/`（2026-11-10 重抓，約 690 KB）。
全文檢索 `.tw`、`idv`、`taiwan`、`taipei`：**0 筆命中**。

### 5. TWNIC 公告：2023-06-01 起 .tw／.台灣 WHOIS 聯絡資訊預設不揭露 — **相符**

來源：`https://www.twnic.tw/dnservice/policy/contents.php?id=20&lang=zh-tw`（2023/04/21 公告，2026-11-10 重抓）。
原文：「自112年6月1日起本中心將調整.tw/.台灣網域名稱之WHOIS查詢結果揭露方式：網域名稱之聯絡資訊(Contact)將預設為不揭露……註冊人可向受理註冊機構辦理聯絡資訊(Contact)揭露與否之異動作業。」
「預設」二字與報告一致（不是強制，報告 4.1 節也寫得正確）。

### 6. .tw 到期時程：到期翌日凍結、到期後第 33 天刪除；Gandi 寫無隔離期 — **相符（附時效註記）**

來源一（TWNIC）：`https://www.twnic.tw/dnservice/policy/contents.php?id=19&lang=zh-tw`（2026-11-10 重抓）。
原文：「(三)域名到期未繳續用管理費者之域名凍結時間：自到期日翌日執行凍結域名……(四)域名到期後未繳續用管理費者之域名刪除時間：自到期日後第33天執行刪除域名」。
來源二（Gandi）：`https://www.gandi.net/en/domain/tld/tw`（2026-11-10 重抓）。
原文：「Deletion process: No quarantine period」。
時效註記：TWNIC 這份公告標題日期為「中華民國106年7月25日公告」，原文本身不含「現行有效」的確認；報告 6.2 節已自行標註此點，判定相符但該時程的現行性需以 TWNIC 現況確認。

### 7. Cloudflare Registrar FAQ：gTLD 約 75 天後釋出，中間有寬限期與贖回期 — **相符**

來源：`https://developers.cloudflare.com/registrar/faq/`（2026-11-10 重抓）。
原文：「Domain names should be released after a period of 75 days, although the exact deletion timeline is ultimately determined by the domain's registry.」
另見階段表：「Day 1 - 30 : Grace Period」、「Day 31 - 40 : Suspension Period」、「Day 41 - 70 : Redemption Period」、「Day 71 - 75 : Pending Delete Period」。報告表格與此一致。

### 8. GitHub 文件：自訂 Actions workflow 部署不產生 CNAME、既有 CNAME 被忽略，網域在 repo 設定填寫 — **相符**

來源：`https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site`（2026-11-10 重抓）。
原文：「If you are publishing from a custom GitHub Actions workflow, no CNAME file is created, and any existing CNAME file is ignored and is not required.」（前文：在 repo Settings > Pages 的 Custom domain 欄填入網域。）

### 9. GitHub 文件：建議即使用 apex 也設 www，設好兩者後自動互相轉址 — **相符**

來源：`https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/about-custom-domains-and-github-pages`（2026-11-10 重抓）。
原文：「We recommend always using a www subdomain, even if you also use an apex domain.」以及「If you configure the correct records for each domain type through your DNS provider, GitHub Pages will automatically create redirects between the domains.」
報告寫「設好兩者的 DNS 後 GitHub 自動在 apex 與 www 之間轉址」，與原文一致。原文的前提是「正確設定各網域的 DNS 紀錄」，報告已隱含。

### 10. 實測：`curl -sI https://twbs.github.io/bootstrap/docs/` 回 301，Location 指向 `getbootstrap.com/docs/` — **相符**

本次實跑（2026-11-10，唯讀 HEAD 請求）回應標頭：

```
HTTP/2 301
server: GitHub.com
content-type: text/html
location: http://getbootstrap.com/docs/
x-github-request-id: 3B1E:2AC3:1019AA:125310:6AF29B44
x-github-edge-region: southeastasia
accept-ranges: bytes
age: 806
date: Tue, 10 Nov 2026 04:32:43 GMT
via: 1.1 varnish
x-served-by: cache-sin-wsat1880051-SIN
x-cache: HIT
x-cache-hits: 0
x-timer: S1794285163.354466,VS0,VE5
vary: Accept-Encoding
x-fastly-request-id: 94754679271e9872558f34134648f71d425ccf56
content-length: 162
```

註記：回應由快取提供（`x-cache: HIT`，`age: 806`），但 Location 與報告一致。Location 的 scheme 是 `http://`，報告 1b 節已正確記錄。此為單一公開站的觀察，不是官方保證（報告本身也如此標註）。

## 內部矛盾、不完整與佔位符

矛盾與不完整（依重要性排序）：

1. **domain-pricing.md §0 與 §4.1 不一致（需修正）**：§0 以「事實」寫「樣本中的個人 .idv.tw 姓名、地址、電話、Email 都顯示 'Redacted for Privacy'，不用另外加錢」；§4.1 卻寫「對『是否加錢』我沒有看到任何收費項，但註冊商端的實際結帳頁無法逐家驗證，未確認」。§0 的「不用另外加錢」超出證據，應改為「未見收費項，未確認」。
2. **domain-pricing.md §5.2 寫「續約 180 天」不完整**：ICANN 新 gTLD 基本合約 §2.10(b) 原文為續約漲價須提前 180 天通知，但同條 (i) 款規定：若新價格不高於先前已通知過的價格，只需 30 天通知。報告省略此例外。不影響「無價格上限」的主結論，但「續約漲價有 180 天緩衝」的說法偏強。
3. **domain-pricing.md §0 的預算前提無來源**：「US$99 ≈ NT$3,245，NT$4,000 扣掉後網域剩約 NT$755/年」中的 US$99 與 NT$4,000 在報告中沒有來源，也未標明是使用者提供。若這是使用者預算，應註明；若是假設，應標為假設。本次無法核對。
4. **domain-pricing.md §1.3 與引用的第二來源衝突**：Namecheap .com 續約採 15.98（比價站），報告自己記錄 WebSearch 摘要另有 14.78（2026-07 資料）；§8 排序表第 19 列直接採用 524 TWD，未在表中註記此衝突。
5. **排序表混用推估值**：domain-pricing.md §8 第 2 列採 366（推估現價），而 §1.2 的同一列原始列價為 343；表內以「列價 343」註記，但排序依推估值。可接受，但讀者需注意。

截斷：

1. **origin-options.md §3d**：「COMMIT ab73a956…」為截斷的 commit 雜湊，無法用於定位版本，但不影響 PSL 結論。
2. **domain-pricing.md §6.2**：引用 TWNIC 範例句時以「…」省略後半（「如到期凍結30日仍未繳費，執行刪除日為…」），是範例句的省略，不影響規則本身（規則原文在 (四) 已完整）。

佔位符殘留：未發現 TODO／TBD／待補／佔位字串。`…` 僅出現於上述兩處引文省略。

報告中的其他未確認項（如 Namecheap、Cloudflare 官方價、.tw 的 Porkbun 來源）已在各報告內自行標示「未確認」，本次未擴大核對。

## 附帶事項（操作紀錄）

- 本次在核對過程中，因 shell 工作目錄錯誤，曾將 8 個 curl 下載檔（`cffaq.html`、`cftld.html`、`gandi_tw.html`、`gh_about.html`、`gh_managing.html`、`pchome_tw.html`、`twnic19.html`、`twnic20.html`）寫入專案根目錄 （路徑略）。已確認這些檔案為未追蹤的新檔（`git status` 顯示 `??`，時間戳 12:33，大小與本次下載一致）後全數刪除，之後 `git status` 恢復為原本狀態（僅 `AGENTS.md` 修改與 `.impeccable/`、`.worktrees/`、`PRODUCT.md`、`docs/superpowers/` 未追蹤，皆為既有狀態，非本次產生）。
- 本次未修改兩份報告、未修改任何專案檔、未做 git 或 gh 寫入。
- 本次唯一的對外請求為唯讀 GET／HEAD（TWNIC、Cloudflare、GitHub、Gandi、PChome、bootstrap 站），未帶任何個人資訊。

## 來源清單（本次實際讀取）

- https://www.twnic.tw/dnservice/policy/contents.php?id=4&lang=zh-tw
- https://www.twnic.tw/dnservice/policy/contents.php?id=19&lang=zh-tw
- https://www.twnic.tw/dnservice/policy/contents.php?id=20&lang=zh-tw
- https://myname.pchomebiz.com.tw/shopping/price_tw.htm
- https://domain.hinet.net/#/dnprice （JS 落地檔 `raw/hinet_price.js`）
- https://www.cloudflare.com/tld-policies/
- https://developers.cloudflare.com/registrar/faq/
- https://www.gandi.net/en/domain/tld/tw
- https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site
- https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/about-custom-domains-and-github-pages
- https://twbs.github.io/bootstrap/docs/ （HEAD）
- ICANN 新 gTLD 基本登記局合約 §2.10（落地檔 `raw/base_ra.txt`，用於 domain-pricing §5.2 的附帶核對）
