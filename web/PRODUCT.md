# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

一般 TVL 球迷，手機閱讀為主。典型情境是賽後或賽中拿起手機查戰績與自己支持的球隊、球員的數據；要在幾秒內找到自己的隊。

次要讀者（未確認、不特別設計）：教練、球員、媒體。

## Product Purpose

把每日爬蟲累積的 TVL 企業排球聯賽資料（賽季、球隊、賽事、週登錄、單場數據）整理成一般球迷在手機上好讀的靜態網站。成功 = 球迷打開就能找到當季、自己的組別與球隊，資料與官網一致且每日更新。

本站取代舊的 Streamlit 儀表板（ADR-0001）。Issue #5 只打通「資料庫 → JSON → 靜態頁 → GitHub Pages」的路徑，第一頁是球隊列表。

## Positioning

官網只有單場資料；本站把所有場次串起來，做出跨場次累積的球員與球隊數據（PR 值、趨勢、戰績）。這是官網拿不到的東西。

## Operating Context

- 資料來源：TVL 官網與外部統計系統，每日 GitHub Actions 爬蟲寫入 SQLite 並 commit。
- 建置時由 Python 匯出器產 JSON，前端只讀 JSON；沒有執行期 API、沒有登入、沒有使用者輸入。
- 部署在 GitHub Pages 子路徑 `/TVL-Analysis/`，每次 push main 重建。
- 網址結構 `/<賽季>/<組別>/...`，組別值為 men、women；根路徑導向當季男子組；每頁頂部可切換組別並停留在同一頁面類型。
- 當季 = seasons.json 的 current_cup_id（匯出器以 cup_id 最大者判定）。

## Capabilities and Constraints

- 第一期（本票）：球隊列表頁，列出該賽季該組別的球隊，標記為隊色加簡稱的圓形。
- 後續（ADR-0001 第二期）：戰績、賽事、球隊頁、球員頁、賽事詳情；圖表預計用 ECharts。
- 球隊身分唯一來源是 DB 的 teams 表：short_name、color（hex）、is_active。非現役球隊顯示在該季戰績與列表，但不給球隊頁（ADR-0002）。
- 男女組 team_id 會重複，識別一律用 (team_id, group)。
- 資料品質原則：只標記與警告異常，不插補、不竄改原始數據。
- 用詞依 repo 根目錄 `CONTEXT.md` 詞彙表：用「組別、賽季、賽事、戰績、週登錄、單場數據、進階數據、PR 值、賽前勝率、黃金決勝局」；不用「性別、gender、cup、盃、roster、Box Score、預測、standing」等 Avoid 詞。
- 全站繁體中文。

## Brand Commitments

- 站名暫定「TVL 數據」。
- 必須註明非官方：資料整理自官網，非聯賽官方網站。
- 不使用球隊或聯賽的官方 logo；球隊識別只靠 DB 的隊色與簡稱。

## Evidence on Hand

- 真實資料：`data/db/tvl_database.db`（兩個賽季、12 支球隊、約 170 場賽事、逐場球員統計），由匯出器產成 `web/src/lib/data/*.json`（不進 git）。
- 沒有球隊 logo、球員照片、場館照片等圖像資產；不得自行生成或暗示有這些素材。
- 沒有使用者見證、流量數據、媒體報導；不得編造。

## Product Principles

1. 手機優先：單手、豎屏、光線不佳的看台或客廳是主要場景。
2. 找得到比看得美重要：賽季、組別、球隊三層導航每頁可見。
3. 資料忠實：顯示值與 DB 一致，缺值就顯示缺值，不猜。
4. 靜態與可預測：所有頁面建置時產生，URL 可分享、可書籤。
5. 用詞一致：介面文案遵守詞彙表，不混用同義詞。

## Accessibility & Inclusion

- 隊色僅作輔助識別，任何靠顏色區分的地方都要有文字（簡稱）。
- 圓形隊標上的文字需達可讀對比；隊色由 DB 給定、不可改，對比不足時用文字描邊或底色處理而非換色。
