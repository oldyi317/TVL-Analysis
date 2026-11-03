---
version: 1
slug: "src-routes-season-group-leaders-page-svelte"
primary_target: "src/routes/[season]/[group]/leaders/+page.svelte"
related_targets: ["src/lib/components/LeaderBoard.svelte","src/lib/components/SiteNav.svelte"]
---

# 排行榜頁（/<賽季>/<組別>/leaders/）

範圍：Operate 模式；spec #106、本輪為子票 #108（tracer bullet：只接「總得分」一張卡，四王卡、分段控制、chip 列、比率榜留給 #109–#111）。讀者是賽後拿手機的球迷，任務是「這季誰是得分王、我支持的球員排第幾」，幾秒內看到前 10 名，再展開找人。內容全部來自匯出檔 `leaders.json`（名次已由匯出器算好，前端只渲染）；頁首一行「統計至 X 月 X 日 · 共 N 場」。視覺世界完全繼承 `DESIGN.md`「賽後看板」，不另立世界；設計 brief 2026-11-03 已於 #106 經 grilling 與 shape 訪談確認，本輪依票文直接 shape、不擲骰。

## Direction contract

THESIS：一頁只做「先給王，再給榜」。拒絕表格式全員清單、篩選、搜尋與圖表；每個項目一張卡、先列前 10 名、可展開全部，名次由資料決定、前端不重算。
OWN-WORLD：依 `DESIGN.md`（淺灰藍底、白色 20px 圓角卡片、靛藍只說「你在哪、能點哪」）。列 48px：名次（前三名沿用戰績表的 24px 靛藍 tint 圓點，其餘附註灰 600 數字）、32px 隊徽圓標（缺隊徽退回隊色圓標加簡稱）、球員名 700 連到球員頁加位置附註灰、右側數值 800 tabular。h2 在卡外。展開鈕 44px、section-link 樣式。
STORY：球迷看到 h1「2026-27 男子組 排行榜」與統計範圍，掃總得分卡認出得分王與自己的球員，點名字進球員頁；人不在前 10 就按「展開全部（N 人）」找，再「收合」。整組沒有單場數據時只看到「本季尚無單場數據」。
FIRST VIEWPORT（手機 390）：頁首殼層 → h1 → 14px 附註灰「統計至 11 月 2 日 · 共 6 場」→ h2「總得分」→ 白卡內前 10 列（第 1–3 列名次圓點可見）。桌面 1440：sidebar 左、內容區單卡 ≥40rem 兩欄、≥64rem 三欄格線（本輪只有一張卡，格線先就位）。主要動作是點球員名。
FORM：使用者釘定的 brief（#106 Implementation Decisions「設計 brief」），依 new-work.md §3「precisely specified narrow request」直接 shape，無 seed key。code-led。簽名互動：卡底展開鈕只影響該卡（aria-expanded），第 10 名並列時全部顯示、不截；首屏卡片一次性 rise，reduced-motion 全停。
FINISH：unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance.

## 未決

- 導覽 icon 圖形留給實作（本輪選三階頒獎台 `podium`）。
- 四王卡、分段控制、吸頂 chip、比率榜門檻文案與「榮耀」Tag 變體、DESIGN.md 榮耀標籤規則：#109–#111。

## #109 擴充方向（2026-11-03）

THESIS：先給王，再給榜；七個計數榜沿用既有卡片。
OWN-WORLD：維持賽後看板，榮耀標籤沿用 warn-tint／warn-ink。
STORY：首屏辨識四王，點球員或用吸頂 chip 跳項目。並列共用稱號、每人各自連結。
FIRST VIEWPORT：h1、統計範圍、本季四王（手機四列 56px，桌面四欄）、七項 chip、計數榜。
FORM：#106 已確認 brief，code-led；深淺模式與 reduced-motion 沿用全站。
FINISH：unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance.

## #110 比率榜實作紀錄（2026-11-03）

本節追加目前程式碼的互動事實；上方 #108／#109 為各階段已確認方向，保留原文。視覺仍繼承 `DESIGN.md`，未新增設計 token 或改變全站系統。

- 四王之後顯示「計數榜／比率榜」分段，預設計數榜；兩個按鈕以 `aria-pressed` 表達選中，沿用組別分段的內嵌面板軌道、白色選中面與靛藍文字。
- 比率榜依匯出資料呈現攻擊成功率、發球得分率、接發球到位率、防守到位率、舉球到位率、場均得分、每局攔網；整組缺局數資料時匯出器省略每局攔網，呈現六卡。各榜 h2 下直接顯示匯出器提供的門檻附註；主要比率值下顯示「分子／分母」，兩者皆為 14px、600、附註灰，主要數字維持 16px、800、tabular。
- 切換分段時只顯示該類型的卡片與項目 chip；chip 列以頁底色吸頂、橫向捲動。點擊 chip 使用 `#b-<項目鍵>` 跳至卡片；直接開啟或變更有效 hash 會先切到對應分段再捲動。捲動追蹤目前項目並以 `aria-current="location"` 標記；桌面同列多卡保留讀者已點的項目，到頁底標記最後項目。
- 單榜無達門檻者保留標題與門檻，卡內顯示「尚無球員達門檻」；整組沒有榜單時顯示「本季尚無單場數據」。前十名含第十名並列、逐卡展開／收合與球員連結沿用既有榜卡；切換分段會重建卡片，展開狀態回到收合。
- 卡片維持手機單欄、40rem 起兩欄、64rem 起三欄。吸頂列與收合後卡片跳轉預留相同頂端距離（44px 觸控列 + 16px 上下留白 + 16px 間距），避免標題被遮住。

## #110 Finish review（2026-11-03）

**Verdict：通過。** 依 issue #110 驗收清單核對匯出門檻、七項比率、並列排序、無局數時省略每局攔網、空榜狀態、分段與 hash 導覽；實作符合 brief。手機／桌面、深淺色、鍵盤切換與返回錨點由 Playwright 驗證；未發現需阻擋上線的互動或版面問題。設計規則已同步至 `DESIGN.md` 的「排行榜卡 LeaderBoard」。本次無新增 raster，因此沒有新增素材 provenance。

驗證：`python -m pytest -q`（366 passed）、`npm run check`（0 errors／0 warnings）、`npm run test:unit`（209 passed）、production build 完成、Playwright（62 passed／6 skipped）。Playwright 需先啟動 `npm run preview` 再執行 `npx playwright test`；直接執行串接指令時其 webServer 啟動失敗，分開啟動後全數可執行。Build 有 Vite 大型 chunk 提示，未由本票引入或處理。

核對來源：`src/routes/[season]/[group]/leaders/+page.svelte`、`src/lib/components/LeaderBoard.svelte`、`src/lib/leaders.ts`、`src/lib/advanced.ts`、`../src/export/leaders.py`、`src/app.css`、`PRODUCT.md`、`DESIGN.md` 與 `.impeccable/design.json`。此為原始碼紀錄，不取代瀏覽器驗證。
