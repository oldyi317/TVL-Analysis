---
version: 1
slug: "src-routes-season-group-matches-page-svelte"
primary_target: "src/routes/[season]/[group]/matches/+page.svelte"
related_targets: ["src/routes/[season]/[group]/matches/[game]/+page.svelte","src/lib/components/MatchCard.svelte"]
---

# 賽事列表頁與賽事詳情頁（/<賽季>/<組別>/matches/、/matches/<game_id>/）

範圍：Operate 模式；Issue #8。讀者是手機上的球迷，任務是「找到某一場（通常是最近或某個月的）、一眼看到結果、點進去看逐局」。內容為該賽季該組別全部賽事，依日期排序、按月份分段；未開打的場次以「未開打」狀態呈現（目前無資料，但狀態先定義）。視覺世界見 `DESIGN.md`（2026-10-20 整站重設計建立）。

## Direction contract

THESIS：一頁只做「按月找到那一場」。拒絕篩選器、搜尋框與賽程表格；單欄賽事卡流，月份是唯一的導航層。
OWN-WORLD：2026-10-20 整站重設計後改依 `DESIGN.md`（賽後看板：淺灰藍底、白色大圓角卡片、靛藍只用在導覽與連結、數據磚與圖表用序列色）；本頁不另立世界。舊的「克制預設」世界已退役。
STORY：球迷確認賽季與組別，點月份跳到那一段，掃卡片看到粗體的勝方與局數比，點卡片進詳情看逐局；黃金決勝局在卡片與詳情都標明「黃金決勝局」且不與一般局混淆。
FIRST VIEWPORT：頁首（站名、組別、賽季）；h1「<賽季> <組別> 賽事」；h1 下方吸附的月份列（水平一列 chip，最新月在最右、可橫向捲動）；第一個月份的 h2 與第一張 MatchCard 可見。主要動作是點卡片（整張卡是連結，觸控目標 ≥ 44px）。
FORM：精確指定的既有世界延伸，直接 shape，不擲骰（無 seed key）。code-led。簽名互動：月份 chip 隨捲動位置自動標示當前月份（aria-current）。
FINISH：unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance.

## 詳情頁

h1「<主隊> 對 <客隊>」；上方「賽事列表」返回連結（回到該月份錨點）；MatchCard 原樣放頂部當摘要（無連結）；下方逐局比分 `<table>`：列為第 1–5 局與合計，欄為兩隊簡稱，贏的那局數字粗體；黃金決勝局為表格最後一列、以「黃金決勝局」命名且用 `--text-muted` 區隔。階段與週次、場館在卡片 meta 列。單場數據區塊留位置不顯示（第二期）。未開打：卡片顯示未開打＋時間，表格位置換成 EmptyState。

## 未決

- 開賽時間官網目前全部為空；有值時卡片以時間取代「未開打」字樣（沿用 #6 規則）。
