---
version: 1
slug: "src-routes-season-group-page-svelte"
primary_target: "src/routes/[season]/[group]/+page.svelte"
related_targets: ["src/routes/+layout.svelte"]
---

# 球隊列表頁（/<賽季>/<組別>/）

範圍：Operate 模式；Issue #5 打通路徑的第一頁。讀者是手機上的球迷，任務是幾秒內找到自己支持的球隊。內容為該賽季該組別的球隊清單（4 到 8 隊），空清單顯示「這一季尚無球隊資料」。不動路由結構、匯出器、JSON schema；視覺世界見 `DESIGN.md`（2026-10-20 整站重設計建立）。

## Direction contract

THESIS：一頁只做「找到我的隊」。拒絕卡片牆與 hero 數字；單欄清單，隊標是唯一的彩色。
OWN-WORLD：2026-10-20 整站重設計後改依 `DESIGN.md`（賽後看板：淺灰藍底、白色大圓角卡片、靛藍只用在導覽與連結、數據磚與圖表用序列色）；本頁不另立世界。舊的「克制預設」世界已退役。
STORY：球迷看到站名與「非官方」註記，確認賽季與組別（頁首兩層導航，aria-current 標記所在），往下掃到自己的隊。
FIRST VIEWPORT：頁首（站名、非官方註記、賽季連結、組別切換）；標題「<賽季> <組別> 球隊」；清單從第一列開始可見，每列 48px 圓形隊標加球隊名稱，觸控目標 ≥ 44px；內容寬度上限 40rem 居中。
FORM：使用者選擇「克制預設，不投骰子」；無 concept-seed seed key。code-led。
FINISH：unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance.
