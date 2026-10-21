---
version: 1
slug: "src-routes-layout-svelte"
primary_target: "src/routes/+layout.svelte"
related_targets: ["src/app.css","src/lib/components/SiteHeader.svelte","src/lib/components/SiteNav.svelte"]
---

# 全站殼層與視覺世界（src/routes/+layout.svelte，2026-10 整站重設計）

範圍：Operate 模式；整站替換視覺世界（取代 issue #5／#6 的「克制預設」世界）。讀者是手機上的球迷，任務不變：幾秒內找到當季、組別、自己的隊與球員。使用者 2026-10-20 釘定美學參考 `web/.impeccable/reference/redesign-2026-10-reference.png`（運動即時比分 dashboard），釘的是美學不是資訊架構。使用者同日確認：桌面 ≥1024px 用左側 sidebar、手機維持頂部導覽結構、字型維持系統字型、深淺兩套都做。細部規格在本次 session 的 design-spec（建置後由 DESIGN.md 承接）。

## Direction contract

THESIS：全站是一面「賽後看板」：淺灰藍桌面上的白色大圓角卡片，靛藍只說「你在哪、能點哪」，粉／橘／靛三色只給數據本身。拒絕的預設：灰底單一黑色強調、彩色左邊條卡片、每頁一樣的垂直清單。
OWN-WORLD：底 #F3F4F9、卡 #FFFFFF radius 20px 軟陰影、字 #262B4A、主色 #4F5BD5（tint #E9EBFB）、序列 #5B61E0／#F2558E／#F6A04D；深色 #12141F／#1B1E2E 卡帶 1px 邊線、主色 #7B82F0。系統字型；數字 800 tabular。隊徽圓形白底；隊色只在無隊徽時的圓標。手寫 24px stroke icon 一套。
STORY：球迷打開看到站名與非官方註記、組別分段控制、賽季 chip，點 pill 到五個頁面；桌面上同一組東西排在左側 sidebar，內容區變寬可雙欄。每場比賽都以同一張「對陣式」卡出現，看一眼就知道誰贏、幾比幾、下一場誰有利。
FIRST VIEWPORT（手機 390）：白色頁首卡（站名＋註記、組別分段控制、賽季 chip、頁面 pill 列）→ h1「2026-27 男子組」→ h2「最新比分」→ 第一張對陣卡完整可見（meta 列、左右隊徽 56px＋隊名、中間「3 : 1」）。桌面 1440：左 15.5rem sidebar（站名、註記、組別、賽季、直排導覽含 icon，選中項靛藍 tint＋右側指示條）；右側 h1 與雙欄：左欄最新比分對陣卡、右欄戰績摘要卡，下一場卡帶賽前勝率甜甜圈在左欄下方。主要動作是點對陣卡。
FORM：使用者釘定的方向（依 new-work.md §3「user- or brief-pinned direction beats the roll」條款未跑 concept-seed，無 seed key；mode rules 讀自 mode-operate.md）。code-led。簽名互動：全站統一的對陣式賽事卡，整卡可點、hover 升起；載入時首屏卡片一次性錯開升入。
FINISH：unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance.

## 未決
- 桌面 sidebar 的賽季 chip 在賽季超過四個時改成下拉；目前最多三個，先用 chip。
