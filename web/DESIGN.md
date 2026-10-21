---
name: TVL 數據
description: 球迷在手機上看的賽後看板：淺灰藍桌面、白色大圓角卡片、靛藍只說「你在哪、能點哪」
colors:
  bg: "#f3f4f9"
  surface: "#ffffff"
  surface-2: "#eef0f8"
  text: "#262b4a"
  text-muted: "#5f6788"
  line: "#c9cde2"
  accent: "#4f5bd5"
  accent-strong: "#3d48bf"
  accent-tint: "#e9ebfb"
  on-accent: "#ffffff"
  series-1: "#5258da"
  series-2: "#e2457f"
  series-3: "#cc6a00"
  series-4: "#6b72d4"
  series-5: "#0b9279"
  tile-1: "#5258da"
  tile-2: "#e2457f"
  tile-3: "#cc6a00"
  win: "#12804f"
  win-tint: "#dff5ea"
  loss: "#cc3347"
  loss-tint: "#fde4e8"
  warn: "#b45309"
  warn-tint: "#fff3c4"
  warn-ink: "#7a5a00"
  bg-dark: "#12141f"
  surface-dark: "#1b1e2e"
  surface-2-dark: "#242838"
  text-dark: "#eceef8"
  text-muted-dark: "#9ba1bc"
  line-dark: "#3a405a"
  accent-dark: "#7b82f0"
  accent-strong-dark: "#959bf5"
  accent-tint-dark: "#262b4a"
  on-accent-dark: "#0f1120"
  series-1-dark: "#7b82f0"
  series-2-dark: "#f66fa0"
  series-3-dark: "#f8b36a"
  series-4-dark: "#5f66b8"
  series-5-dark: "#3fd2b6"
  win-dark: "#3ed48e"
  win-tint-dark: "#153a2a"
  loss-dark: "#f2728a"
  loss-tint-dark: "#3d1c24"
  warn-dark: "#fbbf24"
  warn-tint-dark: "#3a2a0a"
  warn-ink-dark: "#fde68a"
typography:
  display:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', sans-serif"
    fontSize: "1.375rem"
    fontWeight: 800
    lineHeight: 1.5
  headline:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', sans-serif"
    fontSize: "1.125rem"
    fontWeight: 700
    lineHeight: 1.5
  title:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', sans-serif"
    fontSize: "1rem"
    fontWeight: 700
    lineHeight: 1.5
  body:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', sans-serif"
    fontSize: "0.875rem"
    fontWeight: 600
    lineHeight: 1.5
    letterSpacing: "normal"
  number-md:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', sans-serif"
    fontSize: "1.5rem"
    fontWeight: 800
    lineHeight: 1.1
    fontFeature: "tabular-nums"
  number-lg:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', sans-serif"
    fontSize: "2.25rem"
    fontWeight: 800
    lineHeight: 1
    fontFeature: "tabular-nums"
rounded:
  md: "0.875rem"
  lg: "1.25rem"
  full: "999px"
spacing:
  "1": "0.25rem"
  "2": "0.5rem"
  "3": "0.75rem"
  "4": "1rem"
  "5": "1.5rem"
  "6": "2rem"
  "7": "3rem"
components:
  card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
    rounded: "{rounded.lg}"
    padding: "{spacing.4}"
  panel:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.text}"
    rounded: "{rounded.md}"
    padding: "{spacing.4}"
  chip:
    backgroundColor: "transparent"
    textColor: "{colors.text-muted}"
    typography: "{typography.body}"
    rounded: "{rounded.full}"
    padding: "0 {spacing.3}"
    height: "2.75rem"
  chip-hover:
    backgroundColor: "{colors.accent-tint}"
    textColor: "{colors.text-muted}"
  chip-selected:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.on-accent}"
  segment-track:
    backgroundColor: "{colors.surface-2}"
    rounded: "{rounded.full}"
    padding: "{spacing.1}"
  segment-item:
    backgroundColor: "transparent"
    textColor: "{colors.text-muted}"
    typography: "{typography.title}"
    rounded: "{rounded.full}"
    height: "2.25rem"
  segment-item-selected:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.accent}"
  nav-pill:
    backgroundColor: "transparent"
    textColor: "{colors.text-muted}"
    typography: "{typography.title}"
    rounded: "{rounded.full}"
    padding: "0 {spacing.3}"
    height: "2.75rem"
  nav-pill-current:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.on-accent}"
  nav-side-item:
    backgroundColor: "transparent"
    textColor: "{colors.text-muted}"
    typography: "{typography.title}"
    rounded: "{rounded.md}"
    padding: "0 {spacing.3}"
    height: "2.75rem"
  nav-side-item-current:
    backgroundColor: "{colors.accent-tint}"
    textColor: "{colors.accent}"
  tag:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.text-muted}"
    typography: "{typography.label}"
    rounded: "{rounded.full}"
    padding: "2px 10px"
  tag-win:
    backgroundColor: "{colors.warn-tint}"
    textColor: "{colors.warn-ink}"
  tag-pending:
    backgroundColor: "{colors.accent-tint}"
    textColor: "{colors.accent}"
  tag-result-w:
    backgroundColor: "{colors.win-tint}"
    textColor: "{colors.win}"
  tag-result-l:
    backgroundColor: "{colors.loss-tint}"
    textColor: "{colors.loss}"
  stat-tile:
    backgroundColor: "{colors.tile-1}"
    textColor: "#ffffff"
    typography: "{typography.number-lg}"
    rounded: "{rounded.lg}"
    padding: "{spacing.4}"
  section-link:
    backgroundColor: "transparent"
    textColor: "{colors.accent}"
    typography: "{typography.label}"
  section-link-hover:
    textColor: "{colors.accent-strong}"
  data-note:
    backgroundColor: "{colors.warn-tint}"
    textColor: "{colors.warn-ink}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: "{spacing.3} {spacing.4}"
  empty-state:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.text-muted}"
    rounded: "{rounded.md}"
    padding: "{spacing.6} {spacing.4}"
  team-badge:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.full}"
    size: "3rem"
  rank-dot-top:
    backgroundColor: "{colors.accent-tint}"
    textColor: "{colors.accent}"
    typography: "{typography.title}"
    rounded: "{rounded.full}"
    size: "1.5rem"
---

# Design System: TVL 數據

## Overview

**Creative North Star: "賽後看板"**

球迷在看台或客廳拿起手機，想在幾秒內知道誰贏、幾比幾、自己的隊排第幾。整站因此長成一面即時比分看板：淺灰藍的桌面上放著白色大圓角卡片，卡片之間靠底色差與一層極淡的陰影分開，沒有彩色邊條、沒有一頁一樣的垂直清單。靛藍只有一個職責——說「你在哪、可以點哪」：導覽選中、區塊右上的小連結、比分中間的冒號、PR 值進度條。粉、橘、靛三色只給數據本身（數據磚與圖表序列），從不拿來裝飾介面。

密度偏緊但不擠：卡片內距 16px（桌面 24px），列表卡片間距 12px，表格列高 48px，所有可點的東西至少 44px 高。文字一律系統字型（使用者決定，不載網路字型），正文最小 16px，數字用 800 字重加 tabular 數字讓比分與戰績欄對齊。深淺兩套跟隨系統設定，深色模式把卡片陰影換成 1px 邊線。

美學來源是使用者釘定的運動即時比分 dashboard 參考截圖（`.impeccable/reference/redesign-2026-10-reference.png`）：釘的是卡片、配色、圓角、圖表風格，不是其資訊架構。確認拒絕的預設：灰底單一黑色強調、隊色左邊條卡片、每頁一樣的垂直清單、emoji 或 unicode 符號當 icon。

**Key Characteristics:**
- 淺灰藍底 + 白色 20px 圓角卡片，淺色靠軟陰影、深色靠 1px 邊線分層
- 靛藍只標「位置與動作」；粉／橘／靛三色只給數據
- 系統字型；數字 800 tabular，正文 ≥16px
- 全站統一的「對陣式」賽事卡：左右隊徽、中間局數比，整卡可點、hover 升起
- 隊徽圓形白底；無隊徽才退回隊色圓標＋簡稱；隊色來自資料庫、不可改
- 手機頂部導覽（分段控制、chip、pill 列），桌面 ≥64rem 變左側 17rem sidebar
- 對比值由 `src/lib/tokens.test.ts` 鎖住：文字對 4.5:1、分隔線 1.5:1、圖表序列 3:1

## Colors

一組靛藍系中性色加一個靛藍主色，配粉／橘／靛三條資料序列與勝敗警示語意色；淺深兩套各自獨立定義，鍵名相同。

### Primary
- **靛藍 Accent**（`accent`，深色 `accent-dark`）：導覽選中底色（手機 pill）、桌面 sidebar 選中文字與右側 3px 指示條、區塊標題右側的小連結、返回連結、比分冒號、PR 值進度條、甜甜圈進度、focus outline。只說「你在哪、可以點哪」。
- **靛藍 Strong**（`accent-strong`）：連結 hover 時的文字色，配 underline offset 0.2em。
- **靛藍 Tint**（`accent-tint`）：選中項底色（桌面 sidebar、前三名名次圓點）、chip hover、甜甜圈與 PR 條的軌道、`::selection` 底色、「未開打」標籤底。深色模式的 tint 正是淺色模式的 `text` 值。
- **On Accent**（`on-accent`）：靛藍底上的文字；深色模式改為近黑 `#0f1120`，因為深色主色變淺。

### Secondary（資料序列）
- **序列靛**（`series-1`）、**序列粉**（`series-2`）、**序列橘**（`series-3`）：數據磚底色（勝／敗／積分、三大項）與圖表前三條序列。深色模式整組換成更亮的對應色。
- **序列淡靛**（`series-4`）、**序列青**（`series-5`）：只在 ECharts 圖表第四、五條序列出現，介面上不使用。
- **數據磚底**（`tile-1/2/3`）：與淺色序列靛／粉／橘同值，但**兩種模式同值**——它是彩色面，白字對比在深色模式也要固定；`BigStat` 與球隊頁的磚只讀這三個 token，不讀 `series-*`。

### Tertiary（語意）
- **勝**（`win`／`win-tint`）、**敗**（`loss`／`loss-tint`）：戰績表「近五場」的勝敗小標籤（底 tint、字主色）。賽事卡上的「勝」標籤**不用**這組，用警示黃（見 Named Rules）。
- **警示黃**（`warn`／`warn-tint`／`warn-ink`）：資料備註 DataNote（底 tint、字 ink），以及賽事卡勝方旁的「勝」標籤。

### Neutral
- **桌面底**（`bg`）：頁面背景；賽事列表的月份 chip 列吸頂時也用它當純色底。
- **卡面**（`surface`）：卡片、頁首／sidebar、分段控制選中項、ECharts tooltip 底。
- **內嵌面板**（`surface-2`）：卡片內要圈出的區域、分段控制軌道、空狀態、球員背號圓點、戰績展開列、階段／黃金決勝局標籤底。
- **深藍黑文字**（`text`）：標題、正文、勝方數字。
- **附註灰**（`text-muted`）：日期、場館、欄名、位置、敗方隊名與數字、未選中的導覽項。
- **分隔線**（`line`）：表格列線、卡片內的次級分隔、深色模式卡片邊線（`--card-edge`）、隊徽圓標的 1px 內描邊。

### Named Rules
**The One Job Rule.** 靛藍只出現在「我在哪、可以點什麼」的位置：選中狀態、連結、焦點、進度。它不當裝飾、不當資料色；資料色只從 `series-*` 拿。

**The Yellow Win Rule.** 賽事卡勝方旁的「勝」標籤用 `warn-tint`／`warn-ink`（參考截圖的黃色 WIN 標），戰績表近五場的 W／L 才用 `win`／`loss` 語意色。兩者不互換。

**The Contrast Gate Rule.** `src/lib/tokens.test.ts` 直接解析 `app.css`，淺深兩套各自檢查：`text`、`text-muted`、`win`、`loss`、`warn`、`focus` 對 `bg` 與 `surface`、`on-accent` 對 `accent`、`warn-ink` 對 `warn-tint` 皆 ≥4.5:1；`line` 對 `surface` ≥1.5:1；每條 `series` 對 `bg` ≥3:1；兩套模式必須定義同一組鍵名。`text-muted`、`line`、`series-*`、`win`、`loss` 的值都是為了過這道門而比原始規格更深的，改色先跑 `npm run test:unit`。

**The Two Copies Rule.** 色值只准在三處出現：`src/app.css`（唯一來源）、`src/lib/chart/theme.ts`（ECharts 不吃 CSS 變數）、`src/lib/color.ts` 的 `SURFACE`（隊色對比計算）。後兩處由 `tokens.test.ts` 鎖住必須與 `app.css` 相同；其他檔案一律 `var(--token)`。

**The Team Colour Rule.** 隊色由資料庫 `teams.color` 給定、不可改。`teamPalette()` 只衍生四個值：填色、填色上的黑或白字（WCAG 亮度門檻 0.179）、淺色與深色表面上達 4.5:1 的文字色（只動明度、不動色相）。靠隊色區分的地方一定有簡稱文字。

## Typography

**Display Font:** system-ui（依序退回 -apple-system、Segoe UI、Roboto、Noto Sans TC、PingFang TC、Microsoft JhengHei、sans-serif）
**Body Font:** 同上，單一字型堆疊
**Label/Mono Font:** 無獨立字型；數字靠 `font-variant-numeric: tabular-nums`（設在 `html`）對齊

**Character:** 系統字型是使用者的決定（不載網路字型），個性全靠字重與尺寸對比：標題 700／800、附註 600、數字 800。中文與英數混排 letter-spacing 維持 normal；只有隊徽圓標內的簡稱用 0.02em。

### Hierarchy
- **Display**（800，`--text-xl` 1.375rem，≥64rem 升到 `--text-2xl` 1.75rem）：每頁唯一的 h1，如「2025-26 男子組」、球隊名、球員名；`text-wrap: balance`。
- **Headline**（700，`--text-lg` 1.125rem）：區塊標題 h2（最新比分、戰績、賽事），放在卡片**外**上方，與卡片間距 12px；右側可配一個 `--text-sm` 700 靛藍小連結。桌面 sidebar 站名也是這級但 800。
- **Title**（700，`--text-md` 1rem）：h3、隊名、球員名、導覽項、分段控制。
- **Body**（400，`--text-md` 1rem，行高 1.5）：正文與表格內容；最小 16px，不往下降。`--measure: 40rem` 保留給文字段落（目前無段落使用）。
- **Label**（600 或 700，`--text-sm` 0.875rem，附註灰）：日期、場館、欄名、位置、數據磚標籤、狀態標籤（700）。只給附註，不給正文。
- **Number MD**（800，`--num-md` 1.5rem）：比分冒號、戰績展開列的大數字、甜甜圈 88px 中心數字、單場數據的主要欄。
- **Number LG**（800，`--num-lg` 2.25rem，行高 1）：賽事卡局數比「3 : 1」、數據磚大數字（≥40rem）。

### Named Rules
**The Heavy Number Rule.** 任何「數字就是結論」的地方（比分、勝敗場、積分、PR 值）用 `--weight-num: 800` 加 tabular；敗方或次要的數字改 `text-muted` 而不是減字重。

**The 16px Floor Rule.** 正文與表格內容最小 16px；`--text-sm`（14px）只給日期、場館、欄名、位置這類附註與標籤。

## Layout

手機優先、單欄。內容欄（`.wrap`，含 `<main>` 與頁尾 `SiteFooter`）內距 16px、底部 48px，頁尾與主內容間距 48px；頁面內容由 h1 起頭，區塊 h2 在卡片外，列表卡片用 grid、間距 12px（`.cards`）。間距以 4px 為基準的七階（`--space-1..7`：4／8／12／16／24／32／48px），最常用的是 12px（元素間）、8px（內部）、16px（卡片內距）。

斷點只有三個：
- **≥40rem**：球隊列表變兩欄；數據磚內距從 12px 升到 16px、數字從 `--num-md` 升到 `--num-lg`。
- **≥48rem**（單場數據表）／**容器 ≥44rem**（戰績表，用 `@container standings`）：隱藏「詳細」展開鈕，直接顯示寬欄。
- **≥64rem**：`body` 變成 `grid-template-columns: var(--sidebar) minmax(0, 1fr)`（`--sidebar: 17rem`）；`<header>` 變成 sticky、`100dvh` 高、可捲的左側 sidebar（內距 24px，右側 1px `--card-edge` 邊線，淺色模式透明）；內容欄 `.wrap` 內距 32px／48px、`max-inline-size: 72rem`、靠左貼 sidebar 不置中，頁尾隨內容欄落在 sidebar 右側。卡片內距升到 24px，h1 升到 1.75rem。首頁雙欄 `minmax(0,1fr) 22rem`（左：最新比分＋下一場；右：戰績摘要）、賽事列表雙欄、球隊列表三欄。

導覽在兩種版型中是同一份 DOM（SiteHeader 內的 GroupSwitch、SeasonSwitch、SiteNav），只用 CSS 改排法：手機是分段控制全寬 → 賽季 chip 換行 → 頁面 pill 橫捲；桌面是直排，chip 改 `auto-fit minmax(3.75rem,1fr)` 網格，導覽項顯示 icon。

觸控目標 `--tap: 2.75rem`（44px）：chip、pill、導覽項、展開鈕、返回連結全部 ≥44px 高。賽事列表的月份 chip 列吸頂（`position: sticky`，純色 `--bg` 底、不模糊），章節 `scroll-margin-block-start` 預留列高 + 16px。

## Elevation & Depth

混合式：淺色模式靠一層極淡的雙層陰影分出卡片，深色模式陰影歸零、改用 1px `--line` 邊線（`--card-edge` 在淺色是 `transparent`、深色是 `var(--line)`，所以 `.card` 永遠帶 `border: 1px solid var(--card-edge)`，不必改元件）。卡片內的次級分層不再疊卡，只用 `--surface-2` 內嵌面板或 1px `--line` 細線。

### Shadow Vocabulary
- **Card**（`box-shadow: 0 6px 20px rgba(38, 43, 74, 0.07), 0 1px 2px rgba(38, 43, 74, 0.05)`；深色 `none`）：`.card`、手機頁首、分段控制選中項。陰影色是 `text` 的靛藍黑，不是純黑。
- **Pop**（`box-shadow: 0 12px 32px rgba(38, 43, 74, 0.14)`；深色 `0 12px 32px rgba(0, 0, 0, 0.5)`）：可點卡片 hover 升起（配 `translateY(-2px)`）、ECharts tooltip。
- **Badge inset**（`inset 0 0 0 1px var(--line)`）：有隊徽的圓標用內描邊，不用外陰影。

### Named Rules
**The Lift-On-Hover Rule.** 陰影只在兩個狀態出現：靜止的 Card，與 hover 的 Pop。沒有第三級。hover 升起只在 `@media (hover: hover)` 與非 reduced-motion 下位移。

**The No Nesting Rule.** 卡片不包卡片。卡片裡要分區就用 `--surface-2` 面板（radius md）或 1px `--line` 線。

## Shapes

圓角三級加圓形：卡片 20px（`--radius-lg`）、內嵌面板與桌面導覽項 14px（`--radius-md`）、chip／標籤／分段控制／PR 條 999px（`--radius-full`），隊徽、名次圓點、背號圓點、甜甜圈是正圓。ECharts tooltip 12px、長條頂端 `[6, 6, 0, 0]`。`--radius-sm: 0.5rem` 定義了但目前沒有元件使用。

邊線極少：淺色模式卡片無邊線，表格只有列線（1px `--line`），有隊徽的圓標 1px 內描邊，深色模式卡片 1px 邊線。沒有彩色左邊條、沒有外框按鈕。Icon 全部是手寫 inline SVG：24 viewBox、`fill="none"`、`stroke="currentColor"`、stroke-width 1.75、round cap／join，預設 20px，集中在 `Icon.svelte`（home、calendar、trophy、shield、user、chevron-left、arrow-up-right、map-pin）。

## Components

### 卡片 Card
純白面、20px 圓角、Card 陰影（深色改邊線）、內距 16px（≥64rem 24px）。`a.card` 或 `MatchLink` 包起來時 hover `translateY(-2px)` + Pop 陰影，160ms ease-out。區塊 h2 放卡片外。戰績表的卡片內距縮為 8px／12px（容器 ≥44rem 時 12px／16px）讓表格貼邊。

### 內嵌面板 Panel／空狀態 EmptyState
`--surface-2` 底、14px 圓角、內距 16px；空狀態內距 32px／16px、置中、附註灰文字、標題與提示都是 16px。

### Chip（賽季、月份跳轉、指標切換）
藥丸形、透明底、附註灰 600、高 44px、左右內距 12px；hover 靛藍 tint；選中（`aria-current` 或 `aria-pressed="true"`）靛藍底白字。桌面 sidebar 的賽季 chip 縮為 36px 高、14px 700、置中。

### 分段控制 GroupSwitch（組別）
軌道 `--surface-2` 藥丸形、內距 4px；兩個等寬項目各 36px 高、700；未選附註灰（hover 轉 `text`），選中白底（深色 `surface`）+ Card 陰影 + 1px `--card-edge` 內 outline + 靛藍文字。focus 時 outline 換回靛藍、offset 2px。

### 導覽 SiteNav
手機：一列可橫捲的 pill（同 chip 規則但 700、`--text-md`、icon 隱藏），選中靛藍底白字。桌面 ≥64rem：直排，每項 44px 高、14px 圓角、icon 20px + 文字、靠左；選中改成靛藍 tint 底 + 靛藍字，並以 `::after` 在 sidebar 右緣畫 3px 寬、上下內縮 8px 的靛藍指示條（絕對定位到 `-24px`，不是 border）。

### 標籤 Tag
藥丸形、14px 700、`padding: 2px 10px`、`nowrap`。預設 `surface-2`／附註灰（階段、週次、黃金決勝局）；`tag-win` 警示黃（賽事卡「勝」）；`tag-pending` 靛藍 tint／靛藍（「未開打」）；戰績表近五場 `.w`／`.l` 用 `win-tint`／`win`、`loss-tint`／`loss`，內距改 `0 8px`。

### 區塊連結 Section link
h2 同列右側，`--text-sm` 700 靛藍；hover `accent-strong` + underline。返回連結（「賽事列表」「球隊列表」）同色、600、44px 高、chevron icon。

### 數據磚 StatTile
20px 圓角、`--tile` 底色（`tile-1/2/3` 擇一，兩種模式同值）、白字；右上 96px 與左下 72px 兩個 `rgba(255,255,255,0.14)` 半透明圓用 `::before/::after` 溢出裁切。內容：標籤 14px 700 白 90% → 大數字 `--num-lg` 800 → 可選附註。只用在三個以內並列的數字（勝／敗／積分、三大項）；超過三個用白卡內清單。深色模式保留彩色面（底色不換）。

### 甜甜圈 WinRing
純 SVG 圓環，預設 56px，首頁下一場大卡 88px、小卡 40px；線寬 `max(5, size/8.8)`；軌道靛藍 tint、進度靛藍、圓端點；載入時 `stroke-dashoffset` 600ms ease-out 填入。中心數字 800（≥80px 用 `--num-md` 並顯示小字標籤）。只在賽前勝率與球隊頁勝率出現，不用 ECharts。

### 隊徽圓標 TeamBadge
正圓，四級：sm 32px、md 48px、lg 64px、xl 96px（賽事卡 md 放大到 56px）。有隊徽（`static/logos/{group}-{team_id}.png`）時白底 + 1px `--line` 內描邊，圖片 82% `object-fit: contain`；無隊徽或載入失敗退回隊色填色 + 黑或白簡稱（≤3 字單行、4 字 2+2、更長在中英交界斷行）。`aria-hidden`，名稱由旁邊文字承載。

### 賽事卡 MatchCard（簽名元件）
第一列 meta：左「日期 星期 時間 · 場館」附註灰 14px，右階段標籤。主體三欄 `minmax(0,1fr) auto minmax(0,1fr)`：左主隊、右客隊各為圓標（56px；lg 64px）上、隊名 700 下、置中，勝方旁加黃色「勝」標籤，敗方整欄轉附註灰。中欄已完賽顯示「3 : 1」（數字 `--num-lg` 800，敗方數字附註灰，冒號 `--num-md` 靛藍）；未開打顯示「未開打」標籤 + 時間。卡底 1px `--line` 線之後列各局比分（領先方 800 `text`），黃金決勝局以標籤標示；未開打則顯示甜甜圈 + 「主隊 62% – 38% 客隊」。整卡是連結時 hover 升起。

### 表格（戰績、逐局比分、單場數據、進階數據）
白卡內，`border-collapse`，16px；表頭 14px 600 附註灰、無底色、底線 1px `--line`；主列 48px 高、列間 1px `--line`（最後一列不畫）；數字欄右對齊，勝場 800、其他數字 600；戰績前三名名次是 24px 靛藍 tint 圓點 + 靛藍數字；展開列底 `--surface-2`、底角 14px，用 `dl` 兩欄顯示。可橫捲時只包一層 `overflow-x: auto` 的 div。PR 值是 6px 高藥丸形進度條（軌道 tint、填色靛藍、最寬 7rem）+ 數字。

### 球員列 PlayerCard
56px 高一列：40px `--surface-2` 背號圓點（800）+ 姓名 700 + 位置附註灰；包成連結時 hover／active 靛藍 tint 底、14px 圓角。

### 資料備註 DataNote
`role="note"`，警示黃 tint 底、ink 字、16px、14px 圓角、內距 12px／16px，無左邊條。

### 圖表 Chart（ECharts）
`chartTheme(mode)`：序列色同 `series-1..5`；背景透明；長條 `barMaxWidth 18`、頂端圓角 6；折線 `smooth 0.3`、線寬 2.5、圓點 6；軸線與刻度不畫、數值軸 splitLine 用 `line` 實線、類別軸不畫格線；軸標籤附註灰 12px；legend 圓點 14px；tooltip 白卡、12px 圓角、Pop 陰影、無邊線、16px 字。

### 動效
只有三種：hover／focus 的 160ms ease-out 過渡（背景、文字色、位移、陰影）；首屏卡片的一次性 `.rise` 進場（`opacity 0 → 1`、`translateY(8px) → 0`，240ms `cubic-bezier(0.2, 0.7, 0.2, 1)`，`animation-delay: calc(var(--i) * 40ms)`，`--i` 最多 6）；甜甜圈的 600ms 填入。`prefers-reduced-motion: reduce` 下 `.rise` 停用、hover 不位移、所有 transition／animation 時長歸 0.01ms。

## Do's and Don'ts

### Do:
- **Do** 把區塊 h2 放在卡片外、上方，與卡片間距 12px；右側連結用 14px 700 靛藍。
- **Do** 用 `--surface-2` 面板或 1px `--line` 線做卡內分區；卡片不巢狀。
- **Do** 數字結論用 800 + tabular；敗方或次要數字改 `text-muted`。
- **Do** 改任何色值前跑 `npm run test:unit`，讓 `tokens.test.ts` 的 4.5:1／1.5:1／3:1 與 JS 同步檢查過關；兩套模式鍵名要一致。
- **Do** 新 icon 加進 `Icon.svelte`，遵守 24 viewBox、stroke 1.75、round cap／join、`currentColor`。
- **Do** 隊徽放 `static/logos/{group}-{team_id}.png`，交給 TeamBadge 退回；顏色區分處一定有簡稱文字。
- **Do** 可點的東西至少 44px 高（`--tap`），hover 升起只在 `(hover: hover)` 下做。

### Don't:
- **Don't** 把靛藍拿來當資料色或裝飾；資料色只從 `series-*` 拿，介面上只用前三條。
- **Don't** 在卡片上加隊色或任何彩色左邊條。
- **Don't** 在 `app.css`、`chart/theme.ts`、`color.ts SURFACE` 以外寫死色值；元件一律 `var(--token)`。
- **Don't** 載網路字型、加 icon 套件、用 emoji 或 unicode 符號當 icon。
- **Don't** 讓正文或表格內容小於 16px；`--text-sm` 只給附註與標籤。
- **Don't** 用 ECharts 畫甜甜圈，用 `WinRing`；數據磚超過三個並列就改回白卡清單。
- **Don't** 改隊色；對比不足時用 `teamPalette()` 的衍生值處理文字，不換色。
- **Don't** 為每個區塊加入場動畫；`.rise` 只給首屏卡片，最多六張。
