---
status: accepted
date: 2026-10-12
---

# 以每日重建的靜態站取代 Streamlit 儀表板

儀表板的主要受眾改為一般球迷，以手機閱讀為主。Streamlit 的外觀像後台工具、Streamlit Cloud 有冷啟動與休眠、手機版型難以控制，且每次互動整頁重跑。資料本來就由每日 GitHub Actions 爬蟲更新並 commit 進 repo，因此決定改為 **SvelteKit 靜態輸出**：Python 匯出器在建置時從 SQLite 產出 JSON，前端只讀 JSON，由 push 觸發的雲端 workflow 建置並部署到 GitHub Pages。圖表用 ECharts。

放棄的東西：執行期互動（what-if 滑桿改為賽前勝率）、執行期即時抓官網（所有資料來自建置時的 DB）。換到的東西：零冷啟動、免費主機、完全自訂的手機優先介面。

## Considered Options

- 留在 Streamlit 深度客製 CSS：解不了冷啟動與整頁重跑。
- Dash / Panel / Reflex（Python 限定）：外觀與手機版型的控制力仍不足。
- Astro：靜態輸出與 island 模式很合適，但使用者兩個前端框架都不熟，SvelteKit 的學習曲線與打包體積更有利。
- Next.js 靜態輸出：生態最大，但體積較大，且 App Router 多數功能在純靜態模式用不到。
- Observable Framework：資料流最順（Python data loader），但外觀是資料儀表板風格，難做成運動網站。

## Consequences

- 舊 Streamlit 站在新站追平前凍結，只修會壞的 bug。追平點是第二期（球隊頁、球員頁、賽事詳情）完成。
- 前端放在同一 repo 的 `web/`；匯出的 JSON 不進 git，由 CI 產生。
- 組別與賽季進 URL 路徑（例如 `/2025-26/men/standings`），根路徑導向當季男子組。

## 後續註記（2026-10-14）

- 追平點於 2026-10-14 達成：#13 賽事詳情單場數據、#14 球隊頁、#16 球員進階數據已合併。
- #17 下線動作：移除 `src/app`、`packages.txt` 與 streamlit/plotly/matplotlib/shap 依賴；README 與 CLAUDE.md 改指向 GitHub Pages 站。
- Streamlit Cloud 的 app 待使用者自行刪除，刪除後在 #17 留言確認舊網址已失效。
