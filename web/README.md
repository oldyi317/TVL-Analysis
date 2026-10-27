# web/

TVL 數據靜態站（SvelteKit + adapter-static，部署於 GitHub Pages `/TVL-Analysis/`）。
本機開發三步（匯出、dev server、建置）見 repo 根目錄 `README.md` 的「本機跑網站」。

- `web/schema/`：JSON schema，匯出器與前端共用的唯一來源
- `web/src/lib/data/`：匯出的 JSON 與產生的型別（不進 git）
- `web/e2e/`：Playwright 手機 smoke
