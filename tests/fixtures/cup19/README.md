# cup 19 探路代表頁

取自 2026-11-05 的小量 HTTP 請求，來源與判讀見
[回填探路報告](../../../docs/research/cup19-backfill-feasibility.md)。

- `*_response.html` 是後續離線測試用 fixture；不是網站建置輸入。
- 同名 `*_response.json` 保存完整來源 URL、請求時間與沿用的 `EXT_HEADERS`、HTTP 狀態、挑戰檢測結果。
- `raw_response_bytes`／`raw_response_sha256` 對應取得時的 HTTP body（requests 解壓縮後）；`fixture_bytes`／`fixture_sha256` 對應此目錄已提交的 HTML。兩者可能不同。
- 全部 HTML 以 UTF-8、LF 儲存。外部 handler 保留全內容；官網賽程擷取包含全部賽事的 `.matches` 父 container；官網單場保留含 `.game_header` 或表格的 section；新聞僅保留與報告有關的兩段。逐檔轉換說明見 `fixture_transform`。未保留導覽、登入表單及追蹤腳本等無關頁面區塊。
- `default_response.html`、`match_page_response.html` 是 HTTP 404 負向樣本，不能當空賽季或成功單場頁。兩份 404 原回應附帶 Cloudflare `challenge-platform` 注入腳本，fixture 已移除並在 metadata 記錄；這與要求完成驗證的阻擋頁不同。本次未取得 HTTP 403 或阻擋式驗證負向 fixture。
- `local_identity_snapshot.json` 是本次 DB 在 `PRAGMA query_only=ON` 下的查詢摘錄，含 DB SHA-256、抽樣姓名的同名候選與既有隊伍。它不是完整名單，也不代表已核定球員身分。

後續回填票應在隔離 DB 測試「來源頁面 → 賽事／登錄／單場數據與待確認清單」。
勿以外部 MatchID 等同官網 game_id；樣本中特別保存了 MatchID 2 → 官方編號 3、MatchID 101 → 官方編號 89 的情形。
本票沒有改爬蟲、共用常數、schema 或正式 DB。
