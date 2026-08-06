# 已知問題追蹤

## #1 系統設定頁「測試連線」未反映表單當下的 TLS 勾選狀態（開放）

- **記錄日期**：2026-08-06
- **症狀**：使用者在系統設定頁取消勾選「驗證 TLS 憑證」後按「測試連線」，仍然失敗（Connection error）。
- **證據**：
  - Server log（13:35–13:37 四次嘗試）底層原因均為 `SSL: CERTIFICATE_VERIFY_FAILED`——測試當下憑證驗證仍是開啟狀態。
  - 同時間 DB 的 `app_settings.mlis_verify_ssl` 值仍為 `'true'`——取消勾選的狀態未曾寫入 DB。
  - 後端路徑已排除：以相同 endpoint/key 直接建 `LLMConfig(verify_ssl=False)` 呼叫 `test_connection` 回傳成功（HTTP 200）；`curl -k` 亦成功。endpoint、token、`llm_client` 的 verify 邏輯均正常。
- **高信度假設**：「測試連線」按鈕使用 `resolve_llm_config`（讀 DB 已儲存值）而非表單當下的欄位值；使用者取消勾選但尚未按「儲存設定」，或儲存流程與測試流程的先後順序造成勾選狀態未持久化。屬 UX 流程缺陷，非連線層缺陷。
- **修正方向**（下次處理）：
  1. 「測試連線」改用表單當下的輸入值（含勾選框狀態）組 config，而非 DB 值；或
  2. 按「測試連線」時先自動儲存表單；並在 UI 明示測試所用的是哪一份設定。
  3. 補 AppTest：取消勾選（不儲存）→ 按測試連線 → 斷言傳給 test_connection 的 config.verify_ssl 為 False。
- **環境備註**：根本解仍是叢集內以 `MLIS_CA_BUNDLE` 信任內部 CA（README 已記載）；本議題只影響「關閉驗證」路徑的 UX。

## #0（已解決，供脈絡）MLIS 內部 CA 憑證導致連線失敗

- 2026-08-06 以 `cb80838`/`d859794` 修正：`verify_ssl` 開關（DB/env）、`MLIS_CA_BUNDLE` 支援、底層例外記錄。後端 E2E 已實證可連。
