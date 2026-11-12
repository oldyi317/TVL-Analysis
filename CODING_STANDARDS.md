# 寫碼規範

審查時逐條對這份檔。能交給工具檢查的規則不寫在這裡（行尾由 `.gitattributes` 管，避免詞由 `tests/test_vocabulary.py` 管，e2e 的匯入規則由 `web/e2e/imports.test.ts` 管）。

## Python

- import 一律走 `from src.utils...` 絕對路徑，從 repo 根目錄執行。
- 共用常數（隊伍對照表、headers、DB 路徑）只放 `src/utils/constants.py` 與
  `src/utils/db_config.py`，個別檔案引用它們，不各自複製一份。
- DB 連線用 `src/utils/db_config.get_connection()`（有開 `PRAGMA foreign_keys`），
  取代裸用 `sqlite3.connect()`。
- Schema DDL 的唯一來源是 `sql/schema.sql`，爬蟲引用它，不自己寫 CREATE TABLE。

## 依賴

- Python 依賴只有一份 `requirements.txt`（爬蟲、模型重訓、pytest 全在裡面，裝在自架
  runner 與本機）。
- pages.yml 的匯出器只用標準庫、不裝 pip 套件；`src/export/` 與它匯入的模組維持只用標準庫。
- 前端依賴在 `web/package.json`，改動要考慮 pages.yml 的 `npm ci`。

## 文案與語言

- UI 文案、註解、commit message 用繁體中文。

## 風格

- 程式保持簡單：解決眼前這張票的需求，票沒要求的參數、抽象、設定留到真的需要時再加。
- 註解與 docstring 只寫程式本身看不出來的事（原因、限制、踩過的雷）。
- 函式小而專注；early return 優於巢狀條件。
- 縮排沿用檔案既有風格（`web/` 一律 tab）。

## 測試

- 測試不寫檔到 repo 外，也不留除錯用截圖；截圖腳本不進 commit。
