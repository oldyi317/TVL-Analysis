---
status: accepted
date: 2026-11-16
---

# 發送服務的狀態存 D1，不重發靠條件式寫入

發送服務（`push/`，Cloudflare Workers 免費方案）要存推播訂閱、哪些賽事發過、今天的今日賽事發過沒有、每一則送到第幾批。原則是寧可漏發不重發：同一場賽事、同一天的今日賽事、同一批推播訂閱都不能收到兩次。定時器每分鐘跳一次，上一次還沒結束時下一次會並行執行（#227 實測），所以「先讀狀態、再決定送不送」不夠，必須有一個只會有一個呼叫成功的動作。選項：(a) Workers KV；(b) Queues 或 Durable Objects；(c) D1，所有「要送」之前先用條件式寫入或唯一鍵認領。決定 **(c)**。

- 狀態全部存在同一個 D1 資料庫，不開讀取複本。
- 每一批送之前用一條帶條件的寫入把進度從 N 推到 N+15，條件是「進度現在還是 N」；`meta.changes` 是 1 才送，是 0 代表別的呼叫已經認領，這一分鐘跳過。
- 「這場發過了」以 `cup_id` 加 `group` 加 `game_id` 為唯一鍵，「今天的今日賽事發過了」以台北日期為唯一鍵；寫入時鍵已存在就不發。
- 不依賴「讀到最新」：讀到舊資料的後果是這一分鐘白跑，不會重發。

理由：(a) KV 在別處約一分鐘才看得到寫入，正好落在定時器間隔上；免費方案每天 1,000 次寫入，會被推播訂閱的每日重新登記用完；也沒有條件式寫入，做不出認領。(b) Queues 與 Durable Objects 都是至少一次遞送，仍得先認領，多一套東西卻沒省掉認領；推播訂閱超過 500 筆再評估。(c) D1 的條件式 `UPDATE` 與唯一鍵在並行下只會有一個呼叫改到。

依據（#227 結論留言，2026-11-12；原型在 `prototype/app-d1-claim` 分支的 `prototypes/d1-claim/`）：

- 文件沒有逐字保證「寫入後下一次呼叫一定讀得到」，也沒寫單一 `UPDATE` 是否原子。間接依據：「D1 routes all queries (both read and write) to a specific database instance」、開啟讀取複本前「all queries will continue to be executed only by the primary database」（<https://developers.cloudflare.com/d1/best-practices/read-replication/>），以及「inherently single-threaded, and processes queries one at a time」（<https://developers.cloudflare.com/d1/reference/faq/>）。
- 實測（免費方案，D1 在 APAC，請求從 LAX 機房進，wrangler 4.159.0）：連續 300 次先讀再寫，讀到舊值 0 次；條件式 `UPDATE … WHERE` 改到時 `meta.changes` 是 1、沒改到是 0 且不丟錯誤；`INSERT OR IGNORE` 寫重複主鍵 `meta.changes` 是 0；50 個並行請求搶同一批 20 輪，每輪剛好 1 個 `changes: 1`；50 個並行請求寫同一個主鍵 20 輪，每輪剛好 1 個 `changes: 1`，0 個錯誤。每分鐘一次的定時呼叫連續 11 次都讀到上一次寫入的值，條件式寫入每次都是 `changes: 1`；故意拖 75 秒的呼叫沒被中止，下一次照常並行開始。
- 限制：所有請求都從同一個機房進來，沒有測到兩次呼叫落在不同機房。認領是寫入，寫入一律到主庫，設計不依賴這一點。

## Consequences

- D1 不開讀取複本；開了之後讀到舊值的機率變高，雖然認領仍擋得住重發，白跑的分鐘會變多。
- 推播訂閱上限 500 筆，滿了就拒收新的推播位址。到了 500 筆就是換機制（Queues 或 Durable Objects）的時候。
- 認領成功後呼叫中斷（逾時、例外、部署），那一批會漏發，不重試。這是「寧可漏發不重發」接受的代價。
- 發送紀錄與「發過」的記號都不刪，不寫清理程式；一年約 200–300 筆發送紀錄加一百多筆賽事記號，遠低於免費方案的容量。
