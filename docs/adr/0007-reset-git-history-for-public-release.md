---
status: accepted
date: 2026-10-21
---

# 正式上線前重開 git 歷史，舊歷史只存私有備份

repo 自 2026-03-19 起即為 public。上線前盤點發現歷史裡有 21 版含球員生日的資料庫（ADR-0006）、51 筆帶公司信箱的 commit、一份含公司內部主機名的設計文件。這三者散在幾乎每一個 commit，用 `git filter-repo` 逐項清除要對二進位資料庫逐版改寫，風險高且難驗證。

決定：清理完工作樹後，以單一 commit 作為新的 main 起點並 force-push，舊歷史用 `git bundle` 打包，只存進私有 repo，不在公開 repo 留 tag 或分支。repo 當時 0 個 fork，沒有外部副本需要同步，是重開歷史成本最低的時點。

## Consequences

- 失去 blame 與過去的 commit 訊息；重要決策已由 docs/adr 與 CONTEXT.md 承載。
- GitHub Issues 不受影響，但 issue 內引用的舊 commit hash 會失效，接受。
- self-hosted runner 端的 clone 必須重新 clone，Pages 需確認重建成功。
- 一併移出公開 repo 的：`docs/superpowers/`、`notebooks/`、pcai 相關文件與分支，全部進同一個私有 repo。
