#!/usr/bin/env bash
# domain-expiry.yml 剩餘天數少於門檻時呼叫：同標題（整行完全相同）的 issue 還開著就不重開，沒有才開一張提醒續約（#233、#259）。
# gh issue list 失敗就以非零結束，不開 issue。
#
# 用法：bash scripts/domain-expiry-issue.sh <到期日> <剩餘天數>
# 需要的環境變數：GH_TOKEN，以及 GitHub Actions 內建的 GITHUB_SERVER_URL、GITHUB_REPOSITORY、GITHUB_RUN_ID
set -euo pipefail

expires=$1
days_left=$2
title="vstat.tw 網域快到期了，請到 PChome 續約"

open_titles=$(gh issue list --state open --limit 1000 --json title --jq '.[].title')
if grep -Fxq "$title" <<<"$open_titles"; then
	echo "同標題的 issue 還開著，不重開"
	exit 0
fi

gh issue create --title "$title" --label ready-for-human --body-file - <<EOF
vstat.tw 的到期日是 **$expires**，剩 **$days_left 天**。

PChome 後台沒有自動續約。網域過期後整站打不開，已訂閱的推播也會一起失效。

- [ ] 到 PChome 買網址的後台替 vstat.tw 續約
- [ ] 續約後 TWNIC 的公開查詢大約一天才會更新；隔天手動跑一次 domain-expiry workflow，確認到期日已經往後延
- [ ] 確認後關掉這張 issue（這張還開著的期間，每週檢查不會再開新的）

這張 issue 由每週檢查（.github/workflows/domain-expiry.yml）自動開出：$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID
EOF
