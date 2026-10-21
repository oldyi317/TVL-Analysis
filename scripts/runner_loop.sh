#!/usr/bin/env bash
# ephemeral self-hosted runner 迴圈：取註冊 token → 一次性註冊 → 跑一個 job → 回到開頭
# 由 setup_runner.sh 複製到 runner 目錄，並以 systemd service（tvl-runner）開機自啟
set -uo pipefail

REPO="oldyi317/TVL-Analysis"
RUNNER_DIR="${RUNNER_DIR:-$HOME/actions-runner}"
# oldyi317 的 fine-grained PAT（只限本 repo、Administration 讀寫）；個人 repo 的 collaborator 拿不到 admin，取不了註冊 token
TOKEN_FILE="${RUNNER_TOKEN_FILE:-$HOME/.config/tvl-runner/token}"
BACKOFF_START=30
BACKOFF_MAX=900
# run.sh 對 listener 異常（驗證失敗、session 衝突等）也回 0，跑不到這麼久就當作失敗
MIN_JOB_SECONDS="${RUNNER_MIN_JOB_SECONDS:-60}"

cd "$RUNNER_DIR" || exit 1

run_once() {
    # 上一輪異常中斷可能留下本機設定（config.sh 會拒絕重複註冊）與上個 job 的工作目錄
    rm -f .runner .credentials .credentials_rsaparams
    rm -rf _work
    local token
    [ -r "$TOKEN_FILE" ] || { echo "找不到 PAT 檔 $TOKEN_FILE"; return 1; }
    # 每輪重讀，PAT 換新不用重啟；只給 gh，不進 job 環境
    token=$(GH_TOKEN=$(<"$TOKEN_FILE") gh api -X POST "repos/${REPO}/actions/runners/registration-token" --jq '.token') \
        || { echo "取註冊 token 失敗"; return 1; }
    ./config.sh --unattended --ephemeral --replace --url "https://github.com/${REPO}" \
        --token "$token" --name tvl-wsl --labels tvl || { echo "註冊失敗"; return 1; }
    local started=$SECONDS
    ./run.sh || { echo "run.sh 異常結束"; return 1; }
    (( SECONDS - started >= MIN_JOB_SECONDS )) || { echo "run.sh 在 ${MIN_JOB_SECONDS} 秒內結束，視為異常"; return 1; }
}

backoff=$BACKOFF_START
while true; do
    if run_once; then
        backoff=$BACKOFF_START
        sleep 5
        continue
    fi
    echo "本輪失敗，${backoff} 秒後重試"
    sleep "$backoff"
    backoff=$(( backoff * 2 > BACKOFF_MAX ? BACKOFF_MAX : backoff * 2 ))
done
