#!/usr/bin/env bash
# job 開始前的守門（runner_loop.sh 以 ACTIONS_RUNNER_HOOK_JOB_STARTED 掛上，任何步驟執行前先跑）：
# 公開 repo 的 fork PR 能自訂 workflow，把 job 指到這台 runner；只放行本 repo 的 schedule 與 workflow_dispatch。
# hook 失敗只會把 job 標成失敗，if: always() 的步驟照樣會跑，所以拒絕時直接砍掉 Runner.Worker。
# 由 setup_runner.sh 安裝到 ~/.config/tvl-runner/
REPO="oldyi317/TVL-Analysis"

allowed() {
    [ "${GITHUB_REPOSITORY:-}" = "$REPO" ] || return 1
    case "${GITHUB_EVENT_NAME:-}" in
        schedule|workflow_dispatch) return 0 ;;
    esac
    return 1
}

kill_worker() {
    if [ "$(cat "/proc/$PPID/comm" 2>/dev/null)" = "Runner.Worker" ]; then
        kill -KILL "$PPID" && return
    fi
    pkill -KILL -u "$EUID" -x Runner.Worker
}

allowed && exit 0
echo "拒絕執行：${GITHUB_REPOSITORY:-?} 的 ${GITHUB_EVENT_NAME:-?} 事件不在允許清單（只放行 schedule、workflow_dispatch）" >&2
kill_worker
exit 1
