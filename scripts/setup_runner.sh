#!/usr/bin/env bash
# self-hosted runner 一次性安裝精靈（WSL 內執行）
# 前置需求：gh 已登入、sudo 權限、WSL 已啟用 systemd（ps -p 1 顯示 systemd）
set -euo pipefail

REPO="oldyi317/TVL-Analysis"
RUNNER_DIR="$HOME/actions-runner"

step() { echo; echo "── $1 ──"; read -rp "按 Enter 繼續（Ctrl+C 中止）..."; }

echo "TVL self-hosted runner 安裝精靈"

step "步驟 1/5：檢查 gh 登入與 systemd"
gh auth status
[ "$(ps -p 1 -o comm=)" = "systemd" ] || { echo "WSL 未啟用 systemd，請先在 /etc/wsl.conf 開啟後 wsl --shutdown 重進"; exit 1; }

step "步驟 2/5：下載 actions-runner 到 $RUNNER_DIR"
mkdir -p "$RUNNER_DIR" && cd "$RUNNER_DIR"
VER=$(gh api repos/actions/runner/releases/latest --jq '.tag_name' | tr -d v)
curl -fL -o runner.tar.gz \
  "https://github.com/actions/runner/releases/download/v${VER}/actions-runner-linux-x64-${VER}.tar.gz"
tar xzf runner.tar.gz && rm runner.tar.gz

step "步驟 3/5：向 repo 註冊 runner（label: tvl）"
TOKEN=$(gh api -X POST "repos/${REPO}/actions/runners/registration-token" --jq '.token')
./config.sh --url "https://github.com/${REPO}" --token "$TOKEN" \
  --name tvl-wsl --labels tvl --unattended

step "步驟 4/5：安裝並啟動 systemd service（需要 sudo）"
sudo ./svc.sh install "$USER"
sudo ./svc.sh start
sudo ./svc.sh status || true

step "步驟 5/5：Windows 端設定（手動）"
cat <<'EOF'
在 Windows 設定「登入時自動啟動 WSL」，讓 runner 開機即上線：
1. 開始功能表搜尋「工作排程器」→ 建立基本工作
2. 名稱：Start WSL for TVL runner；觸發程序：當我登入時
3. 動作：啟動程式；程式：wsl.exe；引數：-d <發行版名稱> --exec /bin/true
   （發行版名稱在 PowerShell 跑 `wsl -l -q` 查）
4. 完成。runner 的 systemd service 隨 WSL 啟動，常駐進程使 WSL 不被閒置回收。
EOF
echo "安裝完成。到 https://github.com/${REPO}/settings/actions/runners 確認 runner 顯示 Idle。"
