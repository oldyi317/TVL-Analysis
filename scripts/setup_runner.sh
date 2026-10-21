#!/usr/bin/env bash
# self-hosted runner 安裝精靈（WSL 內執行）：ephemeral 模式，每跑完一個 job 自動註銷並重新註冊
# 前置需求：gh 已登入、sudo 權限、WSL 已啟用 systemd（ps -p 1 顯示 systemd）、
#           oldyi317 的 fine-grained PAT（只限本 repo、Administration 讀寫）存於 ~/.config/tvl-runner/token（chmod 600）
set -euo pipefail

REPO="oldyi317/TVL-Analysis"
RUNNER_DIR="$HOME/actions-runner"
SERVICE="tvl-runner"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOKEN_FILE="$HOME/.config/tvl-runner/token"

step() { echo; echo "── $1 ──"; read -rp "按 Enter 繼續（Ctrl+C 中止）..."; }

echo "TVL self-hosted runner 安裝精靈（ephemeral）"

step "步驟 1/5：檢查 gh 登入與 systemd"
gh auth status
# 取註冊 token 需要 repo admin（個人 repo 只有 owner 有），用 owner 的 PAT；迴圈每輪都要取，這裡先擋
[ "$(stat -c '%a' "$TOKEN_FILE" 2>/dev/null)" = "600" ] || { echo "找不到 $TOKEN_FILE 或權限不是 600"; exit 1; }
[ "$(GH_TOKEN=$(<"$TOKEN_FILE") gh api "repos/${REPO}" --jq '.permissions.admin')" = "true" ] || { echo "PAT 對 ${REPO} 沒有 admin 權限，無法取 runner 註冊 token"; exit 1; }
[ "$(ps -p 1 -o comm=)" = "systemd" ] || { echo "WSL 未啟用 systemd，請先在 /etc/wsl.conf 開啟後 wsl --shutdown 重進"; exit 1; }

# 檢查 runner 工作環境的 venv 是否存在
[ -x "$HOME/venvs/tvl/bin/python" ] || { echo "找不到 $HOME/venvs/tvl/bin/python，請先建立 venv 並 pip install -r requirements.txt"; exit 1; }

step "步驟 2/5：下載 actions-runner 到 $RUNNER_DIR（已存在則略過）"
mkdir -p "$RUNNER_DIR" && cd "$RUNNER_DIR"
if [ ! -x ./config.sh ]; then
  VER=$(gh api repos/actions/runner/releases/latest --jq '.tag_name' | tr -d v)
  curl -fL -o runner.tar.gz \
    "https://github.com/actions/runner/releases/download/v${VER}/actions-runner-linux-x64-${VER}.tar.gz"
  tar xzf runner.tar.gz && rm runner.tar.gz
fi

step "步驟 3/5：停掉現有 runner 並移除註冊（svc.sh 舊服務或 ${SERVICE}；沒有則略過，需要 sudo）"
sudo systemctl stop "$SERVICE" 2>/dev/null || true
if [ -f .service ]; then
  sudo ./svc.sh stop || true
  sudo ./svc.sh uninstall
fi
if [ -f .runner ]; then
  TOKEN=$(GH_TOKEN=$(<"$TOKEN_FILE") gh api -X POST "repos/${REPO}/actions/runners/remove-token" --jq '.token')
  # ephemeral 註冊在伺服器端可能已自動註銷，remove 失敗就只清本機檔
  ./config.sh remove --token "$TOKEN" || rm -f .runner .credentials .credentials_rsaparams
fi

step "步驟 4/5：安裝迴圈腳本並以 systemd service（${SERVICE}）開機自啟（需要 sudo）"
cp "$SCRIPT_DIR/runner_loop.sh" "$RUNNER_DIR/runner_loop.sh"
chmod +x "$RUNNER_DIR/runner_loop.sh"
sudo tee "/etc/systemd/system/${SERVICE}.service" > /dev/null <<EOF
[Unit]
Description=TVL ephemeral GitHub Actions runner loop
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$RUNNER_DIR
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/bin/bash $RUNNER_DIR/runner_loop.sh
Restart=always
RestartSec=60
TimeoutStopSec=5min

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE"
sudo systemctl restart "$SERVICE"
sudo systemctl status "$SERVICE" --no-pager || true

step "步驟 5/5：Windows 端設定（手動）"
cat <<'EOF'
在 Windows 設定「登入時自動啟動 WSL」，讓 runner 開機即上線：
1. 開始功能表搜尋「工作排程器」→ 建立基本工作
2. 名稱：Start WSL for TVL runner；觸發程序：當我登入時
3. 動作：啟動程式；程式：wsl.exe；引數：-d <發行版名稱> --exec /bin/true
   （發行版名稱在 PowerShell 跑 `wsl -l -q` 查）
4. 完成。tvl-runner service 隨 WSL 啟動，常駐進程使 WSL 不被閒置回收。

【公開 repo 安全加強】
到 GitHub repo Settings → Actions → General 確認
「Fork pull request workflows」的 approval 設定為「Require approval for all outside
collaborators」（公開 repo 上惡意 fork PR 可能指向 self-hosted runner 執行任意程式碼）：
https://github.com/oldyi317/TVL-Analysis/settings/actions
EOF
echo "安裝完成。到 https://github.com/${REPO}/settings/actions/runners 確認 runner 顯示 Idle；"
echo "日誌：journalctl -u ${SERVICE} -f"
