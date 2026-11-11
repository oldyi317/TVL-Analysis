#!/usr/bin/env bash
# 每日更新的本機觸發（#196）：GitHub 的 schedule 實際比 cron 晚 4–7 小時，改由 runner 所在的 WSL
# 以 user 層級 systemd timer 每天台北 09:23 用 gh 觸發 daily-crawl（workflow_dispatch）；
# 錯過（未開機）就在開機後補跑一次。daily-crawl.yml 的 schedule 留作備援。
# 前置需求：gh 已登入且對 repo 有 push 權限、loginctl enable-linger（WSL 開機即啟動 user systemd）。
# 不需要 sudo；重跑會覆蓋同名 unit。
set -euo pipefail

REPO="oldyi317/TVL-Analysis"
UNIT="tvl-crawl-trigger"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
GH=$(command -v gh) || { echo "找不到 gh，請先安裝並登入" >&2; exit 1; }
[ "$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null)" = "yes" ] ||
  echo "警告：未開 linger，WSL 開機後沒有登入 session 時 timer 不會跑；執行 loginctl enable-linger $(id -un)" >&2

mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/$UNIT.service" <<EOF
[Unit]
Description=觸發 TVL daily-crawl workflow
StartLimitIntervalSec=30min
StartLimitBurst=5

[Service]
Type=oneshot
ExecStart=$GH workflow run daily-crawl.yml --repo $REPO --ref main
Restart=on-failure
RestartSec=2min
EOF

cat > "$UNIT_DIR/$UNIT.timer" <<EOF
[Unit]
Description=每天台北 09:23 觸發 TVL daily-crawl

[Timer]
OnCalendar=*-*-* 09:23:00 Asia/Taipei
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "$UNIT.timer"
echo "已啟用 $UNIT.timer；下次觸發：systemctl --user list-timers $UNIT.timer，日誌：journalctl --user -u $UNIT"
