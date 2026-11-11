#!/usr/bin/env bash
# 在自己挑的空 port 起 preview 跑 Playwright，不會沿用 4173 上別的 worktree 的 preview。
# 不自己 build：web/build/ 要先由 npm run build 產生。
#
# 用法：bash scripts/e2e.sh [--repeat N] [spec…]
#   spec：以 web/ 為起點，例如 e2e/smoke.e2e.ts；不給就跑整套
#   --repeat N：連跑 N 輪，每輪印出 passed／failed／flaky／skipped 與 skip 原因，任何一輪失敗就以非零結束
# 環境變數：TVL_PYTHON（預設 ~/venvs/tvl/bin/python）、PW_PORT（預設每輪挑一個空的 port）
set -uo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
python=${TVL_PYTHON:-$HOME/venvs/tvl/bin/python}

usage() {
	echo "用法：bash scripts/e2e.sh [--repeat N] [spec…]" >&2
	exit 2
}

repeat=0
specs=()
while [ $# -gt 0 ]; do
	case $1 in
		--repeat)
			[[ ${2:-} =~ ^[1-9][0-9]*$ ]] || usage
			repeat=$2
			shift 2
			;;
		*)
			specs+=("$1")
			shift
			;;
	esac
done

if [ ! -d "$repo/web/build" ]; then
	echo "找不到 web/build/：先在 web/ 跑 npm run build" >&2
	exit 1
fi

free_port() {
	"$python" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

# 挑不到 port 就停：PW_PORT 空著會退回 4173 並沿用別人的 preview
next_port() {
	port=${PW_PORT:-$(free_port)}
	[ -n "$port" ] || { echo "挑不到空的 port（$python）" >&2; exit 1; }
}

cd "$repo/web" || exit 1

if [ "$repeat" -eq 0 ]; then
	next_port
	echo "e2e：port $port"
	PW_PORT=$port exec npx playwright test "${specs[@]}"
fi

logs=$(mktemp -d "${TMPDIR:-/tmp}/tvl-e2e.XXXXXX")
failed=0
for round in $(seq "$repeat"); do
	next_port
	start=$SECONDS
	log=$logs/$round.log
	mark=✓ note=
	if ! PW_PORT=$port npx playwright test --output "$logs/$round" "${specs[@]}" > "$log" 2>&1; then
		mark=✗ note="  （$log）"
		failed=$((failed + 1))
	fi
	printf '%s 第 %d/%d 輪  %4ds  port %s%s\n' "$mark" "$round" "$repeat" $((SECONDS - start)) "$port" "$note"
	# 結尾的計數區塊（含失敗、flaky 的測試名稱）；沒有計數就是沒跑起來，改印 log 末段
	summary=$(awk '/^ +[0-9]+ (passed|failed|flaky|skipped|interrupted|did not run)/ { p = 1 } p' "$log")
	[ -n "$summary" ] || summary=$(tail -n 15 "$log")
	sed 's/^/  /' <<< "$summary"
	grep '跳過：' "$log" | sed 's/^/  /'
done

echo
if [ "$failed" -gt 0 ]; then
	echo "$failed/$repeat 輪失敗（log 與失敗現場在 $logs）"
	exit 1
fi
echo "$repeat 輪全過（log 在 $logs）"
