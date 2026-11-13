#!/usr/bin/env bash
# 在自己挑的空 port 起 preview 跑 Playwright，不會沿用 4173 上別的 worktree 的 preview。
# 不自己 build：web/build/ 要先由 npm run build 產生。
#
# 用法：bash scripts/e2e.sh [--repeat N | --profile] [spec…]
#   spec：以 web/ 為起點，例如 e2e/smoke.e2e.ts；不給就跑整套
#   --repeat N：連跑 N 輪，每輪印出 passed／failed／flaky／skipped，任何一輪失敗就以非零結束；
#               skip 原因第 1 輪印一次，之後只印與第 1 輪不同的
#   --profile：跑一輪，再從 Playwright 的 JSON 報表印實際 worker 數、測試數、逐檔耗時與最慢的測試
# 環境變數：TVL_PYTHON（預設 ~/venvs/tvl/bin/python）、PW_PORT（預設每輪挑一個空的 port）
set -uo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
python=${TVL_PYTHON:-$HOME/venvs/tvl/bin/python}

usage() {
	echo "用法：bash scripts/e2e.sh [--repeat N | --profile] [spec…]" >&2
	exit 2
}

repeat=0
profile=0
specs=()
while [ $# -gt 0 ]; do
	case $1 in
		--repeat)
			[[ ${2:-} =~ ^[1-9][0-9]*$ ]] || usage
			repeat=$2
			shift 2
			;;
		--profile)
			profile=1
			shift
			;;
		*)
			specs+=("$1")
			shift
			;;
	esac
done
if [ "$repeat" -gt 0 ] && [ "$profile" -eq 1 ]; then
	usage
fi

if [ ! -d "$repo/web/build" ]; then
	echo "找不到 web/build/：先在 web/ 跑 npm run build" >&2
	exit 1
fi

bash "$repo/scripts/busy-warning.sh"

free_port() {
	"$python" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

# 挑不到 port 就停：PW_PORT 空著會退回 4173 並沿用別人的 preview
next_port() {
	port=${PW_PORT:-$(free_port)}
	[ -n "$port" ] || { echo "挑不到空的 port（$python）" >&2; exit 1; }
}

cd "$repo/web" || exit 1

if [ "$profile" -eq 1 ]; then
	next_port
	report=$(mktemp -d "${TMPDIR:-/tmp}/tvl-e2e-profile.XXXXXX")/report.json
	echo "e2e（profile）：port $port，報表 $report"
	PW_PORT=$port PLAYWRIGHT_JSON_OUTPUT_FILE=$report \
		npx playwright test --reporter=list,./e2e/skip-reporter.ts,json "${specs[@]}"
	rc=$?
	echo
	"$python" "$repo/scripts/e2e_profile.py" "$report" || exit 1
	exit "$rc"
fi

if [ "$repeat" -eq 0 ]; then
	next_port
	echo "e2e：port $port"
	PW_PORT=$port exec npx playwright test "${specs[@]}"
fi

# 有內容才逐行輸出：空字串給 comm 會被當成一行空白
lines() {
	[ -z "$1" ] || printf '%s\n' "$1"
}

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
	# 並行時 skip 的順序每輪不同，排序後再比
	skips=$(grep '跳過：' "$log" | sort)
	if [ "$round" -eq 1 ]; then
		first_skips=$skips
		lines "$skips" | sed 's/^/  /'
	elif [ "$skips" != "$first_skips" ]; then
		echo "    略過原因與第 1 輪不同："
		comm -23 <(lines "$first_skips") <(lines "$skips") | sed 's/^ */      少了：/'
		comm -13 <(lines "$first_skips") <(lines "$skips") | sed 's/^ */      多了：/'
	fi
done

echo
if [ "$failed" -gt 0 ]; then
	echo "$failed/$repeat 輪失敗（log 與失敗現場在 $logs）"
	exit 1
fi
echo "$repeat 輪全過（log 在 $logs）"
