#!/usr/bin/env bash
# 收工四項依序跑：pytest、npm run check、npm run test:unit、e2e（先 build，再跑 scripts/e2e.sh）。
# 每一項都會跑完並各自回報，任何一項失敗就以非零結束；輸出寫到 log，失敗才印末段。
#
# 用法：bash scripts/verify.sh
# 環境變數：TVL_PYTHON（預設 ~/venvs/tvl/bin/python）、PW_PORT（預設挑一個空的 port）
set -uo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
python=${TVL_PYTHON:-$HOME/venvs/tvl/bin/python}
logs=$(mktemp -d "${TMPDIR:-/tmp}/tvl-verify.XXXXXX")
failed=()

case $repo in
	/mnt/*) echo "注意：這個 checkout 在 Windows 檔案系統上，build 要十幾分鐘。用 scripts/new-worktree.sh 開在 ~/work 會快很多。" ;;
esac

# step <名稱> <目錄> <命令…>：跑一項、記耗時與結果
step() {
	local name=$1 dir=$2 start=$SECONDS
	shift 2
	if (cd "$dir" && "$@") > "$logs/$name.log" 2>&1; then
		printf '✓ %-6s %4ds\n' "$name" $((SECONDS - start))
		return 0
	fi
	printf '✗ %-6s %4ds  （%s）\n' "$name" $((SECONDS - start)) "$logs/$name.log"
	tail -n 25 "$logs/$name.log" | sed 's/^/    /'
	failed+=("$name")
	return 1
}

step export "$repo" "$python" -m src.export.site_data
step pytest "$repo" env OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" "$python" -m pytest -q
step check "$repo/web" npm run check
step unit "$repo/web" npm run test:unit
if step build "$repo/web" npm run build; then
	step e2e "$repo" bash scripts/e2e.sh
	grep -E '^\s+[0-9]+ (passed|failed|flaky|skipped)' "$logs/e2e.log" | sed 's/^/    /'
else
	echo "✗ e2e      略過（build 失敗）"
	failed+=(e2e)
fi

echo
if [ ${#failed[@]} -gt 0 ]; then
	echo "失敗：${failed[*]}（共 ${SECONDS}s，log 在 $logs）"
	exit 1
fi
echo "收工四項全過：pytest、check、unit、e2e（export、build 為前置步驟；共 ${SECONDS}s，log 在 $logs）"
