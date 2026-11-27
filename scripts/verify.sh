#!/usr/bin/env bash
# 收工各項：pytest、npm run check、npm run test:unit、e2e（先 build，再跑 scripts/e2e.sh）。
# 另外驗 push/、score/ 的 npm run check 與 npm test（CI 的 push、score job 同款），並用 actionlint 檢查 .github/workflows、用 shellcheck 檢查 scripts/、用 ruff 檢查 Python。
# actionlint、shellcheck 找不到就算失敗（actionlint 版本以 .github/workflows/ci.yml 的 VERSION 為準）；push/、score/ 缺 node_modules 也算失敗，先在該目錄跑 npm ci，腳本不自己安裝。
# pytest 在背景與網站那一串同時跑；網站那一串依序跑（check、unit、build 啟動時都會重寫 gen 與 svelte-kit sync 的產生檔，不能並行）。
# push/、score/ 接在網站那一串之後依序跑（各自獨立目錄，不碰 web/ 的產生檔，也不與它並行）。
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
bash "$repo/scripts/busy-warning.sh"

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

if command -v actionlint > /dev/null; then
	step lint "$repo" actionlint
else
	echo "✗ lint     找不到 actionlint。裝到 PATH，版本照 .github/workflows/ci.yml 的 VERSION：https://github.com/rhysd/actionlint/blob/main/docs/install.md"
	failed+=(lint)
fi
# SC1087 出在 /wizard skill 產生的固定函式庫（score_token_wizard.sh），不手改它
if command -v shellcheck > /dev/null; then
	step shell "$repo" shellcheck -S warning -e SC1087 "$repo"/scripts/*.sh
else
	echo "✗ shell    找不到 shellcheck（sudo apt install shellcheck）。少了它，actionlint 也會靜默略過 workflow 的 run 區塊"
	failed+=(shell)
fi
step ruff "$repo" "$python" -m ruff check
# 缺 node_modules 就失敗並提示 npm ci，不自己安裝
need_deps() {
	[ -d "$repo/$1/node_modules" ] && return 0
	echo "✗ $1  缺 node_modules，先跑：(cd $repo/$1 && npm ci --prefer-offline --no-audit --no-fund)"
	failed+=("$1")
	return 1
}

step export "$repo" "$python" -m src.export.site_data
# 背景的 step 改不到這裡的 failed：結果行先寫進檔案，網站那一串跑完再印
step pytest "$repo" env OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" "$python" -m pytest -q > "$logs/pytest.result" &
pytest_pid=$!
step check "$repo/web" npm run check
step unit "$repo/web" npm run test:unit
if step build "$repo/web" npm run build; then
	step e2e "$repo" bash scripts/e2e.sh
	grep -E '^\s+[0-9]+ (passed|failed|flaky|skipped)' "$logs/e2e.log" | sed 's/^/    /'
else
	echo "✗ e2e      略過（build 失敗）"
	failed+=(e2e)
fi
# 發送服務與比分服務（score 匯入 push/src/time.ts 與 web/schema，改到那些檔也靠這裡擋）
for svc in push score; do
	if need_deps "$svc"; then
		step "$svc-chk" "$repo/$svc" npm run check && step "$svc-test" "$repo/$svc" npm test
	fi
done
wait "$pytest_pid" || failed+=(pytest)
cat "$logs/pytest.result"

echo
if [ ${#failed[@]} -gt 0 ]; then
	echo "失敗：${failed[*]}（共 ${SECONDS}s，log 在 $logs）"
	exit 1
fi
echo "收工全過：ruff、pytest、check、unit、e2e、lint、shell、push、score（export、build 為前置步驟；共 ${SECONDS}s，log 在 $logs）"
