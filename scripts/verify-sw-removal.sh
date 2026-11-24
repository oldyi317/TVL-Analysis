#!/usr/bin/env bash
# 自我移除版 service worker 的建置檢查加真瀏覽器檢查（要建置兩次，不併進平常的 e2e）。
# 最後留下的 web/build 是一般建置，平常的 preview 與 e2e 可以接著用。
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo/web"

VITE_SW_REMOVE=1 npm run build
rm -rf build-remove
mv build build-remove
npm run build

failed=0
check() {
	local label=$1
	shift
	if "$@"; then
		echo "✓ $label"
	else
		echo "✗ $label"
		failed=$((failed + 1))
	fi
}
has() { grep -q -- "$1" "$2"; }
# 檔案或目錄不在時 grep 也回非 0，「不含」類的檢查要先確認目標存在
lacks() { [ -f "$2" ] && ! grep -q -- "$1" "$2"; }
any_file_has() { grep -rlq -- "$1" "$2/_app/immutable"; }
no_file_has() { [ -d "$2/_app/immutable" ] && ! grep -rlq -- "$1" "$2/_app/immutable"; }

check "自我移除版的 service-worker.js 含 unregister" has unregister build-remove/service-worker.js
check "自我移除版的 service-worker.js 不含 precache-" lacks 'precache-' build-remove/service-worker.js
check "自我移除版的頁面程式沒有 service-worker.js（不註冊）" no_file_has 'service-worker.js' build-remove
check "一般建置的 service-worker.js 不含 unregister" lacks unregister build/service-worker.js
check "一般建置的頁面程式有 service-worker.js（上一條的 grep 抓得到）" any_file_has 'service-worker.js' build

bash "$repo/scripts/isolated-net.sh" npx playwright test --config playwright.removal.config.ts || failed=$((failed + 1))

if [ "$failed" -gt 0 ]; then
	echo "✗ 自我移除版檢查：$failed 項失敗"
	exit 1
fi
echo "✓ 自我移除版檢查全過"
