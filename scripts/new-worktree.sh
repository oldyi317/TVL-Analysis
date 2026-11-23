#!/usr/bin/env bash
# 在 Linux 檔案系統開 ticket 用的 git worktree，並備好前端、push/、score/ 依賴與資料。
# repo 在 /mnt/d（Windows 檔案系統）上 vite build 要十幾分鐘，搬到 ~/work 只要幾秒。
#
# 用法：bash scripts/new-worktree.sh <分支> [base]
#   <分支>  已存在就直接檢出；不存在就從 base 開新分支
#   [base]  預設 origin/main；spec 指定整合分支時傳它，例如 origin/feat/shell-redesign
#   目錄已存在且是這個分支的 worktree 時視為接手：印未 commit 的改動、領先 base 的 commit、筆記目錄，不 fetch、不裝依賴
# 環境變數：TVL_WORKTREE_HOME（預設 ~/work）、TVL_PYTHON（預設 ~/venvs/tvl/bin/python）
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
	sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//' >&2
	exit 2
fi

branch=$1
base=${2:-origin/main}
repo=$(cd "$(dirname "$0")/.." && pwd)
dir=${TVL_WORKTREE_HOME:-$HOME/work}/tvl-${branch##*/}
python=${TVL_PYTHON:-$HOME/venvs/tvl/bin/python}

show_handover() {
	echo "已經有 $dir（分支 $branch），接著做，不重開。"
	echo
	echo "== 未 commit 的改動 =="
	local status
	status=$(git -C "$dir" status --short)
	echo "${status:-工作樹乾淨}"
	echo
	echo "== 領先 $base 的 commit =="
	if ! git -C "$dir" rev-parse --verify --quiet "$base^{commit}" >/dev/null; then
		echo "找不到 $base，略過"
	else
		local ahead
		ahead=$(git -C "$dir" log --oneline "$base..HEAD")
		echo "${ahead:-沒有領先 $base 的 commit}"
	fi
	show_notes
}

show_notes() {
	[[ $branch =~ issue-([0-9]+) ]] || return 0
	local notes=${TVL_WORKTREE_HOME:-$HOME/work}/tvl-issue-${BASH_REMATCH[1]}-notes
	echo
	if [ ! -d "$notes" ]; then
		echo "沒有筆記目錄 $notes"
		return 0
	fi
	echo "== 筆記目錄 $notes =="
	ls -A "$notes"
	if [ -f "$notes/STATE.md" ]; then
		echo
		echo "== STATE.md =="
		cat "$notes/STATE.md"
	fi
}

if [ -e "$dir" ]; then
	current=
	# git -C 會往上找到外層 repo；$dir 自己是工作樹的根才算
	if [ "$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null)" = "$(cd "$dir" 2>/dev/null && pwd -P)" ]; then
		current=$(git -C "$dir" symbolic-ref --quiet --short HEAD 2>/dev/null || true)
	fi
	if [ "$current" = "$branch" ]; then
		show_handover
		exit 0
	fi
	echo "已經有 $dir${current:+（目前分支 $current）}，不覆蓋。要重開請先 git worktree remove 它。" >&2
	exit 1
fi

cd "$repo"
git fetch --quiet origin
mkdir -p "$(dirname "$dir")"
if git show-ref --verify --quiet "refs/heads/$branch"; then
	git worktree add "$dir" "$branch"
else
	git worktree add -b "$branch" "$dir" "$base"
fi

cd "$dir"
"$python" -m src.export.site_data
cd web
npm ci --prefer-offline --no-audit --no-fund
npm run gen
for svc in push score; do
	(cd "../$svc" && npm ci --prefer-offline --no-audit --no-fund)
done

echo
echo "worktree 好了：$dir（分支 $branch）"
echo "驗證：cd $dir && bash scripts/verify.sh"
