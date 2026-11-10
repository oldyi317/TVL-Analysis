#!/usr/bin/env bash
# 在 Linux 檔案系統開 ticket 用的 git worktree，並備好前端依賴與資料。
# repo 在 /mnt/d（Windows 檔案系統）上 vite build 要十幾分鐘，搬到 ~/work 只要幾秒。
#
# 用法：bash scripts/new-worktree.sh <分支> [base]
#   <分支>  已存在就直接檢出；不存在就從 base 開新分支
#   [base]  預設 origin/main；spec 指定整合分支時傳它，例如 origin/feat/shell-redesign
# 環境變數：TVL_WORKTREE_HOME（預設 ~/work）、TVL_PYTHON（預設 ~/venvs/tvl/bin/python）
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
	sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//' >&2
	exit 2
fi

branch=$1
base=${2:-origin/main}
repo=$(cd "$(dirname "$0")/.." && pwd)
dir=${TVL_WORKTREE_HOME:-$HOME/work}/tvl-${branch##*/}
python=${TVL_PYTHON:-$HOME/venvs/tvl/bin/python}

if [ -e "$dir" ]; then
	echo "已經有 $dir，不覆蓋。要重開請先 git worktree remove 它。" >&2
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

echo
echo "worktree 好了：$dir（分支 $branch）"
echo "驗證：cd $dir && bash scripts/verify.sh"
