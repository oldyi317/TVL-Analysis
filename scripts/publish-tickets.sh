#!/usr/bin/env bash
# 把一個目錄裡的 spec 與子票草稿開成 GitHub issue，掛 sub-issue 與阻擋邊，最後讀回驗證。
#
# 用法：bash scripts/publish-tickets.sh <目錄> [--parent <issue 編號>]
#   <目錄>/manifest.tsv  一行一張票，四欄用 tab 分隔：代號、label、被哪些代號擋（空白分隔，可空）、標題
#                        代號 SPEC 那一行是母票，內文讀 <目錄>/spec.md；沒有這一行就用 --parent 指定既有的母票
#   <目錄>/<代號>.md      每張票的內文
#   標題與內文裡的 {{SPEC}}、{{<代號>}} 會換成 #編號（標題只支援 {{SPEC}}）
# 可重跑：開過的票記在 <目錄>/.published，不重開。
# 環境變數：DRY_RUN=1 只印出會做什麼；REPO 預設取目前 repo
set -euo pipefail

if [ $# -lt 1 ]; then
	sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//' >&2
	exit 2
fi

dir=$1
parent=
[ "${2:-}" = --parent ] && parent=${3:?--parent 要帶 issue 編號}
manifest=$dir/manifest.tsv
state=$dir/.published
[ -f "$manifest" ] || { echo "找不到 $manifest" >&2; exit 1; }
repo=${REPO:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}
dry=${DRY_RUN:-}
[ -n "$dry" ] && state=$(mktemp)
touch "$state"

get() { grep -E "^$1=" "$state" | tail -1 | cut -d= -f2 || true; }
put() { echo "$1=$2" >> "$state"; }
run() { if [ -n "$dry" ]; then echo "會執行：$*"; else "$@" > /dev/null; fi; }

create() { # 標題 label 內文檔 → 印出 issue 編號
	if [ -n "$dry" ]; then
		local fake=$((9001 + $(grep -c '' "$state")))
		echo "會開票 #$fake [$2] $1" >&2
		echo "$fake"
		return
	fi
	local url
	url=$(gh issue create --repo "$repo" --title "$1" --label "$2" --body-file "$3")
	echo "${url##*/}"
}

dbid() { # issue 編號 → 資料庫 id（sub-issue 與阻擋邊的 API 要用這個，不是 #編號）
	[ -n "$dry" ] && { echo "dbid-of-$1"; return; }
	gh api "repos/$repo/issues/$1" --jq .id
}

ids=()
declare -A label blocked title
while IFS=$'\x1f' read -r id lb bl tt; do
	[ -z "$id" ] && continue
	[ -f "$dir/$([ "$id" = SPEC ] && echo spec || echo "$id").md" ] || { echo "缺 $id 的內文檔" >&2; exit 1; }
	label[$id]=$lb; blocked[$id]=$bl; title[$id]=$tt
	[ "$id" = SPEC ] || ids+=("$id")
done < <(tr '\t' '\037' < "$manifest")

spec=${parent:-$(get SPEC)}
if [ -z "$spec" ]; then
	[ -n "${title[SPEC]:-}" ] || { echo "manifest 沒有 SPEC 那一行，也沒給 --parent" >&2; exit 1; }
	spec=$(create "${title[SPEC]}" "${label[SPEC]}" "$dir/spec.md")
	put SPEC "$spec"
fi
echo "母票 #$spec"

for id in "${ids[@]}"; do
	[ -n "$(get "$id")" ] && continue
	put "$id" "$(create "${title[$id]//\{\{SPEC\}\}/#$spec}" "${label[$id]}" "$dir/$id.md")"
done

subst=(-e "s/{{SPEC}}/#$spec/g")
for id in "${ids[@]}"; do subst+=(-e "s/{{$id}}/#$(get "$id")/g"); done
tmp=$(mktemp -d)
for id in "${ids[@]}"; do
	sed "${subst[@]}" "$dir/$id.md" > "$tmp/$id.md"
	if grep -qE '\{\{[A-Za-z0-9_-]+\}\}' "$tmp/$id.md"; then
		echo "$id 還有沒換掉的佔位符：$(grep -oE '\{\{[A-Za-z0-9_-]+\}\}' "$tmp/$id.md" | sort -u | tr '\n' ' ')" >&2
		exit 1
	fi
	run gh issue edit "$(get "$id")" --repo "$repo" --body-file "$tmp/$id.md"
done

for id in "${ids[@]}"; do
	n=$(get "$id")
	if [ -z "$(get "sub:$id")" ]; then
		run gh api --method POST "repos/$repo/issues/$spec/sub_issues" -F sub_issue_id="$(dbid "$n")"
		put "sub:$id" 1
	fi
	for b in ${blocked[$id]}; do
		[ -n "$(get "$b")" ] || { echo "$id 被 $b 擋，但 manifest 沒有 $b" >&2; exit 1; }
		[ -n "$(get "dep:$id:$b")" ] && continue
		run gh api --method POST "repos/$repo/issues/$n/dependencies/blocked_by" -F issue_id="$(dbid "$(get "$b")")"
		put "dep:$id:$b" 1
	done
done

[ -n "$dry" ] && exit 0
gh api "repos/$repo/issues/$spec/sub_issues" --paginate \
	--jq '.[] | "#\(.number) [\([.labels[].name] | join(","))] 被擋=\(.issue_dependencies_summary.total_blocked_by) \(.title)"'
