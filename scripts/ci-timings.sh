#!/usr/bin/env bash
# 最近 N 次成功的 CI：web job、e2e 步驟（含 build）、Playwright 測試本身的耗時與 worker 數，最後印範圍。
# 比較改動前後的 CI 耗時先看這個範圍：GitHub 的機器同一天就可差兩成，只拿一兩筆當基準會看錯因果（#258）。
#
# 用法：bash scripts/ci-timings.sh [N]（預設 10）
set -uo pipefail

repo=oldyi317/TVL-Analysis
limit=${1:-10}
[[ $limit =~ ^[1-9][0-9]*$ ]] || { echo "用法：bash scripts/ci-timings.sh [N]" >&2; exit 2; }

runs=$(gh run list --repo "$repo" --workflow ci --status success --limit "$limit" \
	--json databaseId,createdAt,headBranch --jq '.[] | "\(.databaseId) \(.createdAt[:10]) \(.headBranch)"') || exit 1
[ -n "$runs" ] || { echo "查不到成功的 CI 紀錄" >&2; exit 1; }

# 每列：web job 秒數、e2e 步驟秒數、測試本身（Playwright 的「N passed (…)」）；查不到的 run 不放進來，免得被當成 0 秒
rows=()
# 中文字佔兩格，printf 的欄寬對不齊，標題列手動排
echo "run          日期        web job  e2e 步驟  測試本身  worker  分支"
while read -r id date branch; do
	read -r job_id job_s e2e_s < <(gh api "repos/$repo/actions/runs/$id/jobs" --jq '.jobs[] | select(.name == "web")
		| [.id, ((.completed_at | fromdate) - (.started_at | fromdate)),
		   (.steps[] | select(.name | test("e2e")) | (.completed_at | fromdate) - (.started_at | fromdate))] | @tsv')
	if [ -z "$e2e_s" ]; then
		printf '%-12s %-10s %s\n' "$id" "$date" "查不到 web job 或 e2e 步驟，不算進範圍"
		continue
	fi
	log=$(gh api "repos/$repo/actions/jobs/$job_id/logs")
	workers=$(grep -oE 'using [0-9]+ workers?' <<< "$log" | grep -oE '[0-9]+' | head -n 1)
	tests=$(grep -oE '[0-9]+ passed \([0-9.]+[ms]\)' <<< "$log" | tail -n 1 | grep -oE '[0-9.]+[ms]')
	printf '%-12s %-10s %7ss %8ss %9s %7s  %s\n' "$id" "$date" "$job_s" "$e2e_s" "${tests:--}" "${workers:--}" "$branch"
	rows+=("$job_s $e2e_s ${tests:-}")
done <<< "$runs"
[ ${#rows[@]} -gt 0 ] || { echo "每次 run 都查不到耗時，沒有範圍可印" >&2; exit 1; }

# 測試本身要從 job log 抓，抓不到的 run 只算進前兩項，範圍行標出有紀錄的次數
printf '%s\n' "${rows[@]}" | awk '
	function sec(t) { return t ~ /m$/ ? substr(t, 1, length(t) - 1) * 60 : substr(t, 1, length(t) - 1) }
	function lo(a, b) { return a == "" || b < a ? b : a }
	function hi(a, b) { return a == "" || b > a ? b : a }
	{ jl = lo(jl, $1); jh = hi(jh, $1); el = lo(el, $2); eh = hi(eh, $2) }
	$3 != "" { s = sec($3); tl = lo(tl, s); th = hi(th, s); n++ }
	END {
		printf "\n範圍（%d 次）：web job %d–%d 秒、e2e 步驟 %d–%d 秒", NR, jl, jh, el, eh
		if (n) printf "、測試本身 %.1f–%.1f 秒", tl, th
		if (n && n < NR) printf "（有紀錄 %d/%d 次）", n, NR
		print ""
	}'
