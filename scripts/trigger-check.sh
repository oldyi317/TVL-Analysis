#!/usr/bin/env bash
# 某一天每日更新（workflow_dispatch）在每個預定時間點有沒有被比分服務準時啟動。
# 驗收與排查「該有的 run 沒出現」時用，取代手寫 gh api --jq。
# 時間點、repo、workflow 都讀 score/wrangler.json 的 vars，這裡不再寫一份。
#
# 用法：bash scripts/trigger-check.sh [YYYY-MM-DD]（台北日期，預設台北今天）
# 前置：gh（已登入）、GNU date、python3（只用標準庫 json 讀 wrangler.json）
# TRIGGER_CHECK_NOW=2026-11-23T04:00:00Z 可覆寫「現在」，只供測試用。
# exit：整包時間點已過卻沒有配到 run → 1；gh 查詢失敗 → 1；日期格式不對 → 2；其餘 0。
set -uo pipefail

usage() { echo "用法：bash scripts/trigger-check.sh [YYYY-MM-DD]（台北日期）" >&2; exit 2; }

now=$(date -u -d "${TRIGGER_CHECK_NOW:-now}" +%s) || exit 1
day=${1:-$(TZ=Asia/Taipei date -d "@$now" +%F)}
[[ $day =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && date -d "$day" >/dev/null 2>&1 || usage
[ $# -le 1 ] || usage

cfg="$(dirname "${BASH_SOURCE[0]}")/../score/wrangler.json"
{ read -r full_time; read -r slim_times; read -r repo; read -r workflow; } < <(python3 -c '
import json, sys
v = json.load(open(sys.argv[1], encoding="utf-8"))["vars"]
for k in ("FULL_TIME", "SLIM_TIMES", "GITHUB_REPO", "GITHUB_WORKFLOW"):
    print(v[k])' "$cfg") || { echo "讀不到 $cfg 的 vars" >&2; exit 1; }
[ -n "${workflow:-}" ] || { echo "讀不到 $cfg 的 vars" >&2; exit 1; }
owner=${repo%%/*}

# 台北日的 00:00–24:00；created 區間含兩端，所以迄點取前 1 秒
start=$(date -u -d "$day 00:00:00 +0800" +%s)
end=$((start + 86399))
iso() { date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ; }

runs=$(gh api "repos/$repo/actions/workflows/$workflow/runs?event=workflow_dispatch&created=$(iso "$start")..$(iso "$end")&per_page=100" \
	--jq '.workflow_runs[] | [.id, .created_at, .actor.login, .status, (.conclusion // "-"), .html_url] | @tsv') || exit 1

declare -A epoch actor status conclusion url
ids=()
while IFS=$'\t' read -r id created who st concl link; do
	[ -n "$id" ] || continue
	ids+=("$id"); epoch[$id]=$(date -u -d "$created" +%s); actor[$id]=$who
	status[$id]=$st; conclusion[$id]=$concl; url[$id]=$link
done <<< "$runs"
# 依啟動時間排序，同一時間點要取最早的一筆
mapfile -t ids < <(for id in "${ids[@]}"; do echo "${epoch[$id]} $id"; done | sort -n | cut -d' ' -f2)

taipei() { TZ=Asia/Taipei date -d "@$1" +%T; }
fail=0
declare -A used

# 中文字佔兩格，printf 的欄寬對不齊，標題列手動排
echo "時間點  模式  run          啟動時刻  晚幾秒  結論  網址"
slots=("$full_time 整包")
IFS=, read -ra slim <<< "$slim_times"
for t in "${slim[@]}"; do slots+=("$t 精簡"); done
for slot in "${slots[@]}"; do
	t=${slot% *}; mode=${slot#* }
	slot_epoch=$(date -u -d "$day $t:00 +0800" +%s)
	match=
	for id in "${ids[@]}"; do
		lag=$((epoch[$id] - slot_epoch))
		if [ "${actor[$id]}" = "$owner" ] && [ -z "${used[$id]:-}" ] && [ "$lag" -ge 0 ] && [ "$lag" -lt 600 ]; then
			match=$id; break
		fi
	done
	if [ -n "$match" ]; then
		used[$match]=1
		printf '%-6s  %s  %-11s  %s  %4s 秒  %s  %s\n' "$t" "$mode" "$match" "$(taipei "${epoch[$match]}")" \
			"$((epoch[$match] - slot_epoch))" "$([ "${status[$match]}" = completed ] && echo "${conclusion[$match]}" || echo "${status[$match]}")" "${url[$match]}"
	elif [ "$now" -lt "$slot_epoch" ]; then
		printf '%-6s  %s  還沒到\n' "$t" "$mode"
	elif [ "$mode" = 整包 ]; then
		printf '%-6s  %s  沒有 run\n' "$t" "$mode"
		fail=1
	else
		printf '%-6s  %s  沒有 run（當天沒有賽事時屬正常）\n' "$t" "$mode"
	fi
done

first=1
for id in "${ids[@]}"; do
	[ -z "${used[$id]:-}" ] || continue
	[ -z "$first" ] || { printf '\n其餘 workflow_dispatch run（沒配到時間點）：\n'; first=; }
	printf '%-11s  %s  %s  %s\n' "$id" "$(taipei "${epoch[$id]}")" "${actor[$id]}" \
		"$([ "${status[$id]}" = completed ] && echo "${conclusion[$id]}" || echo "${status[$id]}")"
done
exit $fail
