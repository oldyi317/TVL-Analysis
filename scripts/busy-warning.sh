#!/usr/bin/env bash
# 量耗時前的提醒：本機 runner 正在跑 job，或 CPU 已經在忙，就印一行警告。只提醒、不擋，一律以 0 結束。
# 這台筆電同一套驗證連跑可差三成，背景有 daily-crawl 時慢過六成（#258），這種輪的耗時不能拿來比較。
#
# 用法：bash scripts/busy-warning.sh（verify.sh 與 e2e.sh 開跑前會呼叫）

# 開跑前 1 秒內所有核心合計的忙碌比例超過這個值就提醒；安靜時約 5%，別的 session 在跑 e2e 時超過 50%
threshold=25

if pgrep -x Runner.Worker > /dev/null; then
	echo "注意：本機 runner 正在跑 job（gh run list 看是哪個 workflow），這輪耗時不能拿來和別輪比較"
fi

# /proc/stat 第一行：cpu user nice system idle iowait irq softirq steal …
cpu_sample() {
	awk '/^cpu / { print $5 + $6, $2 + $3 + $4 + $5 + $6 + $7 + $8 + $9 }' /proc/stat
}

[ -r /proc/stat ] || exit 0
read -r idle1 total1 < <(cpu_sample)
sleep 1
read -r idle2 total2 < <(cpu_sample)
[ $((total2 - total1)) -gt 0 ] || exit 0
busy=$((100 - 100 * (idle2 - idle1) / (total2 - total1)))
if [ "$busy" -gt "$threshold" ]; then
	echo "注意：開跑前 CPU 已有 ${busy}% 在忙，這輪耗時不能拿來和別輪比較"
fi
exit 0
