#!/usr/bin/env bash
# 對部署好的站台量實際傳輸量（curl 帶 Accept-Encoding: gzip, br，加總 size_download）。
# 用法：bash prototypes/offline/measure-live.sh <站台網址，含子路徑，例 https://example.github.io/tvl-shell-proto/> <cached-urls-xxx.json> [前 N 個，測試用]
# 清單分兩組：precache-*（首次下載）與 data-*（獨立視窗多出來的資料檔）；單位 byte。
set -euo pipefail
site="${1:?缺站台網址}"
list="${2:?缺 cached-urls 清單}"
limit="${3:-0}"
site="${site%/}"

# 輸出「群組<TAB>相對路徑」；去掉清單裡舊伺服器的 origin 與 base 兩段
paths="$(node -e '
const [file, limit] = process.argv.slice(1);
const c = JSON.parse(require("fs").readFileSync(file, "utf8"));
for (const [name, urls] of Object.entries(c)) {
  const g = name.startsWith("precache-") ? "precache" : name.startsWith("data-") ? "data" : null;
  if (!g) continue;
  let n = 0;
  for (const u of urls) {
    if (limit > 0 && n++ >= limit) break;
    console.log(g + "\t" + decodeURIComponent(new URL(u).pathname).replace(/^\/[^/]+\//, ""));
  }
}' "$list" "$limit")"

declare -A total count
fail=0
while IFS=$'\t' read -r group path; do
  [ -z "$path" ] && continue
  # 路徑逐段編碼；/ 保留
  enc="$(node -e 'console.log(process.argv[1].split("/").map(encodeURIComponent).join("/"))' "$path")"
  out="$(curl -s -o /dev/null -w '%{http_code} %{size_download}' -H 'Accept-Encoding: gzip, br' "$site/$enc")"
  code="${out%% *}"; size="${out##* }"
  if [ "$code" != "200" ]; then echo "! $code $site/$enc" >&2; fail=$((fail + 1)); continue; fi
  total[$group]=$(( ${total[$group]:-0} + size ))
  count[$group]=$(( ${count[$group]:-0} + 1 ))
done <<< "$paths"

for g in precache data; do
  printf '%-9s %5s 檔  %10s byte  (%.2f MB)\n' "$g" "${count[$g]:-0}" "${total[$g]:-0}" "$(echo "${total[$g]:-0} / 1048576" | bc -l)"
done
all=$(( ${total[precache]:-0} + ${total[data]:-0} ))
printf '合計      %17s byte  (%.2f MB)；失敗 %d 個\n' "$all" "$(echo "$all / 1048576" | bc -l)" "$fail"
[ "$fail" -eq 0 ]
