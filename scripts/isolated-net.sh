#!/usr/bin/env bash
# 在自己的 network namespace 裡跑一條命令，主機網卡的變動傳不進去。
# 主機網卡的位址一變（筆電 Wi-Fi 重連），Chromium 會以 ERR_NETWORK_CHANGED 中斷進行中的請求，連 localhost 也是；
# service worker 預存到一半有一個檔失敗，整個安裝就失敗，測試等不到它接手（#304）。
# namespace 裡只有 lo 與一張位址固定的假網卡：沒有非 loopback 的網卡，瀏覽器會判成離線。
# 開不了 namespace 的環境（沒有 unshare，或核心不准）照原樣跑。
#
# 用法：bash scripts/isolated-net.sh <命令> [參數…]
setup='ip link set lo up && ip link add e2e0 type dummy && ip addr add 192.0.2.1/24 dev e2e0 && ip link set e2e0 up'

if unshare -Urn sh -c "$setup" 2> /dev/null; then
	exec unshare -Urn sh -c "$setup"' && exec "$@"' sh "$@"
fi
exec "$@"
