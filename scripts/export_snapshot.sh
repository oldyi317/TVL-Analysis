#!/usr/bin/env bash
# 匯出器行為鎖：改前 save 一份 JSON 基準，改後 diff 逐檔 cmp
# 用法：bash scripts/export_snapshot.sh {save|diff} [dir]
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="web/src/lib/data"
MODE="${1:-}"
SNAP_DIR="${2:-.snapshots/export}"

if [ -n "${PYTHON:-}" ]; then
    :
elif [ -x "$HOME/venvs/tvl/bin/python" ]; then
    PYTHON="$HOME/venvs/tvl/bin/python"
else
    PYTHON="python3"
fi

case "$MODE" in
    save)
        "$PYTHON" -m src.export.site_data
        mkdir -p "$SNAP_DIR"
        cp "$DATA_DIR"/*.json "$SNAP_DIR"/
        echo "已存 $(ls "$SNAP_DIR"/*.json | wc -l) 個檔到 $SNAP_DIR"
        ;;
    diff)
        [ -d "$SNAP_DIR" ] || { echo "找不到基準目錄 $SNAP_DIR，先跑 save" >&2; exit 2; }
        "$PYTHON" -m src.export.site_data
        status=0
        for base in "$SNAP_DIR"/*.json; do
            name=$(basename "$base")
            if cmp -s "$base" "$DATA_DIR/$name"; then
                echo "IDENTICAL $name"
            else
                echo "DIFFERS   $name"
                status=1
            fi
        done
        exit $status
        ;;
    *)
        echo "用法：bash scripts/export_snapshot.sh {save|diff} [dir]" >&2
        exit 2
        ;;
esac
