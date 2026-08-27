#!/usr/bin/env bash
# TVL 每日增量更新：爬蟲 → .db 有變更才 commit/push（觸發 Streamlit Cloud 重佈）
# 本機驗證：DRY_RUN=1 bash scripts/daily_update.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$HOME/venvs/tvl/bin/python}"

"$PYTHON" -m src.etl.match_crawler
"$PYTHON" -m src.etl.stats_crawler --rosters
"$PYTHON" -m src.etl.stats_crawler --incremental

if [ -z "$(git status --porcelain data/db/tvl_database.db)" ]; then
    echo "資料庫無變更，今日無新資料。"
    exit 0
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN：資料庫有變更，略過 commit/push。"
    git status --porcelain data/db/tvl_database.db
    exit 0
fi

git add data/db/tvl_database.db
git -c user.name="github-actions[bot]" \
    -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
    commit -m "chore: 每日自動更新 $(date +%F)"
git push
