#!/usr/bin/env bash
# TVL 每日增量更新：爬蟲 → .db 邏輯內容有變更才 commit/push（push 後觸發 pages.yml 重建靜態站）
# 本機驗證：DRY_RUN=1 bash scripts/daily_update.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$HOME/venvs/tvl/bin/python}"

# 計算 DB 邏輯內容的 SHA-256 雜湊（用 SQLite dump 避免位元組雜訊）
db_hash() {
    "$PYTHON" << 'PYTHON_EOF'
import sqlite3
import hashlib

conn = sqlite3.connect('data/db/tvl_database.db')
hasher = hashlib.sha256()
for line in conn.iterdump():
    # 濾除 sqlite_sequence 計數器：upsert 即使無資料變更也會遞增 AUTOINCREMENT，屬位元組噪音
    if line.startswith('INSERT INTO "sqlite_sequence"') or line.startswith('DELETE FROM "sqlite_sequence"'):
        continue
    hasher.update(line.encode() + b'\n')
conn.close()
print(hasher.hexdigest())
PYTHON_EOF
}

# 爬蟲前記錄 hash
BEFORE=$(db_hash)

# 賽事列表（ADR 0003）：先寫未開打場次的基本資料；失敗只記 log，不中止每日更新
"$PYTHON" -m src.etl.schedule_crawler || echo "賽事列表步驟失敗（exit $?），略過。"
# 賽季報名名單（ADR 0004）：排在賽事列表後，新季賽程寫入當天就歸到新季；失敗只記 log
{ "$PYTHON" -m src.etl.crawler && "$PYTHON" -m src.etl.db_loader; } || echo "賽季報名名單步驟失敗（exit $?），略過。"
"$PYTHON" -m src.etl.match_crawler
# stats_crawler 讀 matches.round_name 決定 week_label，必須排在 match_crawler 之後（ADR 0008）
"$PYTHON" -m src.etl.stats_crawler --rosters
"$PYTHON" -m src.etl.stats_crawler --incremental
# 賽前勝率（ADR 0003）：失敗只記 log，不中止每日更新與 commit
"$PYTHON" -m src.models.predict || echo "預測步驟失敗（exit $?），略過。"

# 爬蟲後記錄 hash
AFTER=$(db_hash)

# 邏輯內容無變更：還原 .db 並結束
if [ "$BEFORE" = "$AFTER" ]; then
    echo "資料庫無邏輯變更，今日無新資料。"
    git checkout -- data/db/tvl_database.db
    exit 0
fi

# 邏輯內容有變更：依 DRY_RUN 決定是否 commit/push
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN：資料庫有變更，略過 commit/push。"
    git status --porcelain data/db/tvl_database.db
    git checkout -- data/db/tvl_database.db
    echo "DRY_RUN：已還原 .db，工作樹維持乾淨。"
    exit 0
fi

git add data/db/tvl_database.db
git -c user.name="github-actions[bot]" \
    -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
    commit -m "chore: 每日自動更新 $(date +%F)"
git push

# 依 GitHub 文件，用 GITHUB_TOKEN push 不會觸發其他 workflow（防遞迴規則；實證紀錄見
# docs/ops/season-switch.md），要自己點火 pages.yml 重建靜態站；workflow 以 GH_TOKEN 帶入 token，
# 本機手動跑沒有就略過
if [ -n "${GH_TOKEN:-}" ]; then
    gh workflow run pages.yml --ref main
else
    echo "未設定 GH_TOKEN，略過觸發 pages.yml。"
fi
