# 換季手動 Checklist

每年新賽季開打前逐項執行。未完成前每日排程照跑但抓不到新季資料（EXT_CUP_ID
仍指舊季），屬預期行為，不需先停 workflow。

## 常數更新（src/utils/constants.py 為主）

- [ ] `EXT_CUP_ID`（constants.py）：到 `http://114.35.229.141/Match.aspx` 逐一
      試 CupID 找到新季編號。
- [ ] 在 `seasons` 插一列新賽季：於 `sql/schema.sql` 的種子 `INSERT OR IGNORE INTO seasons`
      加一列（`cup_id`、名稱如 `2026-27`、起訖日 `2026-07-01`～`2027-06-30`）；下次
      `match_crawler` 執行時 `init_matches_table` 重跑 schema.sql，`INSERT OR IGNORE`
      會把新列帶進既有 DB 的 `seasons` 表（只補列，不改既有表結構），也可直接用
      sqlite 對 DB 插入。未插入時 `match_crawler` 會因找不到賽季而報錯。
- [ ] `SEASON_YEAR_MAP` 與 `DEFAULT_YEAR`（constants.py）：改成新賽季的
      「11、12 月屬 X 年，其餘屬 X+1 年」。
- [ ] `match_crawler.py` 的 `--range-start`/`--range-end` argparse 預設值：
      新季 game_id 區間要實際到官網賽程頁確認起始編號。
- [ ] 球隊身分（有新隊伍/改名才需要動）：改 `sql/team_seed.sql`（teams 與 team_aliases），
      再套用：`python -c "from src.utils.db_config import get_connection; from src.utils.teams import apply_team_seed; c=get_connection(); apply_team_seed(c); c.commit()"`。
      過渡期相容：`EXT_TEAM_MAP`、`OPP_SHORT_TO_TEAM`、`TEAM_NAME_SHORT`（constants.py）與
      `TEAM_ALIAS`（match_crawler.py）仍被爬蟲使用，需同步修改，直到爬蟲改用 DB。

## 季前作業

- [ ] 跑 `python -m src.etl.crawler` 抓新季官網名單 → `data/raw/`。
- [ ] 跑 `python -m src.etl.db_loader` 更新 players 身分層與 teams。
- [ ] 到 Actions 頁確認 daily-crawl workflow 未因 60 天不活動被自動停用（被停用則按 Re-enable；GitHub 停用前會寄警告信）。
- [ ] 手動 `workflow_dispatch` 觸發一次 daily-crawl 驗證整條管道，
      確認新季第一批資料正確落庫（cup_id 應為新季編號）。

## 模型（可延後）

- [ ] 新季累積足夠場次後跑 `python -m src.models.train --trials 100` 重訓
      match_predictor_v2；產出後跑 `python -m pytest tests/test_prediction_artifact.py` 驗契約。
