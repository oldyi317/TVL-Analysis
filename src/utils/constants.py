"""
TVL 共用常數模組
統一管理外部系統連線資訊與隊伍對照表，避免各模組重複定義。
"""

# ── 官網 tvl.ctvba.org.tw ─────────────────────────────────────
TVL_BASE_URL = "https://tvl.ctvba.org.tw"
# 賽事列表頁：男子組 /schedule、女子組 /wschedule（/schedule-mix 不納入）
SCHEDULE_PAGES = {"M": "/schedule", "F": "/wschedule"}
# 單場數據自此賽季起以官網賽事頁為唯一來源（spec #75）；歷史賽季永不成為寫入候選
OFFICIAL_BOX_SCORE_FROM_CUP = 22

# ── 外部數據系統 (114.35.229.141)：歷史補抓工具，已停用（ADR 0009）──────────
# 單場數據與週登錄自 cup 22 起只走官網 match_stats_writer；EXT_CUP_ID 固定在外部系統最後一季 21，
# 換季不再更新，stats_crawler 對 cup_id >= OFFICIAL_BOX_SCORE_FROM_CUP 一律拒絕執行。
EXT_BASE = "http://114.35.229.141"
EXT_CUP_ID = 21

EXT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# 球隊身分單一來源為 DB 的 teams/team_aliases（種子 sql/team_seed.sql，解析用 src/utils/teams.resolve_team）；
# 只有 EXT_TEAM_MAPS、TEAM_NAME_SHORT 仍是手動維護的對照表；EXT_TEAM_MAPS 止於 cup 21，不再補新季（ADR 0009）。
# 2024-25 舊隊 team_id 8/9/10：訓練集刻意排除，見 issue #48。
LEGACY_TEAM_IDS = {8, 9, 10}
# ── 外部系統 TeamID → 本地 DB (team_id, gender) 對照表（cup 20、21 各一份，以 cup_id 為鍵）──
# TeamID 每季重編；查無該賽季時爬蟲會拋錯而不是落回其他賽季。cup 22 起不補（ADR 0009）。
EXT_TEAM_MAPS: dict[int, dict[int, tuple[int, str]]] = {
    20: {
        1: (1, "M"),    # 屏東台電(男)
        2: (2, "M"),    # 雲林美津濃(男)
        3: (8, "M"),    # 彰化三大有線(男)
        4: (9, "M"),    # 臺北Conti(男)
        5: (5, "M"),    # 龍邁建設（當季名桃園臺產）(男)
        6: (10, "M"),   # 連莊(男)
        7: (4, "F"),    # 高雄台電(女)
        8: (3, "F"),    # 臺北鯨華(女)
        9: (5, "F"),    # 新北中纖(女)
        10: (7, "F"),   # 義力營造(女)
    },
    21: {
        1: (1, "M"),   # 屏東台電(男)
        2: (2, "M"),   # 雲林美津濃(男)
        3: (7, "M"),   # 獅子王(男)
        4: (4, "M"),   # 臺北國北獅(男)
        5: (5, "M"),   # 桃園臺產(男)
        6: (4, "F"),   # 高雄台電(女)
        7: (3, "F"),   # 臺北鯨華(女)
        8: (5, "F"),   # 新北中纖(女)
        9: (7, "F"),   # 義力營造(女)
    },
}

# ── 官網全名 → 簡寫對應（依 CLAUDE.md 第 7 節）─────────────────
TEAM_NAME_SHORT = {
    "臺北鯨華女子排球隊": "臺北鯨華",
    "新北中國人纖企業女子排球隊": "新北中纖",
    "台灣電力公司女子排球隊": "高雄台電",
    "義力營造女子排球隊": "義力營造",
    "台灣電力公司男子排球隊": "屏東台電",
    "美津濃男子排球隊": "雲林美津濃",
    "桃園臺產隼鷹排球隊": "桃園臺產",
    "臺北鳳凰男子排球隊": "臺北鳳凰",
}

# ── 位置相關 ──────────────────────────────────────────────────
VALID_POSITIONS = {"OH", "MB", "OP", "S", "L"}
VALID_GENDERS = {"M", "F"}

POSITION_MAP = {
    "主攻手": "OH",  # 官網原字
    "中間手": "MB",  # 官網原字
    "副攻手": "OP",
    "舉球員": "S",  # 官網原字
    "自由球員": "L",  # 官網原字
}

# ── 外部統計系統（Match.ashx）位置用語 → 內部代碼 ─────────────
# 用語與官網 POSITION_MAP 不同（官網：主攻手/中間手/副攻手/舉球員/自由球員）， # 官網原字
# 已用資料庫既有球員資料交叉驗證（見 Phase 2 計畫文件）。
MATCH_POSITION_MAP = {
    "對角": "OP",  # 官網原字
    "長攻": "OH",  # 官網原字
    "攔中": "MB",
    "舉球": "S",
    "自由": "L",
}

# 官網 2024-25 賽季把總決賽寫成「總冠軍賽」，兩個爬蟲都正規化為「總決賽」  # 官網原字
ROUND_STAGE_ALIAS = {"總冠軍賽": "總決賽"}  # 官網原字
