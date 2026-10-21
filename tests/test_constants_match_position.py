from src.utils.constants import MATCH_POSITION_MAP, VALID_POSITIONS


def test_match_position_map_covers_five_positions():
    assert set(MATCH_POSITION_MAP.keys()) == {"對角", "長攻", "攔中", "舉球", "自由"}


def test_match_position_map_values_are_valid_codes():
    assert set(MATCH_POSITION_MAP.values()) == VALID_POSITIONS


def test_known_player_position_mappings():
    # 已用 DB 既有資料驗證過的具體對照（見計畫文件的驗證表格）
    verified = {
        "對角": "OP",  # 張瓈文 #2 新北中纖
        "長攻": "OH",  # 劉映彤 #6 新北中纖
        "攔中": "MB",  # 杜家馨 #17 新北中纖
        "舉球": "S",   # 陳妘臻 #9 新北中纖
        "自由": "L",   # 范張予馨 #16 新北中纖
    }
    for raw, expected in verified.items():
        assert MATCH_POSITION_MAP[raw] == expected
