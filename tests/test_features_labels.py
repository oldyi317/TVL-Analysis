import pytest

from src.models.features import normalize_match_team
from src.utils.constants import LEGACY_TEAMS, OPP_SHORT_TO_TEAM

# matches 表實際出現過的 14 種隊名（2026-08-27 實查）
MATCH_TEAM_NAMES = [
    "屏東台電", "屏東台電男排", "彰化三大有線", "新北中纖", "桃園臺產",
    "獅子王", "義力營造", "臺北Conti", "臺北國北獅", "臺北鯨華",
    "連莊", "雲林美津濃", "高雄台電", "高雄台電女排",
]


def test_all_match_team_names_normalize():
    for name in MATCH_TEAM_NAMES:
        short = normalize_match_team(name)
        assert short in OPP_SHORT_TO_TEAM or short in LEGACY_TEAMS


def test_suffix_variants_map_to_short():
    assert normalize_match_team("屏東台電男排") == "屏東台電"
    assert normalize_match_team("高雄台電女排") == "高雄台電"


def test_unknown_team_raises():
    with pytest.raises(ValueError, match="未知隊名"):
        normalize_match_team("不存在的隊")
