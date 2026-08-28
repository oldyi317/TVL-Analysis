"""
球隊層級特徵與真實勝負標籤（Phase 4）。
不依賴 streamlit，訓練 script 與測試皆直接 import。
"""

from src.utils.constants import LEGACY_TEAMS, MATCH_TEAM_ALIASES, OPP_SHORT_TO_TEAM


def normalize_match_team(name: str) -> str:
    short = MATCH_TEAM_ALIASES.get(name, name)
    if short in OPP_SHORT_TO_TEAM or short in LEGACY_TEAMS:
        return short
    raise ValueError(f"未知隊名：{name}（請補 constants.MATCH_TEAM_ALIASES）")
