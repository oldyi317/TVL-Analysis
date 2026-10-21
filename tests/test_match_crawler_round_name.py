"""輪次正名：官網 2024-25 賽季把總決賽寫成「總冠軍賽 Week 19」，須正規化為「總決賽 Week 19」（#35）。"""

from datetime import date

import pytest

from src.etl.match_crawler import parse_round_name
from src.etl.stats_crawler import _parse_match_title
from src.utils.seasons import Season

SEASON_21 = Season(21, "2025-26", date(2025, 7, 1), date(2026, 6, 30))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2025-03-22 16:30:00 | 臺北和平籃球館 | 總冠軍賽 Week 19 | Game 261", "總決賽 Week 19"),
        ("總冠軍賽Week19", "總決賽 Week 19"),
        ("例行賽 Week 1 | Game 1", "例行賽 Week 1"),
        ("挑戰賽 Week 17", "挑戰賽 Week 17"),
        ("總決賽 Week 19", "總決賽 Week 19"),
        ("2025-11-01 18:30:00 | 國立成功大學體育館 | Game 3", None),
    ],
)
def test_parse_round_name(text, expected):
    assert parse_round_name(text) == expected


@pytest.mark.parametrize(
    "title, expected_round",
    [
        ("企業20年男子組 總冠軍賽(第三戰) 3月22日", "總決賽"),
        ("企業20年女子組 總決賽(第一戰) 3月15日", "總決賽"),
        ("企業21年男子組 第1週 11月1日", "第1週"),
        ("企業21年男子組 挑戰賽 2月7日", "挑戰賽"),
        ("企業21年男子組 熱身賽 10月1日", "未知賽別"),
    ],
)
def test_parse_match_title_round(title, expected_round):
    assert _parse_match_title(title, SEASON_21)[1] == expected_round
