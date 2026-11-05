from src.utils.constants import EXT_BASE


def test_ext_base_uses_statistics_domain():
    assert EXT_BASE.startswith("https://statistics.ctvba.org.tw")
