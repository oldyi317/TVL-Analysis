import logging

from src.utils.logger import _resolve_level


def test_resolve_level_reads_valid_level_name():
    assert _resolve_level("DEBUG") == logging.DEBUG
    assert _resolve_level("info") == logging.INFO


def test_resolve_level_falls_back_to_info_for_invalid_name():
    assert _resolve_level("NOT_A_LEVEL") == logging.INFO
