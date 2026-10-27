"""外部統計系統路徑的防呆：已改用官網 box score 的賽季（cup_id >= 22）拒絕執行（ADR 0009）。"""
from unittest.mock import patch

import pytest

from src.etl.stats_crawler import cli, crawl_all_rosters, main
from src.utils.constants import EXT_CUP_ID, EXT_TEAM_MAPS, OFFICIAL_BOX_SCORE_FROM_CUP


def test_ext_constants_stay_at_last_external_season():
    assert EXT_CUP_ID == 21
    assert max(EXT_TEAM_MAPS) < OFFICIAL_BOX_SCORE_FROM_CUP


def test_main_refuses_official_source_cup_before_touching_db():
    with patch("src.etl.stats_crawler.get_connection") as conn, \
         pytest.raises(SystemExit, match="match_stats_writer"):
        main(incremental=True, cup_id=OFFICIAL_BOX_SCORE_FROM_CUP)
    conn.assert_not_called()


def test_crawl_all_rosters_refuses_official_source_cup_before_fetching():
    with patch("src.etl.stats_crawler.fetch_match_list") as fetch, \
         pytest.raises(SystemExit, match="match_stats_writer"):
        crawl_all_rosters(conn=None, cup_id=OFFICIAL_BOX_SCORE_FROM_CUP + 1)
    fetch.assert_not_called()


def test_cli_refuses_official_source_cup_in_both_modes():
    with patch("src.etl.stats_crawler.get_connection") as conn:
        with pytest.raises(SystemExit, match="官網"):
            cli(["--incremental", "--cup-id", "22"])
        with pytest.raises(SystemExit, match="官網"):
            cli(["--rosters", "--cup-id", "22"])
    conn.assert_not_called()


def test_last_external_season_still_runs():
    with patch("src.etl.stats_crawler.crawl_all_rosters", return_value={}) as crawl, \
         patch("src.etl.stats_crawler.get_connection"):
        cli(["--rosters", "--cup-id", "21"])
    assert crawl.call_args.kwargs["cup_id"] == 21
