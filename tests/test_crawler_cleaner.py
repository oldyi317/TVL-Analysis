import pandas as pd
from bs4 import BeautifulSoup

from src.etl.cleaner import validate_positions
from src.etl.crawler import extract_team_name


def test_extract_team_name_maps_full_name_to_short():
    soup = BeautifulSoup("<title>臺北鯨華女子排球隊 | TVL</title>", "html.parser")
    assert extract_team_name(soup) == "臺北鯨華"


def test_validate_positions_invalidates_unknown_code():
    df = pd.DataFrame({"name": ["A", "B"], "position": ["OH", "XX"]})
    result = validate_positions(df)
    assert result.loc[0, "position"] == "OH"
    assert pd.isna(result.loc[1, "position"])
