from pathlib import Path
import json

import pytest
from bs4 import BeautifulSoup

from src.etl.official_box_score import parse_box_score


FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name="wgame_281"):
    return (FIXTURES / f"boxscore_{name}.html").read_text(encoding="utf-8")


def test_womens_box_score_preserves_statistics_and_skips_empty_names():
    result = parse_box_score(fixture())
    home, away = result["teams"]
    assert [home["name"], away["name"]] == ["臺北鯨華", "高雄台電"]
    assert [home["source_row_count"], away["source_row_count"]] == [16, 17]
    assert [len(home["players"]), len(away["players"])] == [16, 16]
    lin = next(p for p in home["players"] if p["name"] == "林良黛")
    assert (lin["attack_points"], lin["attack_total"], lin["total_points"]) == (3, 14, 3)
    kim = next(p for p in home["players"] if p["name"] == "金彩娜")
    assert (kim["set_excellent"], kim["set_total"]) == (37, 111)
    assert [home["player_points"], away["player_points"]] == [58, 63]
    assert result["writable"] is True
    assert result["warnings"] == []


def test_mens_links_and_zero_rows_are_preserved():
    home, away = parse_box_score(fixture("game_331"))["teams"]
    assert [len(home["players"]), len(away["players"])] == [14, 14]
    shi = home["players"][0]
    assert (shi["name"], shi["jersey_number"], shi["position"]) == ("施宇祥", 1, "OH")
    assert (shi["block_points"], shi["block_total"], shi["serve_points"], shi["serve_total"]) == (1, 4, 0, 7)
    assert (shi["receive_excellent"], shi["receive_total"], shi["dig_excellent"], shi["dig_total"]) == (7, 10, 1, 5)
    chiu = next(p for p in home["players"] if p["name"] == "邱煒皓")
    assert chiu["all_zero"] is True
    assert shi["all_zero"] is False


def test_golden_set_historical_page_is_parsed_but_not_writable():
    result = parse_box_score(fixture("game_328"))
    regular = parse_box_score(fixture("game_331"))
    home, away = result["teams"]
    assert [home["player_points"], away["player_points"]] == [9, 14]
    keys = set(regular["teams"][0]["players"][0])
    for team in result["teams"]:
        assert team["players"]
        assert all(set(p) == keys for p in team["players"])
    captain = next(p for p in home["players"] if p["jersey_number"] == 99)
    assert captain["name"] == "黃正良"
    assert all("(C)" not in p["name"] for t in result["teams"] for p in t["players"])
    assert result["writable"] is False
    assert any("姓名空白但有非零數值" in e for e in result["errors"])


def test_changed_player_score_blocks_the_entire_match_and_explains_difference():
    soup = BeautifulSoup(fixture(), "html.parser")
    soup.select_one('#boxscore_1 tbody td[title="win_points"]').string = "4"
    result = parse_box_score(str(soup))
    assert result["writable"] is False
    assert any("臺北鯨華" in e and "差 +1" in e for e in result["errors"])


def test_unnamed_nonzero_row_blocks_even_when_points_still_match():
    soup = BeautifulSoup(fixture(), "html.parser")
    soup.select('#boxscore_11 tbody tr')[-1].select('td')[9].string = "1 - 2"
    result = parse_box_score(str(soup))
    assert result["writable"] is False
    assert any("高雄台電" in e and "#20" in e and "姓名空白" in e for e in result["errors"])
    assert result["warnings"] == []
    row = result["teams"][1]["unresolved_rows"][0]
    assert (row["receive_excellent"], row["receive_total"]) == (1, 2)


def test_unknown_position_is_none_and_warns():
    soup = BeautifulSoup(fixture(), "html.parser")
    soup.select('#boxscore_1 tbody tr')[0].select('td')[2].string = "未知"
    home = parse_box_score(str(soup))
    player = home["teams"][0]["players"][0]
    assert player["position"] is None
    assert player["position_text"] == "未知"
    assert any("臺北鯨華" in w and "未知位置 未知" in w for w in home["warnings"])


def test_official_player_id_is_read_from_links():
    men = parse_box_score(fixture("game_331"))
    ids = [p["official_player_id"] for t in men["teams"] for p in t["players"]]
    assert all(isinstance(i, int) and i > 0 for i in ids)
    women = parse_box_score(fixture())
    lin = next(p for p in women["teams"][0]["players"] if p["name"] == "林良黛")
    assert lin["official_player_id"] == 59


@pytest.mark.parametrize("text", ["１ - ３", "٣ - ٤"])
def test_non_ascii_digits_are_rejected(text):
    soup = BeautifulSoup(fixture(), "html.parser")
    soup.select_one('#boxscore_1 tbody tr').select('td')[3].string = text
    with pytest.raises(ValueError):
        parse_box_score(str(soup))


def test_page_total_mismatch_only_warns():
    soup = BeautifulSoup(fixture(), "html.parser")
    soup.select('#boxscore_1 tfoot b')[-1].string = "93"
    result = parse_box_score(str(soup))
    assert result["writable"] is True
    assert result["errors"] == []
    assert any("臺北鯨華" in w and "差 -1" in w for w in result["warnings"])


@pytest.mark.parametrize("damage", ["missing_team", "missing_footer", "bad_pair", "reordered_columns"])
def test_incomplete_or_changed_layout_is_rejected(damage):
    soup = BeautifulSoup(fixture(), "html.parser")
    if damage == "missing_team":
        soup.select_one('#boxscore_11').decompose()
    elif damage == "missing_footer":
        soup.select_one('#boxscore_1 tfoot').decompose()
    elif damage == "bad_pair":
        soup.select_one('#boxscore_1 tbody tr').select('td')[3].string = "待補"
    else:
        soup.select_one('#boxscore_1 thead tr').select('th')[3].string = "攔網"
    with pytest.raises(ValueError):
        parse_box_score(str(soup))


@pytest.mark.parametrize("gender,game_id,prefix", [("F", 281, "wgame"), ("M", 331, "game")])
def test_dry_run_prints_parsed_results_without_opening_database(monkeypatch, capsys, gender, game_id, prefix):
    from src.etl.official_box_score import main
    from src.utils.constants import EXT_HEADERS, TVL_BASE_URL
    import requests

    def get(url, *, headers, timeout):
        assert url == f"{TVL_BASE_URL}/{prefix}/{game_id}"
        assert headers == EXT_HEADERS
        assert timeout == 15
        response = requests.Response()
        response.status_code = 200
        response._content = fixture(f"{prefix}_{game_id}").encode()
        return response

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run 不得開啟資料庫")

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr("sqlite3.connect", forbidden)
    assert main(["--dry-run", "--game-id", str(game_id), "--gender", gender]) == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["gate"] == "可寫入"
    assert len(report["teams"]) == 2
    assert report["teams"][0]["players"][0]["name"] in output


@pytest.mark.parametrize("failure", ["network", "layout", "gate"])
def test_dry_run_failures_return_nonzero_with_context(monkeypatch, capsys, failure):
    from src.etl.official_box_score import main
    import requests

    def get(*args, **kwargs):
        if failure == "network":
            raise requests.Timeout("測試逾時")
        response = requests.Response()
        response.status_code = 200
        html = fixture().replace('title="win_points">3</td>', 'title="win_points">4</td>', 1)
        response._content = ("<html></html>" if failure == "layout" else html).encode()
        return response

    monkeypatch.setattr(requests, "get", get)
    assert main(["--dry-run", "--game-id", "281", "--gender", "F"]) == 1
    output = capsys.readouterr()
    if failure == "gate":
        report = json.loads(output.out)
        assert report["gate"] == "不可寫入"
        assert report["errors"]
    else:
        assert "F/281" in output.err
        assert "/wgame/281" in output.err
