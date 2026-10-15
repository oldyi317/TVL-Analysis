
import pytest
from bs4 import BeautifulSoup

from src.etl.crawler import parse_player_card


def _card(p_html: str, h5: str = ""):
    html = f"""<div class="col-md-3"><div class="PC_only">
<h3 class="player_number"><small>#</small>7</h3>
<h3 class="fs16">張祐晨 <span>主攻手</span></h3>
<h5 class="fs12">{h5}</h5>
<p>{p_html}</p></div></div>"""
    return BeautifulSoup(html, "html.parser").find("div", class_="col-md-3")

def test_new_two_i_card():
    p = parse_player_card(_card("<i>188cm</i> ｜ <i>84kg</i>"))
    assert (p["dob"], p["height_cm"], p["weight_kg"]) == (None, 188.0, 84.0)
    assert p["jersey_number"] == 7 and p["name"] == "張祐晨"

def test_old_three_i_card():
    p = parse_player_card(_card("<i>2004.07.17</i>｜<i>187cm</i>｜<i>80kg</i>"))
    assert (p["dob"], p["height_cm"], p["weight_kg"]) == ("2004-07-17", 187.0, 80.0)

def test_partial_fields():
    p = parse_player_card(_card("<i>2004.07.17</i>"))
    assert (p["dob"], p["height_cm"], p["weight_kg"]) == ("2004-07-17", None, None)
