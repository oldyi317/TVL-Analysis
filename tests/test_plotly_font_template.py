import plotly.io as pio

from src.app.helpers import register_plotly_font_template


def test_register_plotly_font_template_sets_default_and_sizes():
    register_plotly_font_template()

    assert pio.templates.default == "plotly+tvl"

    layout = pio.templates["tvl"].layout
    assert layout.font.size == 16
    assert layout.title.font.size == 18
    assert layout.legend.font.size == 14
    assert layout.xaxis.title.font.size == 16
    assert layout.xaxis.tickfont.size == 14
    assert layout.yaxis.title.font.size == 16
    assert layout.yaxis.tickfont.size == 14
