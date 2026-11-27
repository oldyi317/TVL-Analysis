import pytest

from src.export.metrics import METRICS, Measure, columns, measure


@pytest.mark.parametrize("points,total", [(1, 3), (2, 7), (5, 11)])
def test_percentage_divides_before_scaling(points, total):
    result = measure("attack_success_rate", {"attack_points": points, "attack_total": total})
    assert result == Measure(points, total, points / total * 100)


@pytest.mark.parametrize("key,totals,denominator", [
    ("serve_point_rate", {"serve_points": 0, "serve_total": 0}, 0),
    ("points_per_game", {"total_points": 0, "games": 0}, 0),
    ("blocks_per_set", {"block_points": 0}, 0),
])
def test_zero_denominator(key, totals, denominator):
    assert measure(key, totals, sets=0) == Measure(0, denominator, None)


@pytest.mark.parametrize("sets,expected", [(None, None), (0, None), (4, 1.5)])
def test_per_set_only_uses_explicit_sets(sets, expected):
    result = measure("blocks_per_set", {"block_points": 6, "sets": 99}, sets=sets)
    assert result == Measure(6, sets, expected)


def test_defense_efficiency_combines_counts_before_division():
    totals = {"receive_excellent": 2, "dig_excellent": 3, "receive_total": 4, "dig_total": 9}
    assert measure("defense_efficiency", totals) == Measure(5, 13, 5 / 13 * 100)


def test_defense_load_is_an_integer_without_denominator():
    result = measure("defense_load", {"receive_total": 30, "dig_total": 17})
    assert result == Measure(47, None, 47)
    assert type(result.value) is int


def test_unknown_key_and_missing_column_raise_key_error():
    with pytest.raises(KeyError, match="unknown"):
        measure("unknown", {})
    with pytest.raises(KeyError, match="games"):
        measure("blocks_per_game", {"block_points": 2})


def test_columns_lists_only_the_required_sorted_fields():
    assert columns(["blocks_per_set", "defense_load"]) == ("block_points", "dig_total", "receive_total")
    for key in METRICS:
        fields = columns([key])
        assert "sets" not in fields
        assert measure(key, dict.fromkeys(fields, 1), sets=1).value is not None
