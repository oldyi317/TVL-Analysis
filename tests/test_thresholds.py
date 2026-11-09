from src.export.thresholds import eligible_metrics


def counts(**changes):
    return {"position": "OH", "attack_total": 0, "serve_total": 0, "receive_total": 0, "dig_total": 0, "set_total": 0, **changes}


def test_attack_success_rate_needs_attack_total_twice_team_sets():
    assert "attack_success_rate" in eligible_metrics(counts(attack_total=20), 10)
    assert "attack_success_rate" not in eligible_metrics(counts(attack_total=19), 10)
    assert "attack_success_rate" in eligible_metrics(counts(attack_total=40), 20)
    assert "attack_success_rate" not in eligible_metrics(counts(attack_total=39), 20)


def test_serve_receive_dig_set_rates_use_their_own_count_and_coefficient():
    cases = [("serve_point_rate", "serve_total", 17), ("receive_excellent_rate", "receive_total", 30),
             ("dig_excellent_rate", "dig_total", 10)]
    for key, field, needed in cases:
        assert key in eligible_metrics(counts(**{field: needed}), 10), key
        assert key not in eligible_metrics(counts(**{field: needed - 1}), 10), key
    assert "serve_point_rate" in eligible_metrics(counts(serve_total=35), 20)
    assert "serve_point_rate" not in eligible_metrics(counts(serve_total=34), 20)
    assert "set_excellent_rate" in eligible_metrics(counts(position="S", set_total=80), 10)
    assert "set_excellent_rate" not in eligible_metrics(counts(position="S", set_total=79), 10)


def test_points_group_uses_attack_plus_serve_total():
    group = {"points_per_game", "blocks_per_set", "blocks_per_game", "defense_load"}
    assert group <= eligible_metrics(counts(attack_total=18, serve_total=12), 10)
    assert not group & eligible_metrics(counts(attack_total=18, serve_total=11), 10)


def test_defense_efficiency_met_by_receive_or_dig():
    assert "defense_efficiency" in eligible_metrics(counts(receive_total=30), 10)
    assert "defense_efficiency" in eligible_metrics(counts(dig_total=10), 10)
    assert "defense_efficiency" not in eligible_metrics(counts(receive_total=29, dig_total=9), 10)


def test_blocks_per_set_never_eligible_without_team_sets_and_floor_is_ten():
    assert "blocks_per_set" not in eligible_metrics(counts(attack_total=10), 0)
    assert "blocks_per_game" in eligible_metrics(counts(attack_total=10), 0)
    assert "attack_success_rate" in eligible_metrics(counts(attack_total=10), 3)
    assert "attack_success_rate" not in eligible_metrics(counts(attack_total=9), 3)


def test_set_excellent_rate_only_for_setters():
    assert "set_excellent_rate" not in eligible_metrics(counts(position="OH", set_total=100), 10)
    assert "set_excellent_rate" not in eligible_metrics(counts(position=None, set_total=100), 10)
    assert "set_excellent_rate" in eligible_metrics(counts(position="S", set_total=100), 10)
