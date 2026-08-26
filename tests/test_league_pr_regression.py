from pathlib import Path

LEAGUE_PR_PY = Path(__file__).resolve().parents[1] / "src" / "app" / "tabs" / "league_pr.py"
HELPERS_PY = Path(__file__).resolve().parents[1] / "src" / "app" / "helpers.py"


def test_league_pr_has_no_direct_sql():
    """Verify league_pr.py has no direct SQL strings (no player_match_stats or FROM players)."""
    source = LEAGUE_PR_PY.read_text(encoding="utf-8")
    assert "player_match_stats" not in source
    assert "FROM players" not in source


def test_league_pr_uses_get_league_aggregated_stats():
    source = LEAGUE_PR_PY.read_text(encoding="utf-8")
    assert "get_league_aggregated_stats" in source


def test_helpers_get_league_aggregated_stats_still_returns_position_column():
    source = HELPERS_PY.read_text(encoding="utf-8")
    assert "latest.position AS position" in source
