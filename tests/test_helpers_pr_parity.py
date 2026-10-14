import json
import math

from src.app import helpers
from src.export.site_data import export_site_data
from tests.conftest import connect_seeded, seed_player, seed_registration, seed_stat

PR_COLUMNS = {
    "asr": "attack_success_rate", "gp_pct": "receive_excellent_rate", "ace_pct": "serve_point_rate",
    "dig_pct": "dig_excellent_rate", "set_pct": "set_excellent_rate", "blk_per_set": "blocks_per_set",
    "def_pct": "defense_efficiency",
}
RATIO_COLUMNS = {**PR_COLUMNS, "ppg": "points_per_game", "def_load": "defense_load"}


def _seed(conn, player_id, position, sets, stats, gender="M", team_id=1):
    seed_player(conn, player_id, f"球員{player_id}", gender)
    seed_registration(conn, player_id, player_id, team_id=team_id, position=position, gender=gender)
    seed_stat(conn, player_id, (sets, *stats))


def test_old_site_helper_and_exporter_produce_same_ratios_and_pr(tmp_db_path, tmp_path, monkeypatch):
    """舊站 get_league_aggregated_stats 與匯出器算的是同一套：同分、單人位置、零分母都要一致。"""
    conn = connect_seeded(tmp_db_path)
    #          atk_tot atk_pts blk srv_tot srv_pts rcv_tot rcv_exc dig_tot dig_exc set_tot set_exc pts
    _seed(conn, 1, "OH", 10, (20, 5, 1, 10, 2, 10, 5, 8, 4, 4, 2, 8))
    _seed(conn, 2, "OH", 10, (20, 10, 3, 10, 2, 10, 7, 8, 2, 0, 0, 15))
    _seed(conn, 3, "OH", 10, (20, 10, 3, 0, 0, 10, 7, 8, 2, 0, 0, 13), team_id=2)
    _seed(conn, 4, "OH", 10, (20, 15, 0, 10, 5, 0, 0, 0, 0, 4, 4, 20), team_id=2)
    _seed(conn, 5, "MB", 10, (20, 19, 7, 10, 1, 0, 0, 2, 1, 0, 0, 27), team_id=4)
    _seed(conn, 6, "OH", 4, (20, 19, 0, 10, 0, 10, 0, 8, 0, 0, 0, 19), team_id=4)
    _seed(conn, 7, None, 10, (20, 1, 0, 10, 0, 10, 0, 8, 0, 0, 0, 1), team_id=5)
    _seed(conn, 8, "OH", 10, (20, 1, 0, 10, 0, 10, 0, 8, 0, 0, 0, 1), gender="F", team_id=3)
    conn.commit()
    conn.close()

    monkeypatch.setattr(helpers, "DB_PATH", tmp_db_path)
    helpers.load_data.clear()
    helpers.get_league_aggregated_stats.clear()
    df = helpers.get_league_aggregated_stats("M")

    export_site_data(tmp_db_path, tmp_path / "out")
    payload = json.loads((tmp_path / "out" / "player_advanced.json").read_text(encoding="utf-8"))
    exported = {p["player_id"]: p for p in payload["players"] if p["cup_id"] == 21 and p["group"] == "men"}

    assert sorted(df["player_id"]) == [1, 2, 3, 4, 5, 7]
    assert exported[6]["insufficient_sets"] is True
    for _, row in df.iterrows():
        entry = exported[row["player_id"]]
        for col, key in RATIO_COLUMNS.items():
            assert round(float(row[col]), 2) == entry["ratios"][key], (row["player_id"], col)
        for col, key in PR_COLUMNS.items():
            old, new = row[f"{col}_pr"], entry["pr"][key]
            if new is None:
                assert math.isnan(old), (row["player_id"], col)
            else:
                assert old == new, (row["player_id"], col)
    assert math.isnan(df.set_index("player_id").loc[7, "asr_pr"])
    assert df.set_index("player_id").loc[5, "asr_pr"] == 100.0
