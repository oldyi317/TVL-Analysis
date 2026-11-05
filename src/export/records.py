from src.export.leaders import COUNT_BOARDS, rank_ordered

RECORD_LIMIT = 10
RECORD_BOARDS = tuple(spec for spec in COUNT_BOARDS
                      if spec.key in ("total_points", "attack_points", "block_points", "serve_points"))
SCORING_KEYS = tuple(spec.key for spec in RECORD_BOARDS)
ROW_FIELDS = ("player_id", "name", "cup_id", "season", "games", "team_id", "team_short", "in_progress")
CUMULATIVE_FIELDS = (*ROW_FIELDS, "season_count")
LATEST_SEASON_FIELDS = tuple(f for f in ROW_FIELDS if f not in ("games", "in_progress"))
MATCH_FIELDS = (*(f for f in ROW_FIELDS if f != "games"), "match_date", "opponent", "game_id")


def _season_totals(match_rows: list[dict]) -> list[dict]:
    totals: dict[tuple[int, int], dict] = {}
    for m in match_rows:
        entry = totals.setdefault((m["cup_id"], m["player_id"]), {
            **{f: m[f] for f in ROW_FIELDS if f != "games"}, "games": 0, **{k: 0 for k in SCORING_KEYS}})
        for k in SCORING_KEYS:
            entry[k] += m["stats"][k] or 0
        if any(m["stats"].values()):
            entry["games"] += 1
            entry["team_id"], entry["team_short"] = m["team_id"], m["team_short"]
    return list(totals.values())


def _cumulative_totals(seasons: list[dict]) -> list[dict]:
    totals: dict[int, dict] = {}
    played = sorted((season for season in seasons if season["games"]), key=lambda season: season["cup_id"])
    for season in played:
        entry = totals.setdefault(season["player_id"], {
            "season_count": 0, "games": 0, "in_progress": False, **{k: 0 for k in SCORING_KEYS}})
        entry.update({f: season[f] for f in LATEST_SEASON_FIELDS})
        entry["season_count"] += 1
        entry["games"] += season["games"]
        entry["in_progress"] = entry["in_progress"] or season["in_progress"]
        for k in SCORING_KEYS:
            entry[k] += season[k]
    return list(totals.values())


def _match_values(match_rows: list[dict]) -> list[dict]:
    # 倒序輸入，讓 _top 的穩定排序在同值同人時把較近的場次排前面
    return [{**{f: m[f] for f in MATCH_FIELDS}, **{k: m["stats"][k] or 0 for k in SCORING_KEYS}}
            for m in reversed(match_rows)]


def _top(key: str, rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    ordered = sorted((r for r in rows if r[key] > 0), key=lambda r: (-r[key], r["name"], -r["cup_id"]))
    ranked = [
        {"rank": rank, "value": r[key], **{f: r[f] for f in fields}}
        for r, rank in zip(ordered, rank_ordered([r[key] for r in ordered]))
    ]
    if len(ranked) <= RECORD_LIMIT:
        return ranked
    cutoff = ranked[RECORD_LIMIT - 1]["rank"]
    return [r for r in ranked if r["rank"] <= cutoff]


def compute_records(match_rows: list[dict]) -> list[dict]:
    """一個組別依日期排序的單場數據列（不含黃金決勝局）→ 四個得分項的單季、累計、單場紀錄榜。"""
    seasons = _season_totals(match_rows)
    cumulative = _cumulative_totals(seasons)
    matches = _match_values(match_rows)
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "season_records": _top(spec.key, seasons, ROW_FIELDS),
            "cumulative_records": _top(spec.key, cumulative, CUMULATIVE_FIELDS),
            "match_records": _top(spec.key, matches, MATCH_FIELDS),
        }
        for spec in RECORD_BOARDS
    ]
