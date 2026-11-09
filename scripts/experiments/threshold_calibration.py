# PROTOTYPE：丟棄式實驗腳本，回答「浮動門檻係數是否適合本聯賽」，不進正式匯出流程
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
# 執行（從 worktree 根目錄）：~/venvs/tvl/bin/python scripts/experiments/threshold_calibration.py
import csv
import logging
import re
import sqlite3
import statistics
import time
from collections import defaultdict
from dataclasses import replace

import numpy as np
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture

from src.export import leaders
from src.export.player_advanced import compute_ratios
from src.export.site_data import _advanced_scopes, _box_score_games
from src.export.thresholds import RATE_BOARDS, attempts

logging.disable(logging.WARNING)

CUPS = range(16, 22)
GENDERS = ("M", "F")
WEEK = re.compile(r"Week\s*(\d+)")
STARTERS = {"OH": 2, "MB": 2, "OP": 1, "S": 1, "L": 1}
GROUPS = {
    "attack_success_rate": ("OH", "OP", "MB"),
    "serve_point_rate": ("OH", "OP", "MB", "S"),
    "receive_excellent_rate": ("OH", "L"),
    "dig_excellent_rate": ("OH", "OP", "MB", "S", "L"),
    "set_excellent_rate": ("S",),
    "points_per_game": ("OH", "OP", "MB"),
    "blocks_per_set": ("OH", "OP", "MB"),
}


def coefs(spec):
    top = 12.0 if spec.key == "set_excellent_rate" else 8.0
    return [round(0.25 * i, 2) for i in range(1, int(top / 0.25) + 1)]


def end_value(row, spec, sets):
    ratios = compute_ratios(row, sets) if spec.key == "blocks_per_set" else compute_ratios(row)
    v = ratios[spec.key]
    return None if v is None else round(v, 2)


def starters(rows, spec, team_sets, group):
    by_team = defaultdict(list)
    for r in rows:
        if r["position"] in group and team_sets.get(r["team_id"], 0) > 0:
            a = attempts(r, spec)
            if a > 0:
                by_team[(r["team_id"], r["position"])].append((-a / team_sets[r["team_id"]], r["player_id"]))
    out = set()
    for (_, pos), lst in by_team.items():
        out |= {pid for _, pid in sorted(lst)[:STARTERS[pos]]}
    return out


def gmm_breakpoint(rows, spec, team_sets, group):
    x = np.array([attempts(r, spec) / team_sets[r["team_id"]] for r in rows
                  if r["position"] in group and team_sets.get(r["team_id"], 0) > 0]).reshape(-1, 1)
    g = GaussianMixture(n_components=2, random_state=0).fit(x)
    grid = np.linspace(x.min(), x.max(), 4000).reshape(-1, 1)
    p = g.predict_proba(grid)
    d = p[:, 0] - 0.5
    m = sorted(g.means_.ravel())
    cross = [grid[i, 0] for i in range(len(grid) - 1) if d[i] * d[i + 1] < 0 and m[0] <= grid[i, 0] <= m[1]]
    if not cross:
        return float("nan")
    mid = (m[0] + m[1]) / 2
    return min(cross, key=lambda c: abs(c - mid))


def main():
    t0 = time.time()
    conn = sqlite3.connect("file:data/db/tvl_database.db?mode=ro", uri=True)
    games_all = _box_score_games(conn)
    scopes = {(c, g): rows for c, g, rows in _advanced_scopes(conn, exclude_golden_set=True)}
    results, gmm_rows, info = [], [], []
    for cup in CUPS:
        for gender in GENDERS:
            games = [g for g in games_all.get((cup, gender), {}).values() if (g["round_name"] or "").startswith("例行賽")]
            by_week = defaultdict(list)
            for g in games:
                by_week[int(WEEK.search(g["round_name"]).group(1))].append(g)
            weeks = sorted(by_week)
            gaps = [w for w in range(weeks[0], weeks[-1] + 1) if w not in by_week]
            total = len(games)
            cum, mid_i = 0, None
            for i, w in enumerate(weeks):
                cum += len(by_week[w])
                if mid_i is None and cum >= total / 2:
                    mid_i = i
            snaps = []
            acc = []
            for w in weeks:
                acc = acc + by_week[w]
                totals = leaders.regular_totals(acc)
                ts = leaders.count_team_sets(acc)
                named = [{**{k: r[k] for k in ("player_id", "team_id", "position")}, **totals[r["player_id"]],
                          "name": str(r["player_id"]), "team_short": ""}
                         for r in scopes[(cup, gender)] if r["player_id"] in totals]
                snaps.append((named, ts))
            end_rows, end_ts = snaps[-1]
            zero_s = sorted(t for t, s in end_ts.items() if s == 0)
            info.append((cup, gender, total, len(weeks), weeks[0], weeks[-1], gaps, len(end_ts), zero_s, weeks[mid_i]))
            for base in RATE_BOARDS:
                group = GROUPS[base.key]
                pool_rows = [r for r in end_rows if r["position"] is not None]
                stars = starters(pool_rows, base, end_ts, group)
                gmm_rows.append((cup, gender, base.key, gmm_breakpoint(pool_rows, base, end_ts, group)))
                end_by_pid = {r["player_id"]: r for r in end_rows}
                for c in coefs(base):
                    spec = replace(base, coefficient=c)
                    boards = []
                    for named, ts in snaps:
                        b = leaders.compute_leaders(named, ts, boards=[spec])[0]["rows"]
                        boards.append(b)
                    att = [{r["player_id"]: attempts(r, spec) for r in named} for named, _ in snaps]
                    # 季末
                    end_b = boards[-1]
                    q = {r["player_id"] for r in end_b if r["position"] in group}
                    tp = len(q & stars)
                    prec = tp / len(q) if q else 0.0
                    rec = tp / len(stars) if stars else 0.0
                    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
                    # 名次穩定度
                    mid_b = boards[mid_i]
                    sp = float("nan")
                    if len(mid_b) >= 3:
                        mv, ev = [], []
                        for r in mid_b:
                            e = end_value(end_by_pid[r["player_id"]], spec, end_ts.get(r["team_id"], 0))
                            if e is not None:
                                mv.append(r["value"])
                                ev.append(e)
                        if len(mv) >= 3 and len(set(mv)) > 1 and len(set(ev)) > 1:
                            sp = float(spearmanr(mv, ev)[0])
                    overlap = len({r["player_id"] for r in mid_b[:10]} & {r["player_id"] for r in end_b[:10]}) / 10
                    # 卡位率
                    stall = n = 0
                    for i in range(3, len(snaps)):
                        for r in boards[i]:
                            n += 1
                            stall += att[i][r["player_id"]] == att[i - 3].get(r["player_id"], 0)
                    results.append((cup, gender, base.key, c, f1, prec, rec, sp, overlap,
                                    stall / n if n else 0.0, len(end_b)))
    with open("scripts/experiments/threshold_calibration_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cup_id", "gender", "board", "coefficient", "f1", "precision", "recall", "spearman",
                    "top10_overlap", "stall_rate", "board_size"])
        w.writerows(results)
    with open("scripts/experiments/threshold_calibration_gmm.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cup_id", "gender", "board", "breakpoint"])
        w.writerows(gmm_rows)
    # 第二輪：疊合曲線
    import pandas as pd
    df = pd.DataFrame(results, columns=["cup_id", "gender", "board", "coefficient", "f1", "precision", "recall",
                                        "spearman", "top10_overlap", "stall_rate", "board_size"])
    g = df.groupby(["board", "coefficient"], sort=False)
    pooled = g.agg(f1_med=("f1", "median"), precision_med=("precision", "median"), recall_med=("recall", "median"),
                   stall_med=("stall_rate", "median"), size_med=("board_size", "median"),
                   spearman_med=("spearman", "median"), top10_med=("top10_overlap", "median"),
                   n_pools=("f1", "count")).reset_index()
    pooled.to_csv("scripts/experiments/threshold_calibration_pooled.csv", index=False)
    pooled_results = {}
    for base in RATE_BOARDS:
        d = pooled[pooled.board == base.key]
        ok = d[(d.stall_med <= 0.02) & (d.size_med <= 50)]
        best = ok.f1_med.max()
        rec = ok[ok.f1_med == best].coefficient.min()
        plateau = ok[ok.f1_med >= best - 0.02].coefficient
        lo, hi = plateau.min(), plateau.max()
        inside = lo <= base.coefficient <= hi
        verdict = "已驗證" if inside else ("建議改" if abs(rec - base.coefficient) >= 0.5 else "邊緣")
        pooled_results[base.key] = (rec, lo, hi, verdict)
    # 摘要
    expected = 12 * sum(len(coefs(s)) for s in RATE_BOARDS)
    print(f"對真實資料實跑：{len(info)}/12 池；results.csv 列數 {len(results)}（預期 {expected}）")
    for i in info:
        print(f"cup {i[0]} {i[1]}: 例行賽場次 {i[2]}，週數 {i[3]}（Week {i[4]}–{i[5]}），缺週 {i[6] or '無'}，"
              f"有局數球隊 {i[7]}，S=0 球隊 {i[8] or '無'}，季中切點 Week {i[9]}")
    for base in RATE_BOARDS:
        recs = []
        for cup in CUPS:
            for gender in GENDERS:
                ok = [r for r in results if r[:3] == (cup, gender, base.key) and r[9] <= 0.02 and r[10] <= 50]
                if ok:
                    recs.append(min(ok, key=lambda r: (-r[4], r[3]))[3])
        med = statistics.median(recs)
        agree = sum(abs(r - med) <= 0.5 for r in recs)
        gm = [g[3] for g in gmm_rows if g[2] == base.key and g[3] == g[3]]
        verdict = "建議改" if abs(med - base.coefficient) >= 0.5 and agree >= 9 else "已驗證"
        print(f"{base.key}: 現值 {base.coefficient} 建議 {med:g}（有解 {len(recs)}/12）一致 {agree}/12 "
              f"GMM 中位數 {statistics.median(gm):.2f} → {verdict}")
    for base in RATE_BOARDS:
        rec, lo, hi, v = pooled_results[base.key]
        print(f"[第二輪疊合] {base.key}: 現值 {base.coefficient} 疊合建議 {rec:g} 高原 {lo:g}–{hi:g} → {v}")
    print(f"耗時 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
