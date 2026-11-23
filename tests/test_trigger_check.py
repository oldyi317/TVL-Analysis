import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "trigger-check.sh"
WRANGLER = ROOT / "score" / "wrangler.json"

# 假 gh：記下收到的參數，並把 RUNS 當成 --jq 之後的 TSV 回傳（jq 過濾本身不在這裡測）
GH = r"""#!/usr/bin/env bash
echo "$*" >> "$GH_LOG"
[ -n "$GH_FAILS" ] && { echo "gh failed" >&2; exit 1; }
printf '%b' "$RUNS"
"""

OWNER = "oldyi317"
URL = "https://github.com/oldyi317/TVL-Analysis/actions/runs/"


def _row(run_id, created, actor=OWNER, status="completed", conclusion="success"):
    return f"{run_id}\t{created}\t{actor}\t{status}\t{conclusion}\t{URL}{run_id}\\n"


FOUR = (
    _row(1, "2026-11-23T01:23:41Z")
    + _row(2, "2026-11-23T07:30:05Z")
    + _row(3, "2026-11-23T10:30:12Z")
    + _row(4, "2026-11-23T13:15:00Z")
)
EVENING = "2026-11-23T14:00:00Z"  # 台北 22:00，四個時間點都已過


def _run(tmp_path, *args, runs, now=EVENING, tz="UTC", script=SCRIPT, gh_fails=False):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text(GH, encoding="utf-8")
    gh.chmod(0o755)
    log = tmp_path / "gh.log"
    log.write_text("")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "RUNS": runs, "GH_LOG": str(log),
           "GH_FAILS": "1" if gh_fails else "", "TRIGGER_CHECK_NOW": now, "TZ": tz}
    out = subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True, timeout=30)
    out.gh_args = log.read_text()
    return out


def _line(out, needle):
    return next(line for line in out.stdout.splitlines() if needle in line)


@pytest.fixture(params=["UTC", "America/New_York"])
def tz(request):
    return request.param


def test_all_four_slots_matched_with_lag(tmp_path, tz):
    out = _run(tmp_path, "2026-11-23", runs=FOUR, tz=tz)

    assert out.returncode == 0, out.stdout + out.stderr
    full = _line(out, "09:23 ")
    assert "整包" in full and " 1 " in full and "09:23:41" in full and "41 秒" in full and "success" in full
    assert "精簡" in _line(out, "15:30 ") and "15:30:05" in _line(out, "15:30 ") and "5 秒" in _line(out, "15:30 ")
    assert " 3 " in _line(out, "18:30 ") and "18:30:12" in _line(out, "18:30 ") and "12 秒" in _line(out, "18:30 ")
    assert " 4 " in _line(out, "21:15 ") and "21:15:00" in _line(out, "21:15 ") and "0 秒" in _line(out, "21:15 ")
    assert "其餘" not in out.stdout


def test_missing_full_slot_after_its_time_fails(tmp_path, tz):
    out = _run(tmp_path, "2026-11-23", runs=FOUR.split("\\n", 1)[1], tz=tz)

    assert out.returncode == 1
    assert "沒有 run" in _line(out, "09:23 ")
    assert "屬正常" not in _line(out, "09:23 ")


def test_missing_slim_slot_is_normal(tmp_path, tz):
    runs = _row(1, "2026-11-23T01:23:41Z")
    out = _run(tmp_path, "2026-11-23", runs=runs, tz=tz)

    assert out.returncode == 0, out.stdout + out.stderr
    assert "沒有 run（當天沒有賽事時屬正常）" in _line(out, "15:30 ")


def test_other_actor_run_near_slot_is_not_matched(tmp_path, tz):
    runs = _row(9, "2026-11-23T01:23:10Z", actor="someone") + FOUR
    out = _run(tmp_path, "2026-11-23", runs=runs, tz=tz)

    assert out.returncode == 0, out.stdout + out.stderr
    assert " 1 " in _line(out, "09:23 ")
    rest = out.stdout.split("其餘")[1]
    assert "9" in rest and "09:23:10" in rest and "someone" in rest


def test_owner_run_after_ten_minutes_is_not_matched(tmp_path, tz):
    runs = _row(7, "2026-11-23T01:40:00Z") + FOUR.split("\\n", 1)[1]  # 台北 09:40
    out = _run(tmp_path, "2026-11-23", runs=runs, tz=tz)

    assert out.returncode == 1
    assert "沒有 run" in _line(out, "09:23 ")
    rest = out.stdout.split("其餘")[1]
    assert "7" in rest and "09:40:00" in rest


def test_slots_not_yet_reached_today_are_not_failures(tmp_path, tz):
    out = _run(tmp_path, runs="", now="2026-11-23T00:00:00Z", tz=tz)  # 台北 08:00，不給日期＝台北今天

    assert out.returncode == 0, out.stdout + out.stderr
    assert "還沒到" in _line(out, "09:23 ")
    assert out.stdout.count("還沒到") == 4


def test_taipei_day_is_converted_to_utc_range(tmp_path, tz):
    runs = _row(5, "2026-11-22T16:30:00Z")  # 台北 11-23 00:30
    out = _run(tmp_path, "2026-11-23", runs=runs, tz=tz)

    assert "created=2026-11-22T16:00:00Z..2026-11-23T15:59:59Z" in out.gh_args
    assert "event=workflow_dispatch" in out.gh_args
    assert "repos/oldyi317/TVL-Analysis/actions/workflows/daily-crawl.yml/runs" in out.gh_args
    assert "00:30:00" in out.stdout.split("其餘")[1]


def test_gh_failure_is_an_error(tmp_path):
    out = _run(tmp_path, "2026-11-23", runs="", gh_fails=True)

    assert out.returncode == 1


@pytest.mark.parametrize("bad", ["2026-13-01", "20261123", "2026-11-23x", "abc"])
def test_rejects_a_malformed_date(tmp_path, bad):
    out = _run(tmp_path, bad, runs="")

    assert out.returncode == 2
    assert "用法" in out.stderr
    assert out.gh_args == ""


def test_slots_come_from_wrangler_json(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "score").mkdir()
    script = tmp_path / "scripts" / "trigger-check.sh"
    shutil.copy(SCRIPT, script)
    cfg = json.loads(WRANGLER.read_text(encoding="utf-8"))
    cfg["vars"]["SLIM_TIMES"] = "12:00,20:00"
    cfg["vars"]["GITHUB_REPO"] = "someone/Other"
    (tmp_path / "score" / "wrangler.json").write_text(json.dumps(cfg), encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()

    out = _run(work, "2026-11-23", runs=_row(1, "2026-11-23T04:00:30Z", actor="someone"), script=script)

    assert out.returncode == 1  # 09:23 沒有 run
    assert "12:00 " in out.stdout and "20:00 " in out.stdout
    assert "15:30" not in out.stdout and "21:15" not in out.stdout
    assert " 1 " in _line(out, "12:00 ") and "30 秒" in _line(out, "12:00 ")
    assert "repos/someone/Other/" in out.gh_args
