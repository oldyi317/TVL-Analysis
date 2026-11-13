import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci-timings.sh"

# 假 gh：只依網址回傳 --jq 之後的結果，jq 過濾本身不在這裡測
GH = r"""#!/usr/bin/env bash
case "$*" in
	"run list"*) printf '%b' "$RUNS" ;;
	*runs/111/jobs*) printf '9001\t96\t56\n' ;;
	*runs/222/jobs*) printf '9002\t150\t76\n' ;;
	*jobs/9001/logs*) printf 'Running 175 tests using 2 workers\n  166 passed (44.8s)\n' ;;
	*jobs/9002/logs*) [ -n "$LOG_FAILS" ] && exit 1; printf 'Running 175 tests using 1 worker\n  166 passed (1.1m)\n' ;;
	*) echo "gh failed: $*" >&2; exit 1 ;;
esac
"""


def _run(tmp_path: Path, *args: str, runs: str, log_fails: bool = False) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(GH, encoding="utf-8")
    gh.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "RUNS": runs,
           "LOG_FAILS": "1" if log_fails else ""}
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=30)


TWO_RUNS = r"111 2026-11-13 feat/a\n222 2026-11-12 fix/b\n"


def test_prints_each_run_and_the_range(tmp_path):
    out = _run(tmp_path, "2", runs=TWO_RUNS)

    assert out.returncode == 0, out.stdout + out.stderr
    assert "111" in out.stdout and "feat/a" in out.stdout and "44.8s" in out.stdout
    assert "222" in out.stdout and "fix/b" in out.stdout and "1.1m" in out.stdout
    assert "範圍（2 次）：web job 96–150 秒、e2e 步驟 56–76 秒、測試本身 44.8–66.0 秒" in out.stdout
    assert "有紀錄" not in out.stdout


def test_run_whose_jobs_cannot_be_read_is_left_out_of_the_range(tmp_path):
    # 333 的 jobs 查詢失敗：那一列標明查不到，不能變成 0 秒混進範圍
    out = _run(tmp_path, "3", runs=TWO_RUNS + r"333 2026-11-11 fix/c\n")

    assert out.returncode == 0, out.stdout + out.stderr
    assert "查不到" in out.stdout.split("333")[1].splitlines()[0]
    assert "範圍（2 次）：web job 96–150 秒、e2e 步驟 56–76 秒" in out.stdout


def test_missing_test_time_is_counted_in_the_range_line(tmp_path):
    out = _run(tmp_path, "2", runs=TWO_RUNS, log_fails=True)

    assert out.returncode == 0, out.stdout + out.stderr
    assert "測試本身 44.8–44.8 秒（有紀錄 1/2 次）" in out.stdout


def test_no_runs_is_an_error_without_a_fake_range(tmp_path):
    out = _run(tmp_path, runs="")

    assert out.returncode == 1
    assert "範圍" not in out.stdout
    assert "gh failed" not in out.stderr


def test_rejects_a_non_numeric_count(tmp_path):
    out = _run(tmp_path, "abc", runs=TWO_RUNS)

    assert out.returncode == 2
