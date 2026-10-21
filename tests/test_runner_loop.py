import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT / "scripts" / "runner_loop.sh"


def _stub(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body + "\n", encoding="utf-8")
    path.chmod(0o755)


def _run_loop(tmp_path: Path, gh_body: str, max_sleeps: int, min_job_seconds: int = 0) -> tuple[list[str], list[str]]:
    """以假的 gh/config.sh/run.sh/sleep 跑迴圈，sleep 呼叫滿 max_sleeps 次即結束迴圈。"""
    runner_dir = tmp_path / "actions-runner"
    bin_dir = tmp_path / "bin"
    runner_dir.mkdir()
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    sleeps = tmp_path / "sleeps.log"

    token_file = tmp_path / "token"
    token_file.write_text("pat-xyz\n", encoding="utf-8")
    _stub(bin_dir / "gh", '[ "$GH_TOKEN" = pat-xyz ] || exit 1\n' + gh_body)
    _stub(runner_dir / "config.sh",
          f'[ -f .runner ] && {{ echo already configured; exit 1; }}\n'
          f'echo "config $*" >> {log}; touch .runner')
    _stub(runner_dir / "run.sh",
          f'[ -n "${{GH_TOKEN:-}}" ] && echo "run leaked" >> {log}\n'
          f'[ -d _work ] && echo "run dirty" >> {log} || echo run >> {log}\n'
          f'mkdir -p _work; rm -f .runner')
    _stub(bin_dir / "sleep",
          f'echo "$1" >> {sleeps}\n'
          f'[ "$(wc -l < {sleeps})" -ge {max_sleeps} ] && kill -TERM $PPID\nexit 0')

    # 上一輪異常中斷留下的本機設定與工作目錄
    (runner_dir / ".runner").write_text("{}", encoding="utf-8")
    (runner_dir / "_work").mkdir()

    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "RUNNER_DIR": str(runner_dir),
           "RUNNER_MIN_JOB_SECONDS": str(min_job_seconds), "RUNNER_TOKEN_FILE": str(token_file)}
    env.pop("GH_TOKEN", None)
    subprocess.run(["bash", str(LOOP)], env=env, timeout=30, capture_output=True)

    def lines(p: Path) -> list[str]:
        return p.read_text(encoding="utf-8").splitlines() if p.exists() else []

    return lines(log), lines(sleeps)


def test_each_round_registers_ephemeral_and_runs_one_job_on_clean_dir(tmp_path):
    calls, _ = _run_loop(tmp_path, "echo TOKEN123", max_sleeps=2)

    assert calls[1::2] == ["run", "run"]
    for config in calls[0::2]:
        args = config.split()
        assert "--ephemeral" in args and "--replace" in args and "--unattended" in args
        assert args[args.index("--token") + 1] == "TOKEN123"
        assert args[args.index("--labels") + 1] == "tvl"
        assert args[args.index("--name") + 1] == "tvl-wsl"


def test_failed_token_fetch_backs_off_without_registering(tmp_path):
    calls, sleeps = _run_loop(tmp_path, "exit 1", max_sleeps=7)

    assert calls == []
    assert sleeps == ["30", "60", "120", "240", "480", "900", "900"]


def test_listener_exiting_immediately_backs_off_instead_of_reregistering(tmp_path):
    # run.sh 對 listener 失敗也回 0，只能靠「太快結束」判斷異常
    calls, sleeps = _run_loop(tmp_path, "echo TOKEN123", max_sleeps=3, min_job_seconds=60)

    assert len(calls) == 6
    assert sleeps == ["30", "60", "120"]
