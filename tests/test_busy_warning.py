import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "busy-warning.sh"


def _run(tmp_path: Path, runner_running: bool) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pgrep = bin_dir / "pgrep"
    hit = "exit 0" if runner_running else "exit 1"
    pgrep.write_text(f'#!/usr/bin/env bash\n[ "$*" = "-x Runner.Worker" ] && {hit}\nexit 1\n',
                     encoding="utf-8")
    pgrep.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30)


# CPU 那一行看當下機器忙不忙，測試只斷言 runner 那一行
def test_warns_when_local_runner_is_running_a_job(tmp_path):
    out = _run(tmp_path, runner_running=True)

    assert out.returncode == 0
    assert "本機 runner 正在跑 job" in out.stdout


def test_no_runner_line_when_runner_is_idle(tmp_path):
    out = _run(tmp_path, runner_running=False)

    assert out.returncode == 0
    assert "runner" not in out.stdout
