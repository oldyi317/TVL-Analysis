import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT / "scripts" / "runner_loop.sh"
GATE = ROOT / "scripts" / "runner_job_gate.sh"


def _stub(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body + "\n", encoding="utf-8")
    path.chmod(0o755)


def _run_loop(tmp_path: Path, gh_body: str, max_sleeps: int, min_job_seconds: int = 0,
              gate: bool = True) -> tuple[list[str], list[str]]:
    """以假的 gh/config.sh/run.sh/sleep 跑迴圈，sleep 呼叫滿 max_sleeps 次即結束迴圈。"""
    runner_dir = tmp_path / "actions-runner"
    bin_dir = tmp_path / "bin"
    runner_dir.mkdir()
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    sleeps = tmp_path / "sleeps.log"

    token_file = tmp_path / "token"
    token_file.write_text("pat-xyz\n", encoding="utf-8")
    gate_file = tmp_path / "runner_job_gate.sh"
    if gate:
        gate_file.write_text("exit 0\n", encoding="utf-8")
    _stub(bin_dir / "gh", '[ "$GH_TOKEN" = pat-xyz ] || exit 1\n' + gh_body)
    _stub(runner_dir / "config.sh",
          f'[ -f .runner ] && {{ echo already configured; exit 1; }}\n'
          f'echo "config $*" >> {log}; touch .runner')
    _stub(runner_dir / "run.sh",
          f'[ -n "${{GH_TOKEN:-}}" ] && echo "run leaked" >> {log}\n'
          f'echo "${{ACTIONS_RUNNER_HOOK_JOB_STARTED:-}}" >> {tmp_path / "hooks.log"}\n'
          f'[ -d _work ] && echo "run dirty" >> {log} || echo run >> {log}\n'
          f'mkdir -p _work; rm -f .runner')
    _stub(bin_dir / "sleep",
          f'echo "$1" >> {sleeps}\n'
          f'[ "$(wc -l < {sleeps})" -ge {max_sleeps} ] && kill -TERM $PPID\nexit 0')

    # 上一輪異常中斷留下的本機設定與工作目錄
    (runner_dir / ".runner").write_text("{}", encoding="utf-8")
    (runner_dir / "_work").mkdir()

    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "RUNNER_DIR": str(runner_dir),
           "RUNNER_MIN_JOB_SECONDS": str(min_job_seconds), "RUNNER_TOKEN_FILE": str(token_file),
           "RUNNER_JOB_GATE": str(gate_file)}
    env.pop("GH_TOKEN", None)
    env.pop("ACTIONS_RUNNER_HOOK_JOB_STARTED", None)
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
    hooks = (tmp_path / "hooks.log").read_text(encoding="utf-8").splitlines()
    assert hooks == [str(tmp_path / "runner_job_gate.sh")] * 2


def test_failed_token_fetch_backs_off_without_registering(tmp_path):
    calls, sleeps = _run_loop(tmp_path, "exit 1", max_sleeps=7)

    assert calls == []
    assert sleeps == ["30", "60", "120", "240", "480", "900", "900"]


def test_listener_exiting_immediately_backs_off_instead_of_reregistering(tmp_path):
    # run.sh 對 listener 失敗也回 0，只能靠「太快結束」判斷異常
    calls, sleeps = _run_loop(tmp_path, "echo TOKEN123", max_sleeps=3, min_job_seconds=60)

    assert len(calls) == 6
    assert sleeps == ["30", "60", "120"]


def test_missing_job_gate_backs_off_without_fetching_token_or_registering(tmp_path):
    calls, sleeps = _run_loop(tmp_path, f"touch {tmp_path / 'gh_called'}; echo TOKEN123", max_sleeps=2, gate=False)

    assert calls == []
    assert sleeps == ["30", "60"]
    assert not (tmp_path / "gh_called").exists()


def _run_gate(tmp_path: Path, event: str, repo: str = "oldyi317/TVL-Analysis",
              ref: str = "refs/heads/main", parent: str | None = None) -> tuple[subprocess.CompletedProcess, list[str]]:
    """跑守門腳本；pkill 換成記錄參數的假程式，測試不會砍到本機真的 Runner.Worker。"""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pkill_log = tmp_path / "pkill.log"
    _stub(bin_dir / "pkill", f'echo "$*" >> {pkill_log}')
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "GITHUB_EVENT_NAME": event, "GITHUB_REPOSITORY": repo, "GITHUB_REF": ref}
    cmd = ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", str(GATE)]
    if parent:
        cmd = [parent, "-c", f'{shlex.join(cmd)}; echo survived']
    result = subprocess.run(cmd, env=env, timeout=30, capture_output=True, text=True)
    pkill = pkill_log.read_text(encoding="utf-8").splitlines() if pkill_log.exists() else []
    return result, pkill


@pytest.mark.parametrize("event, ref", [
    ("schedule", "refs/heads/main"),
    ("workflow_dispatch", "refs/heads/main"),
    ("workflow_dispatch", "refs/heads/feat/issue-1-x"),
])
def test_job_gate_allows_schedule_and_dispatch_on_this_repo(tmp_path, event, ref):
    result, pkill = _run_gate(tmp_path, event, ref=ref)

    assert result.returncode == 0
    assert pkill == []


@pytest.mark.parametrize("event, repo", [
    ("pull_request", "oldyi317/TVL-Analysis"),
    ("pull_request_target", "oldyi317/TVL-Analysis"),
    ("push", "oldyi317/TVL-Analysis"),
    ("workflow_run", "oldyi317/TVL-Analysis"),
    ("issue_comment", "oldyi317/TVL-Analysis"),
    ("", "oldyi317/TVL-Analysis"),
    ("schedule", "someone/TVL-Analysis"),
    ("workflow_dispatch", ""),
])
def test_job_gate_rejects_other_events_and_repos_by_killing_worker(tmp_path, event, repo):
    result, pkill = _run_gate(tmp_path, event, repo=repo)

    assert result.returncode == 1
    assert pkill == [f"-KILL -u {os.geteuid()} -x Runner.Worker"]


def test_job_gate_kills_its_parent_worker_before_any_step_runs(tmp_path):
    worker = tmp_path / "Runner.Worker"
    shutil.copy(shutil.which("bash"), worker)

    result, pkill = _run_gate(tmp_path, "pull_request", parent=str(worker))

    assert result.returncode == -9
    assert "survived" not in result.stdout
    assert pkill == []
