import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts" / "setup_crawl_trigger.sh"


def _stub(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body + "\n", encoding="utf-8")
    path.chmod(0o755)


def _parse_unit(path: Path) -> dict[str, str]:
    pairs = (line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if "=" in line)
    return {key: value for key, value in pairs}


def test_installs_enabled_user_timer_that_dispatches_daily_crawl_at_0923_taipei(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "systemctl.log"
    _stub(bin_dir / "systemctl", f'echo "$*" >> {calls}')
    _stub(bin_dir / "gh", "exit 0")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "XDG_CONFIG_HOME": str(tmp_path / "config")}

    subprocess.run(["bash", str(SETUP)], env=env, check=True, timeout=30, capture_output=True)

    unit_dir = tmp_path / "config" / "systemd" / "user"
    timer = _parse_unit(unit_dir / "tvl-crawl-trigger.timer")
    assert timer["OnCalendar"] == "*-*-* 09:23:00 Asia/Taipei"
    assert timer["Persistent"] == "true"

    service = _parse_unit(unit_dir / "tvl-crawl-trigger.service")
    # 開機補跑時網路可能還沒通，觸發失敗要重試，否則當天只剩 GitHub 晚 4–7 小時的備援
    assert service["Restart"] == "on-failure"
    argv = shlex.split(service["ExecStart"])
    assert Path(argv[0]).name == "gh"
    assert argv[1:4] == ["workflow", "run", "daily-crawl.yml"]
    assert argv[argv.index("--repo") + 1] == "oldyi317/TVL-Analysis"
    assert argv[argv.index("--ref") + 1] == "main"

    assert "--user enable --now tvl-crawl-trigger.timer" in calls.read_text(encoding="utf-8").splitlines()
