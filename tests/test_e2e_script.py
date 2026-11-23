import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_A = "  跳過：甲測試 -- 原因一"
SKIP_B = "  跳過：乙測試 -- 原因二"


def _stub(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body + "\n", encoding="utf-8")
    path.chmod(0o755)


def _run_e2e(tmp_path: Path, npx_body: str, *args: str) -> subprocess.CompletedProcess:
    """把 e2e.sh 與它呼叫的檔案複製到 tmp_path，npx 換成假的再跑。"""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("e2e.sh", "busy-warning.sh", "e2e_profile.py", "isolated-net.sh"):
        shutil.copy(ROOT / "scripts" / name, scripts / name)
    (tmp_path / "web" / "build").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir / "npx", npx_body)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PW_PORT": "4999",
        "TMPDIR": str(tmp_path),
        "TVL_PYTHON": sys.executable,
    }
    return subprocess.run(["bash", str(scripts / "e2e.sh"), *args], env=env,
                          capture_output=True, text=True, timeout=60)


def test_repeat_prints_skip_reasons_once_then_only_differences(tmp_path):
    # 第 2 輪順序不同但內容相同，不重印；第 3 輪少了乙，只印差異
    npx = f"""
n=$(( $(cat {tmp_path}/n 2>/dev/null || echo 0) + 1 )); echo $n > {tmp_path}/n
case $n in
	1) printf '%s\\n' '{SKIP_A}' '{SKIP_B}' ;;
	2) printf '%s\\n' '{SKIP_B}' '{SKIP_A}' ;;
	3) printf '%s\\n' '{SKIP_A}' ;;
esac
echo
echo "  2 skipped"
echo "  5 passed (1.0s)"
"""
    out = _run_e2e(tmp_path, npx, "--repeat", "3")

    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.count("跳過：甲測試") == 1
    assert out.stdout.count("跳過：乙測試") == 2
    assert "少了：跳過：乙測試 -- 原因二" in out.stdout
    assert "3 輪全過" in out.stdout


REPORT = {
    "config": {"metadata": {"actualWorkers": 8}},
    "stats": {"duration": 106401.6, "expected": 2, "skipped": 1, "unexpected": 0, "flaky": 0},
    "suites": [
        {"file": "leaders.e2e.ts", "specs": [], "suites": [
            {"file": "leaders.e2e.ts", "specs": [
                {"line": 8, "title": "七項跳轉", "tests": [{"results": [{"duration": 8400}]}]},
            ]},
        ]},
        {"file": "teams.e2e.ts", "specs": [
            {"line": 140, "title": "隊徽載入成功", "tests": [{"results": [{"duration": 13000}]}]},
            {"line": 330, "title": "空狀態", "tests": [{"results": [{"duration": 0}]}]},
        ]},
    ],
}


def _profile_npx(exit_code: int) -> str:
    return f"""
[[ "$*" == *--reporter=*json* ]] || exit 3
cat > "$PLAYWRIGHT_JSON_OUTPUT_FILE" <<'JSON'
{json.dumps(REPORT, ensure_ascii=False)}
JSON
exit {exit_code}
"""


def test_profile_prints_workers_counts_slowest_and_per_file(tmp_path):
    out = _run_e2e(tmp_path, _profile_npx(0), "--profile")

    assert out.returncode == 0, out.stdout + out.stderr
    assert "實際 worker 數：8" in out.stdout
    assert "測試 3 個：2 過、1 略過、0 失敗、0 flaky" in out.stdout
    slowest = out.stdout.split("最慢的測試")[1]
    assert slowest.index("teams.e2e.ts:140 隊徽載入成功") < slowest.index("leaders.e2e.ts:8 七項跳轉")
    per_file = out.stdout.split("逐檔")[1].split("最慢的測試")[0]
    assert per_file.index("teams.e2e.ts") < per_file.index("leaders.e2e.ts")


def test_profile_still_summarises_and_fails_when_tests_fail(tmp_path):
    out = _run_e2e(tmp_path, _profile_npx(1), "--profile")

    assert out.returncode == 1
    assert "實際 worker 數：8" in out.stdout


def test_repeat_and_profile_together_is_a_usage_error(tmp_path):
    out = _run_e2e(tmp_path, "exit 0", "--repeat", "2", "--profile")

    assert out.returncode == 2


def _interfaces(text: str) -> set[str]:
    return {line.split()[1].rstrip(":") for line in text.splitlines() if line[:1].isdigit()}


def test_playwright_runs_where_host_address_changes_cannot_reach_it(tmp_path):
    # #304：主機網卡的位址一變（Wi-Fi 重連），Chromium 會中斷進行中的請求，連 localhost 也是。
    # Playwright 要跑在看不到主機網卡的 network namespace 裡，且裡面要有一張有位址的非 loopback 網卡，否則瀏覽器判成離線
    probe = "ip link set lo up && ip link add probe0 type dummy"
    if subprocess.run(["unshare", "-Urn", "sh", "-c", probe], capture_output=True).returncode != 0:
        pytest.skip("這個環境開不了 network namespace，e2e.sh 會照原樣跑")
    host = _interfaces(subprocess.run(["ip", "-o", "-4", "addr"], capture_output=True, text=True).stdout) - {"lo"}

    out = _run_e2e(tmp_path, 'ip -o -4 addr\necho "  1 passed (1.0s)"')

    assert out.returncode == 0, out.stdout + out.stderr
    seen = _interfaces(out.stdout) - {"lo"}
    assert seen, out.stdout
    assert not seen & host, out.stdout
