import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "daily_update.sh"
DB = "data/db/tvl_database.db"

# 替身 Python：帶 -m 的呼叫記下模組名與當下的 UTC 偏移；不帶參數的呼叫是腳本在算資料庫雜湊
FAKE_PYTHON = """#!/usr/bin/env bash
if [ $# -eq 0 ]; then
    sha256sum data/db/tvl_database.db | cut -d' ' -f1
    exit 0
fi
echo "$(date +%z) $2" >> "$CALLS"
if [ "$2" = "${FAKE_CHANGES_DB_ON:-}" ]; then echo changed >> data/db/tvl_database.db; fi
if [ "$2" = "${FAKE_FAILS_ON:-}" ]; then exit 1; fi
"""

FULL_UPDATE_STEPS = [
    "src.etl.schedule_crawler",
    "src.etl.crawler",
    "src.etl.db_loader",
    "src.etl.match_crawler",
    "src.etl.match_stats_writer",
    "src.models.predict",
]


def _executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def repo(tmp_path):
    """臨時 git repo（含 bare origin）裡放一份腳本與假資料庫，git 是真的，Python 與 gh 是替身。"""
    work = tmp_path / "work"
    origin = tmp_path / "origin.git"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(bin_dir / "python", FAKE_PYTHON)
    _executable(bin_dir / "gh", f'#!/usr/bin/env bash\necho "$*" >> {tmp_path / "gh.log"}\n')

    env = {key: value for key, value in os.environ.items() if key not in {"GH_TOKEN", "DRY_RUN", "TZ"}}
    env.update(
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        PYTHON=str(bin_dir / "python"),
        CALLS=str(tmp_path / "calls.log"),
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
    )

    def git(*args, cwd=work):
        result = subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True)
        return result.stdout.strip()

    (work / "scripts").mkdir(parents=True)
    (work / "scripts" / "daily_update.sh").write_bytes(SCRIPT.read_bytes())
    (work / DB).parent.mkdir(parents=True)
    (work / DB).write_text("original\n", encoding="utf-8")
    git("init", "--bare", "-b", "main", str(origin), cwd=tmp_path)
    git("init", "-b", "main")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-m", "init")
    git("remote", "add", "origin", str(origin))
    git("push", "-u", "origin", "main")

    def run(**extra_env):
        return subprocess.run(
            ["bash", "scripts/daily_update.sh"],
            cwd=work, env={**env, **extra_env}, capture_output=True, text=True, timeout=60,
        )

    def calls():
        log = tmp_path / "calls.log"
        lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return [tuple(line.split(" ", 1)) for line in lines]

    return SimpleNamespace(work=work, origin=origin, git=git, run=run, calls=calls, gh_log=tmp_path / "gh.log")


def test_full_update_runs_schedule_roster_scores_box_scores_predictions_in_order(repo):
    result = repo.run()

    assert result.returncode == 0, result.stderr
    assert [module for _, module in repo.calls()] == FULL_UPDATE_STEPS


def test_steps_see_taipei_time_when_system_timezone_is_utc(repo):
    result = repo.run(TZ="UTC")

    assert result.returncode == 0, result.stderr
    assert [offset for offset, _ in repo.calls()] == ["+0800"] * len(FULL_UPDATE_STEPS)


def test_changed_db_is_committed_pushed_and_triggers_site_build(repo):
    result = repo.run(FAKE_CHANGES_DB_ON="src.etl.match_crawler", GH_TOKEN="token")

    assert result.returncode == 0, result.stderr
    assert repo.git("rev-list", "--count", "HEAD") == "2"
    assert repo.git("show", "--name-only", "--format=", "HEAD") == DB
    assert repo.git("rev-parse", "main", cwd=repo.origin) == repo.git("rev-parse", "HEAD")
    assert repo.gh_log.read_text(encoding="utf-8").splitlines() == ["workflow run pages.yml --ref main"]


def _assert_nothing_committed(repo):
    assert repo.git("rev-list", "--count", "HEAD") == "1"
    assert repo.git("rev-list", "--count", "main", cwd=repo.origin) == "1"
    assert not repo.gh_log.exists()


def test_score_step_failure_fails_the_run_without_commit(repo):
    # 前面的步驟已經改了資料庫：少了「失敗就停」，這些變更會被 commit
    result = repo.run(
        FAKE_CHANGES_DB_ON="src.etl.schedule_crawler", FAKE_FAILS_ON="src.etl.match_crawler", GH_TOKEN="token"
    )

    assert result.returncode != 0
    _assert_nothing_committed(repo)


def test_dry_run_with_changes_restores_db_without_commit(repo):
    result = repo.run(FAKE_CHANGES_DB_ON="src.etl.match_crawler", DRY_RUN="1", GH_TOKEN="token")

    assert result.returncode == 0, result.stderr
    _assert_nothing_committed(repo)
    assert (repo.work / DB).read_text(encoding="utf-8") == "original\n"
    assert repo.git("status", "--porcelain") == ""


def test_unchanged_db_ends_without_commit(repo):
    result = repo.run(GH_TOKEN="token")

    assert result.returncode == 0, result.stderr
    _assert_nothing_committed(repo)
