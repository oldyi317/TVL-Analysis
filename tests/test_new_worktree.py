import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "new-worktree.sh"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def env(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, repo / "scripts" / "new-worktree.sh")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "user.email", "t@example.com")
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    home = tmp_path / "home"
    home.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    npm_log = tmp_path / "npm.log"
    npm = bin_dir / "npm"
    npm.write_text(f'#!/usr/bin/env bash\necho "$*" >> {npm_log}\nexit 1\n', encoding="utf-8")
    npm.chmod(0o755)
    return {
        "repo": repo,
        "home": home,
        "npm_log": npm_log,
        "env": {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "TVL_WORKTREE_HOME": str(home),
        },
    }


def _add_worktree(e, branch: str) -> Path:
    path = e["home"] / f"tvl-{branch.rsplit('/', 1)[-1]}"
    _git(e["repo"], "worktree", "add", "-q", "-b", branch, str(path), "main")
    return path


def _run(e, *args: str):
    return subprocess.run(
        ["bash", str(e["repo"] / "scripts" / "new-worktree.sh"), *args],
        env=e["env"], capture_output=True, text=True, timeout=30,
    )


def test_resume_shows_uncommitted_changes_and_commits_ahead_of_base(env):
    wt = _add_worktree(env, "chore/retro-1-x")
    (wt / "b.txt").write_text("b\n", encoding="utf-8")
    _git(wt, "add", "b.txt")
    _git(wt, "commit", "-q", "-m", "ahead commit title")
    (wt / "dirty.txt").write_text("d\n", encoding="utf-8")

    r = _run(env, "chore/retro-1-x", "main")

    assert r.returncode == 0, r.stderr
    assert "dirty.txt" in r.stdout
    assert "ahead commit title" in r.stdout


def test_resume_clean_tree_with_nothing_ahead(env):
    _add_worktree(env, "chore/retro-1-x")

    r = _run(env, "chore/retro-1-x", "main")

    assert r.returncode == 0, r.stderr
    assert "工作樹乾淨" in r.stdout
    assert "沒有領先" in r.stdout


def test_resume_prints_notes_dir_files_and_state_md(env):
    _add_worktree(env, "feat/issue-123-foo")
    notes = env["home"] / "tvl-issue-123-notes"
    notes.mkdir()
    (notes / "STATE.md").write_text("deployed to step 3\n", encoding="utf-8")
    (notes / "other.log").write_text("x\n", encoding="utf-8")

    r = _run(env, "feat/issue-123-foo", "main")

    assert r.returncode == 0, r.stderr
    assert "deployed to step 3" in r.stdout
    assert "other.log" in r.stdout


def test_resume_reports_missing_notes_dir(env):
    _add_worktree(env, "feat/issue-123-foo")

    r = _run(env, "feat/issue-123-foo", "main")

    assert r.returncode == 0, r.stderr
    assert "沒有筆記目錄" in r.stdout


def test_resume_skips_notes_when_branch_has_no_issue_number(env):
    _add_worktree(env, "chore/retro-1-x")

    r = _run(env, "chore/retro-1-x", "main")

    assert r.returncode == 0, r.stderr
    assert "筆記目錄" not in r.stdout


def test_plain_directory_is_refused(env):
    (env["home"] / "tvl-retro-1-x").mkdir()

    r = _run(env, "chore/retro-1-x", "main")

    assert r.returncode == 1
    assert "不覆蓋" in r.stderr


def test_plain_directory_inside_a_repo_on_the_same_branch_is_refused(env):
    # git -C 會往上找到外層 repo，分支剛好同名時不能當成接手
    home = env["repo"] / "nested"
    (home / "tvl-main").mkdir(parents=True)
    result = subprocess.run(
        ["bash", str(env["repo"] / "scripts" / "new-worktree.sh"), "main", "main"],
        env={**env["env"], "TVL_WORKTREE_HOME": str(home)}, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    assert "不覆蓋" in result.stderr
    assert "目前分支" not in result.stderr


def test_worktree_of_other_branch_is_refused_with_its_branch_name(env):
    path = env["home"] / "tvl-retro-1-x"
    _git(env["repo"], "worktree", "add", "-q", "-b", "other/branch-zz", str(path), "main")

    r = _run(env, "chore/retro-1-x", "main")

    assert r.returncode == 1
    assert "不覆蓋" in r.stderr
    assert "other/branch-zz" in r.stderr


def test_resume_with_missing_base_is_not_an_error(env):
    _add_worktree(env, "chore/retro-1-x")

    r = _run(env, "chore/retro-1-x", "no-such-base")

    assert r.returncode == 0, r.stderr
    assert "no-such-base" in r.stdout
    assert "fatal" not in r.stdout + r.stderr


def test_resume_does_not_fetch_or_install(env):
    _add_worktree(env, "chore/retro-1-x")
    _git(env["repo"], "remote", "add", "origin", str(env["home"] / "no-such-remote"))

    r = _run(env, "chore/retro-1-x", "main")

    assert r.returncode == 0, r.stderr
    assert "no-such-remote" not in r.stdout + r.stderr
    assert not env["npm_log"].exists()
