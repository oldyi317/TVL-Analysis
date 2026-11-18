import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "domain-expiry-issue.sh"
TITLE = "vstat.tw 網域快到期了，請到 PChome 續約"


def _run(tmp_path: Path, open_titles: list[str], list_exit: int = 0) -> tuple[subprocess.CompletedProcess, list[str], str]:
    """以假的 gh 跑腳本；回傳結果、gh issue create 收到的參數（每個一行）與內文。"""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    titles = tmp_path / "titles.txt"
    titles.write_text("".join(t + "\n" for t in open_titles), encoding="utf-8")
    create_args = tmp_path / "create_args.txt"
    create_body = tmp_path / "create_body.txt"
    gh = bin_dir / "gh"
    gh.write_text(
        "#!/usr/bin/env bash\n"
        f'if [ "$1 $2" = "issue list" ]; then cat {titles}; exit {list_exit}; fi\n'
        f'if [ "$1 $2" = "issue create" ]; then printf "%s\\n" "$@" > {create_args}; cat > {create_body}; exit 0; fi\n'
        "exit 99\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "GITHUB_SERVER_URL": "https://github.com",
           "GITHUB_REPOSITORY": "oldyi317/TVL-Analysis", "GITHUB_RUN_ID": "123"}
    result = subprocess.run(["bash", str(SCRIPT), "2027-02-01", "75"], env=env, timeout=30,
                            capture_output=True, text=True)
    args = create_args.read_text(encoding="utf-8").splitlines() if create_args.exists() else []
    body = create_body.read_text(encoding="utf-8") if create_body.exists() else ""
    return result, args, body


def test_opens_issue_with_expiry_date_and_days_left_when_no_same_title_issue_is_open(tmp_path):
    result, args, body = _run(tmp_path, ["別的 issue"])

    assert result.returncode == 0, result.stderr
    assert args[args.index("--title") + 1] == TITLE
    assert args[args.index("--label") + 1] == "ready-for-human"
    assert "**2027-02-01**" in body
    assert "**75 天**" in body
    assert "https://github.com/oldyi317/TVL-Analysis/actions/runs/123" in body


def test_does_not_open_issue_when_same_title_issue_is_still_open(tmp_path):
    result, args, _ = _run(tmp_path, ["別的 issue", TITLE])

    assert result.returncode == 0, result.stderr
    assert args == []


def test_opens_issue_when_open_titles_only_partly_match(tmp_path):
    result, args, _ = _run(tmp_path, [TITLE + "（舊）", "vstat.tw 網域快到期了"])

    assert result.returncode == 0, result.stderr
    assert args[args.index("--title") + 1] == TITLE


def test_exits_non_zero_without_opening_issue_when_issue_list_fails(tmp_path):
    result, args, _ = _run(tmp_path, [], list_exit=1)

    assert result.returncode != 0
    assert args == []
