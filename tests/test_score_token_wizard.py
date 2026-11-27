import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WIZARD = ROOT / "scripts" / "score_token_wizard.sh"

TOKEN = "github_pat_FAKE0123456789abcdef"
EXPIRES = "2027-02-21 00:00:00 UTC"

# 替身 curl：依網址結尾分流，token 由 `-H @-` 從 stdin 進來；argv 與 stdin 各記一份供斷言
FAKE_CURL = r"""#!/usr/bin/env bash
echo "$*" >> "$STUB_DIR/curl_argv.log"
cat >> "$STUB_DIR/curl_stdin.log"
out=/dev/null; hdr=""; url=""; w=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out=$2; shift;; -D) hdr=$2; shift;; -w) w=$2; shift;;
    -H|-X|-d) shift;; -sS) ;; *) url=$1;;
  esac
  shift
done
code=200
case "$url" in
  */actions/runners) code=$FAKE_RUNNERS;;
  */dispatches)
    code=$FAKE_DISPATCH
    if [ -n "$hdr" ]; then
      printf 'HTTP/2 %s\r\n' "$code" > "$hdr"
      [ "$code" = 403 ] && printf 'X-Accepted-GitHub-Permissions: actions=write\r\n' >> "$hdr"
      printf 'github-authentication-token-expiration: %s\r\n\r\n' "$EXPIRES" >> "$hdr"
    fi
    echo "{\"message\": \"fake $code\"}" > "$out";;
  */user) echo '{"login": "someone-else","id": 1}'; code=200;;
esac
[ -n "$w" ] && printf '%s' "$code"
exit 0
"""

# 替身 wrangler：只有 secret put 會讀 stdin
FAKE_WRANGLER = r"""#!/usr/bin/env bash
echo "$*" >> "$STUB_DIR/wrangler_argv.log"
case "$1 $2" in
  "secret put") cat > "$STUB_DIR/wrangler_stdin.log";;
  "secret list") echo '[{"name": "GITHUB_TOKEN", "type": "secret_text"}]';;
esac
exit 0
"""

FAKE_WSLVIEW = '#!/usr/bin/env bash\necho "$*" >> "$STUB_DIR/wslview_argv.log"\n'


def _stub(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _run(tmp_path, *, dispatch=200, runners=403, lines=None, answer="y"):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(WIZARD, scripts / WIZARD.name)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _stub(tmp_path / "score" / "node_modules" / ".bin" / "wrangler", FAKE_WRANGLER)
    bin_dir = tmp_path / "bin"
    _stub(bin_dir / "curl", FAKE_CURL)
    _stub(bin_dir / "wslview", FAKE_WSLVIEW)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STUB_DIR": str(stubs),
        "FAKE_DISPATCH": str(dispatch),
        "FAKE_RUNNERS": str(runners),
        "EXPIRES": EXPIRES,
    }
    # banner、階段 1、階段 2、token、通過後 Enter、是否存入、最後 Enter
    feed = lines if lines is not None else ["", "", "", TOKEN, "", answer, ""]
    result = subprocess.run(
        ["bash", str(scripts / WIZARD.name)],
        input="\n".join(feed) + "\n",
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, stubs


def _log(stubs: Path, name: str) -> str:
    path = stubs / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_dispatch_401_exits_with_status_and_stores_nothing(tmp_path):
    result, stubs = _run(tmp_path, dispatch=401)

    assert result.returncode == 1
    assert "GitHub 回 401" in result.stdout + result.stderr
    assert "secret put" not in _log(stubs, "wrangler_argv.log")


def test_dispatch_403_reports_token_owner_and_required_permission(tmp_path):
    result, stubs = _run(tmp_path, dispatch=403)

    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "someone-else" in output
    # 建立頁網址本身就含 actions=write，所以要鎖在「要求的權限」那一行
    assert "要求的權限：actions=write" in output
    assert "secret put" not in _log(stubs, "wrangler_argv.log")


def test_token_with_administration_permission_is_rejected_before_dispatch(tmp_path):
    result, stubs = _run(tmp_path, runners=200)

    assert result.returncode == 1
    assert "Administration" in result.stdout + result.stderr
    assert "/dispatches" not in _log(stubs, "curl_argv.log")
    assert "secret put" not in _log(stubs, "wrangler_argv.log")


def test_declining_to_store_exits_without_secret_put(tmp_path):
    result, stubs = _run(tmp_path, answer="n")

    assert result.returncode == 1
    assert "secret put" not in _log(stubs, "wrangler_argv.log")


def test_confirmed_token_is_piped_to_wrangler_and_expiry_is_summarised(tmp_path):
    result, stubs = _run(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "secret put GITHUB_TOKEN" in _log(stubs, "wrangler_argv.log")
    assert _log(stubs, "wrangler_stdin.log") == TOKEN
    assert EXPIRES in result.stdout


def test_token_never_appears_in_any_argv_or_in_script_output(tmp_path):
    result, stubs = _run(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("curl_argv.log", "wrangler_argv.log", "wslview_argv.log"):
        assert TOKEN not in _log(stubs, name)
    assert TOKEN not in result.stdout
    assert TOKEN not in result.stderr
    # 對照組：token 確實有從 stdin 送到 curl，否則上面的斷言什麼也沒鎖住
    assert TOKEN in _log(stubs, "curl_stdin.log")


def test_input_not_starting_with_github_pat_is_asked_again(tmp_path):
    lines = ["", "", "", "ghp_wrongkind", TOKEN, "", "y", ""]
    result, stubs = _run(tmp_path, lines=lines)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "開頭不是 github_pat_" in result.stdout + result.stderr
    assert _log(stubs, "wrangler_stdin.log") == TOKEN


def test_creation_page_url_preselects_owner_and_actions_write(tmp_path):
    _, stubs = _run(tmp_path)

    opened = _log(stubs, "wslview_argv.log")
    assert "target_name=oldyi317" in opened
    assert "actions=write" in opened
