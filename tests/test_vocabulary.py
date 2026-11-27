import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile(r"[一-鿿]")
TERM = re.compile(r"^\*\*(.+?)\*\*:\s*$")
AVOID = re.compile(r"^_Avoid_:\s*(.+)$")
HITS_PATH = os.environ.get("VOCAB_HITS_PATH")
# 子字串過寬、自然用法多（CONTEXT.md 自身定義亦用「各局比分」），不納入檢查
GENERIC = {"名單", "比賽", "比分", "性別"}
# 行尾含此標記＝該行引用官網原始字串，豁免（Python「# 官網原字」、TS/Svelte「// 官網原字」或「<!-- 官網原字 -->」）
# 避免詞「成功率」限定攻擊以外使用，「攻擊成功率」是正當用法
ALLOWED = ["攻擊成功率"]
RAW_MARK = "官網原字"
INLINE_CODE = re.compile(r"`[^`]*`")


def parse_glossary():
    formal, avoid = [], {}
    term = None
    for line in (ROOT / "CONTEXT.md").read_text(encoding="utf-8").splitlines():
        if m := TERM.match(line):
            term = m.group(1)
            formal.append(term)
        elif (m := AVOID.match(line)) and term:
            words = [re.sub(r"（.*?）", "", w).strip() for w in re.split(r"[,，、]", m.group(1))]
            avoid.update({w: term for w in words if CJK.search(w) and w not in GENERIC})
    return formal, avoid


def english_avoid_terms():
    words = []
    for line in (ROOT / "CONTEXT.md").read_text(encoding="utf-8").splitlines():
        if m := AVOID.match(line):
            words += [re.sub(r"（.*?）", "", w).strip() for w in re.split(r"[,，、]", m.group(1))]
    return [w for w in words if w and not CJK.search(w)]


# 全部英文詞都查時，README 與 docs 新增 106 筆命中、只有 2 筆成立：全小寫與全大寫的詞同時是
# 識別字、git 動作或通用縮寫。只查首字母大寫且含小寫的專名形（Worker、Box Score），照原大小寫比對。
def proper_noun_avoid_terms():
    return list(dict.fromkeys(w for w in english_avoid_terms() if w[0].isupper() and w != w.upper()))


def source_files():
    yield from (ROOT / "src").rglob("*.py")
    for pattern in ("*.ts", "*.svelte"):
        for path in (ROOT / "web" / "src").rglob(pattern):
            if "lib/data" not in path.as_posix():
                yield path


def find_hits(line, formal, avoid):
    hits = []
    for word in avoid:
        rest = line
        shadows = formal + ALLOWED + [w for w in avoid if len(w) > len(word)]
        for s in sorted(shadows, key=len, reverse=True):
            rest = rest.replace(s, "")
        if word in rest:
            hits.append(word)
    return hits


def find_english_hits(line, english):
    prose = INLINE_CODE.sub("", line)
    return [w for w in english if re.search(rf"(?<![A-Za-z0-9_]){re.escape(w)}(?![A-Za-z0-9_])", prose)]


def is_fence(line):
    marker = line.strip()[:3]
    return marker in ("```", "~~~") and marker not in line.strip()[3:]


def scan_text(label, text, formal, avoid, english=()):
    results = []
    in_fence = False
    for no, line in enumerate(text.splitlines(), 1):
        if AVOID.match(line):
            continue
        if english and is_fence(line):
            in_fence = not in_fence
        if not CJK.search(line) or line.rstrip().endswith(RAW_MARK) or line.rstrip().endswith(RAW_MARK + " -->"):
            continue
        words = find_hits(line, formal, avoid)
        if not in_fence:
            words += find_english_hits(line, english)
        results += [(label, no, word, line.strip()) for word in words]
    return results


def scan():
    formal, avoid = parse_glossary()
    results = []
    for path in sorted(source_files()):
        results += scan_text(path.relative_to(ROOT).as_posix(), path.read_text(encoding="utf-8"), formal, avoid)
    return results


def main(argv):
    if not argv:
        print("用法：python tests/test_vocabulary.py <檔案>…（文案、spec、票的草稿）", file=sys.stderr)
        return 2
    formal, avoid = parse_glossary()
    english = proper_noun_avoid_terms()
    results = []
    for arg in argv:
        results += scan_text(arg, Path(arg).read_text(encoding="utf-8"), formal, avoid, english)
    for f, n, w, t in results:
        print(f"{f}:{n}\t{w}\t{t}")
    return 1 if results else 0


def test_glossary_parsed():
    formal, avoid = parse_glossary()
    assert formal and avoid


def test_no_avoided_terms():
    results = scan()
    if HITS_PATH:
        Path(HITS_PATH).write_text(
            "\n".join(f"{f}:{n}\t{w}\t{t}" for f, n, w, t in results) + "\n", encoding="utf-8"
        )
    detail = "\n".join(f"{f}:{n}\t{w}\t{t}" for f, n, w, t in results)
    assert not results, f"命中 {len(results)} 條避免詞：\n{detail}"


def test_scan_text_reports_avoided_term_with_line_number():
    formal, avoid = parse_glossary()
    word = next(iter(avoid))
    hits = scan_text("草稿", f"{formal[0]}\n這一行用了{word}\n", formal, avoid)
    assert ("草稿", 2, word, f"這一行用了{word}") in hits
    assert all(no == 2 for _, no, _, _ in hits)


def test_scan_text_skips_avoid_lines():
    formal, avoid = parse_glossary()
    word = next(iter(avoid))
    prose = f"這一行用了{word}"
    hits = scan_text("草稿", f"_Avoid_: {word}\n{prose}", formal, avoid)
    assert hits == [("草稿", 2, word, prose)]


def test_main_exit_code_follows_hits(tmp_path, capsys):
    formal, avoid = parse_glossary()
    word = next(iter(avoid))
    clean = tmp_path / "clean.md"
    clean.write_text(f"{formal[0]}\n", encoding="utf-8")
    dirty = tmp_path / "dirty.md"
    dirty.write_text(f"這一行用了{word}\n", encoding="utf-8")
    assert main([str(clean)]) == 0
    assert main([str(clean), str(dirty)]) == 1
    assert f"{dirty}:1\t{word}" in capsys.readouterr().out


def draft_hits(tmp_path, capsys, text):
    f = tmp_path / "draft.md"
    f.write_text(text, encoding="utf-8")
    code = main([str(f)])
    return code, capsys.readouterr().out


def test_proper_noun_terms_come_from_glossary():
    english = proper_noun_avoid_terms()
    assert set(english) <= set(english_avoid_terms())
    assert {"Worker", "Box Score"} <= set(english)
    assert all(w[0].isupper() and w != w.upper() for w in english)


def test_draft_flags_english_avoid_term(tmp_path, capsys):
    code, out = draft_hits(tmp_path, capsys, "改用 Worker 觸發\n")
    assert code == 1
    assert f"draft.md:1\tWorker\t改用 Worker 觸發" in out


def test_draft_allows_cloudflare_workers_product_name(tmp_path, capsys):
    code, out = draft_hits(tmp_path, capsys, "部署在 Cloudflare Workers\n")
    assert (code, out) == (0, "")


def test_draft_ignores_inline_code(tmp_path, capsys):
    code, out = draft_hits(tmp_path, capsys, "程式在 `push/` 目錄\n欄位 `cup_id`\n")
    assert (code, out) == (0, "")
    code, out = draft_hits(tmp_path, capsys, "欄位 `cup_id` 改用 Worker 觸發\n")
    assert code == 1 and "Worker" in out and "cup" not in out.split("\t")[1]


def test_draft_ignores_fenced_code_block(tmp_path, capsys):
    text = "```\n這行有 Worker\n```\n之後這行有 Worker\n"
    code, out = draft_hits(tmp_path, capsys, text)
    assert code == 1
    assert out.count("\n") == 1 and "draft.md:4\tWorker" in out


def test_draft_still_checks_chinese_avoid_terms_inside_fenced_code_block(tmp_path, capsys):
    formal, avoid = parse_glossary()
    word = next(iter(avoid))
    code, out = draft_hits(tmp_path, capsys, f"```\n這一行用了{word}\n```\n")
    assert code == 1
    assert f"draft.md:2\t{word}" in out


def test_draft_single_line_triple_backticks_do_not_open_a_fence(tmp_path, capsys):
    code, out = draft_hits(tmp_path, capsys, "```單行程式碼```\n之後這行有 Worker\n")
    assert code == 1
    assert "draft.md:2\tWorker" in out


def test_draft_ignores_tilde_fenced_code_block(tmp_path, capsys):
    code, out = draft_hits(tmp_path, capsys, "~~~\n這行有 Worker\n~~~\n之後這行有 Worker\n")
    assert code == 1
    assert out.count("\n") == 1 and "draft.md:4\tWorker" in out


def test_draft_matches_proper_noun_case_sensitively_on_whole_words(tmp_path, capsys):
    text = "Cloudflare Worker 跑\n用 worker 跑\n用 WORKER 跑\n用 service worker 跑\n中文 networker 不算\n"
    code, out = draft_hits(tmp_path, capsys, text)
    assert code == 1
    assert out.count("\n") == 1 and "draft.md:1\tWorker" in out


def test_draft_skips_lowercase_and_acronym_avoid_terms(tmp_path, capsys):
    avoided = set(english_avoid_terms())
    assert {"push", "PR"} <= avoided
    code, out = draft_hits(tmp_path, capsys, "我們要 push 分支\n這個 PR 改了三個檔\n")
    assert (code, out) == (0, "")


def test_draft_matches_multi_word_avoid_term(tmp_path, capsys):
    code, out = draft_hits(tmp_path, capsys, "顯示 Box Score 給使用者\n")
    assert code == 1 and "\tBox Score\t" in out


def test_source_scan_does_not_use_english_terms():
    _, avoid = parse_glossary()
    assert not any(w in avoid for w in english_avoid_terms())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
