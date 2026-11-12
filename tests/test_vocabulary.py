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


def scan_text(label, text, formal, avoid):
    results = []
    for no, line in enumerate(text.splitlines(), 1):
        if not CJK.search(line) or line.rstrip().endswith(RAW_MARK) or line.rstrip().endswith(RAW_MARK + " -->"):
            continue
        for word in find_hits(line, formal, avoid):
            results.append((label, no, word, line.strip()))
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
    results = []
    for arg in argv:
        results += scan_text(arg, Path(arg).read_text(encoding="utf-8"), formal, avoid)
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


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
