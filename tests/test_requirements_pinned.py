import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+==[A-Za-z0-9_.]+")


def _lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_requirements_txt_all_pinned():
    for line in _lines(ROOT / "requirements.txt"):
        assert PIN_PATTERN.match(line), f"未釘版本：{line}"


def test_requirements_txt_no_genai():
    source = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "genai" not in source.lower()


def test_requirements_dev_pinned():
    for line in _lines(ROOT / "requirements-dev.txt"):
        assert PIN_PATTERN.match(line), f"未釘版本：{line}"


def test_optuna_not_in_any_requirements():
    for fname in ["requirements.txt", "requirements-dev.txt"]:
        source = (ROOT / fname).read_text(encoding="utf-8").lower()
        assert "optuna" not in source
