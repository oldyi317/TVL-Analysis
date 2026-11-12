import io
import json
from datetime import datetime, timezone
from urllib.error import HTTPError

import pytest

from src.utils import domain_expiry
from src.utils.domain_expiry import check, main


def _payload(expiration="2027-11-12T03:04:05Z"):
    return {
        "objectClassName": "domain",
        "ldhName": "vstat.tw",
        "events": [
            {"eventAction": "registration", "eventDate": "2026-11-12T03:04:05Z"},
            {"eventAction": "expiration", "eventDate": expiration},
            {"eventAction": "last update of RDAP database", "eventDate": "2026-11-13T05:07:11Z"},
        ],
    }


def test_到期日取自_expiration_事件並算出剩餘整天數():
    result = check(_payload(), now=datetime(2026, 11, 12, 3, 4, 5, tzinfo=timezone.utc), threshold=90)

    assert result.expires == datetime(2027, 11, 12, 3, 4, 5, tzinfo=timezone.utc)
    assert result.days_left == 365
    assert result.due is False


@pytest.mark.parametrize(
    "now, days_left, due",
    [
        (datetime(2027, 8, 14, 3, 4, 5, tzinfo=timezone.utc), 90, False),
        (datetime(2027, 8, 14, 3, 4, 6, tzinfo=timezone.utc), 89, True),
        (datetime(2027, 11, 12, 3, 4, 6, tzinfo=timezone.utc), -1, True),
    ],
)
def test_剩餘天數少於門檻才算該提醒(now, days_left, due):
    result = check(_payload(), now=now, threshold=90)

    assert (result.days_left, result.due) == (days_left, due)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"events": [{"eventAction": "registration", "eventDate": "2026-11-12T03:04:05Z"}]},
        _payload(expiration="12/11/2027"),
        _payload(expiration="2027-11-12T03:04:05"),
        _payload(expiration=None),
        ["not", "an", "object"],
    ],
    ids=["沒有 events", "沒有 expiration 事件", "日期格式不對", "日期沒有時區", "日期是 null", "回應不是物件"],
)
def test_回應裡找不到可用的到期日就報錯(payload):
    with pytest.raises(ValueError):
        check(payload, now=datetime(2026, 11, 12, tzinfo=timezone.utc), threshold=90)


@pytest.fixture
def rdap(monkeypatch):
    """把對 TWNIC 的 HTTP 呼叫換成假的；回傳的 dict 記下被查的網址，並可指定回應內容或例外。"""
    state = {"body": json.dumps(_payload(expiration="2099-12-31T23:59:59Z")).encode(), "error": None, "urls": []}

    def fake_urlopen(url, timeout):
        state["urls"].append(url)
        if state["error"]:
            raise state["error"]
        return io.BytesIO(state["body"])

    monkeypatch.setattr(domain_expiry, "urlopen", fake_urlopen)
    return state


@pytest.fixture
def outputs(monkeypatch, tmp_path):
    path = tmp_path / "github_output"
    path.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(path))
    return lambda: dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines())


def test_查到到期日就印出並寫進_step_outputs(rdap, outputs, capsys):
    assert main(["--threshold", "90"]) == 0

    assert rdap["urls"] == ["https://ccrdap.twnic.tw/tw/domain/vstat.tw"]
    written = outputs()
    assert written["expires"] == "2100-01-01 07:59（台灣時間）"
    assert int(written["days_left"]) > 90
    assert written["due"] == "false"
    out = capsys.readouterr().out
    assert "2100-01-01 07:59（台灣時間）" in out
    assert written["days_left"] in out


def test_門檻大於剩餘天數時_due_為_true(rdap, outputs):
    assert main(["--threshold", "100000"]) == 0

    assert outputs()["due"] == "true"


def test_不在_actions_裡跑也能查(rdap, monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    assert main(["--domain", "example.tw"]) == 0

    assert rdap["urls"] == ["https://ccrdap.twnic.tw/tw/domain/example.tw"]
    assert "2100-01-01" in capsys.readouterr().out


@pytest.mark.parametrize(
    "broken",
    [
        {"error": HTTPError("https://ccrdap.twnic.tw/tw/domain/vstat.tw", 404, "Not Found", None, None)},
        {"error": TimeoutError("timed out")},
        {"body": b"<html>503</html>"},
        {"body": b""},
        {"body": json.dumps({"events": []}).encode()},
    ],
    ids=["404", "連線逾時", "回應不是 JSON", "回應是空的", "沒有到期日"],
)
def test_查詢失敗以非零結束且不寫_outputs(rdap, outputs, capsys, broken):
    rdap.update(broken)

    assert main(["--threshold", "90"]) != 0

    assert outputs() == {}
    assert "vstat.tw" in capsys.readouterr().err
