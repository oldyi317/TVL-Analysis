"""查 .tw 網域的到期日（TWNIC 的 RDAP 公開查詢），供 .github/workflows/domain-expiry.yml 每週執行。"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import NamedTuple
from urllib.request import urlopen

RDAP_URL = "https://ccrdap.twnic.tw/tw/domain/{domain}"
TAIPEI = timezone(timedelta(hours=8))


class Expiry(NamedTuple):
    expires: datetime
    days_left: int
    due: bool


def _expiration(payload) -> datetime:
    try:
        raw = next(e["eventDate"] for e in payload["events"] if e["eventAction"] == "expiration")
        expires = datetime.fromisoformat(raw)
    except (KeyError, TypeError, StopIteration, ValueError) as e:
        raise ValueError(f"回應裡找不到可用的到期日（{e!r}）") from e
    if expires.tzinfo is None:
        raise ValueError(f"到期日沒有時區：{raw}")
    return expires


def check(payload, now: datetime, threshold: int) -> Expiry:
    expires = _expiration(payload)
    days_left = (expires - now).days
    return Expiry(expires, days_left, days_left < threshold)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="查 .tw 網域的到期日")
    parser.add_argument("--domain", default="vstat.tw")
    parser.add_argument("--threshold", type=int, default=90, help="剩餘天數少於這個數就算該提醒（預設 90）")
    args = parser.parse_args(argv)

    try:
        with urlopen(RDAP_URL.format(domain=args.domain), timeout=30) as resp:
            payload = json.load(resp)
        result = check(payload, datetime.now(timezone.utc), args.threshold)
    except (OSError, ValueError) as e:
        print(f"查不到 {args.domain} 的到期日：{e}", file=sys.stderr)
        return 1

    expires_text = result.expires.astimezone(TAIPEI).strftime("%Y-%m-%d %H:%M") + "（台灣時間）"
    print(f"{args.domain} 到期日 {expires_text}，剩 {result.days_left} 天（門檻 {args.threshold} 天）")
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"expires={expires_text}\ndays_left={result.days_left}\ndue={str(result.due).lower()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
