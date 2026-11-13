"""彙整 Playwright JSON 報表：實際 worker 數、測試數、逐檔耗時與最慢的測試（scripts/e2e.sh --profile 用）。

用法：python scripts/e2e_profile.py <report.json>
"""

import json
import sys
from collections import defaultdict

SLOWEST = 10


def collect(suite, file=None):
    file = suite.get("file") or file
    for spec in suite.get("specs", []):
        for test in spec.get("tests", []):
            seconds = sum(r.get("duration", 0) for r in test.get("results", [])) / 1000
            yield seconds, f"{file}:{spec['line']} {spec['title']}", file
    for child in suite.get("suites", []):
        yield from collect(child, file)


def main(path):
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    tests = [t for suite in report["suites"] for t in collect(suite)]
    stats = report["stats"]

    print(f"實際 worker 數：{report['config']['metadata'].get('actualWorkers')}")
    print(f"測試 {len(tests)} 個：{stats['expected']} 過、{stats['skipped']} 略過、"
          f"{stats['unexpected']} 失敗、{stats['flaky']} flaky")
    print(f"整輪 {stats['duration'] / 1000:.1f} 秒，各測試耗時加總 {sum(t[0] for t in tests):.1f} 秒")

    per_file = defaultdict(list)
    for seconds, _, file in tests:
        per_file[file].append(seconds)
    print("\n逐檔（加總／個數／平均）：")
    for file, times in sorted(per_file.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {sum(times):7.1f} 秒  {len(times):3d} 個  {sum(times) / len(times):5.2f} 秒  {file}")

    print(f"\n最慢的測試（前 {SLOWEST} 個）：")
    for seconds, name, _ in sorted(tests, key=lambda t: -t[0])[:SLOWEST]:
        print(f"  {seconds:5.1f} 秒  {name}")


if __name__ == "__main__":
    main(sys.argv[1])
