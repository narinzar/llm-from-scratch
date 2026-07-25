#!/usr/bin/env python3
"""Regenerate BENCHMARK.md from benchmarks/runs.jsonl.

Notebooks rewrite the report themselves via `llmfs.bench.log_run`, so you only
need this after editing runs.jsonl by hand, after a merge, or to check the
report is in sync in CI:

    python tools/update_benchmarks.py           # rewrite BENCHMARK.md
    python tools/update_benchmarks.py --check   # exit 1 if it would change
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmfs.bench import REPORT, ROOT, load, render  # noqa: E402


def main() -> int:
    check = "--check" in sys.argv[1:]
    runs = load()

    if check:
        before = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""
        if render(REPORT) != before:
            print("BENCHMARK.md is out of date -- run tools/update_benchmarks.py")
            return 1
        print(f"BENCHMARK.md is up to date ({len(runs)} run(s))")
        return 0

    render()
    stages = {r["stage"] for r in runs}
    print(f"wrote {REPORT.relative_to(ROOT)} from {len(runs)} run(s) "
          f"across {len(stages)} stage(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
