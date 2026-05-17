
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ripii.utils.reporting import write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--summary-json", type=str, default="")
    args = parser.parse_args()
    benchmark_path = Path(args.benchmark)
    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    out = Path(args.output) if args.output else benchmark_path.with_suffix(".md")
    summary_json = Path(args.summary_json) if args.summary_json else out.with_suffix(".summary.json")
    summary = write_report(data, out, summary_json=summary_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
