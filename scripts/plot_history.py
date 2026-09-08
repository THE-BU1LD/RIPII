from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt


def load_history(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    rows = load_history(Path(args.history))
    xs = [r["step"] for r in rows]
    out = Path(args.output) if args.output else Path(args.history).with_suffix(".png")
    plt.figure(figsize=(7, 4))
    plt.plot(xs, [r["train_total"] for r in rows])
    plt.title("Training total")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
