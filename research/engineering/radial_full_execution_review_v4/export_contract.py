"""Describe actual returned artifacts and explicitly account for absent outputs."""

import csv
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from readiness import digest


def emit(root, status):
    root = Path(root)

    def write(name, value):
        (root / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    versions = {}
    for name in ["torch", "numpy", "matplotlib", "PyYAML", "tqdm"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    write(
        "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "distributions": versions,
        },
    )
    write(
        "dataset_checksums.json",
        {
            str(p.relative_to(root)): digest(p)
            for p in sorted((root / "execution/data").glob("*.npz"))
        },
    )
    write(
        "checkpoint_manifest.json",
        {
            str(p.relative_to(root)): {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted((root / "execution/shards").glob("*/fits/*/*/*.pt"))
        },
    )
    with (root / "failures.jsonl").open("w") as f:
        if status.get("reason"):
            f.write(
                json.dumps({"status": status["status"], "reason": status["reason"]})
                + "\n"
            )
    records = []
    for p in sorted((root / "execution/shards").glob("*/fits/*/*/receipt.json")):
        row = json.loads(p.read_text())
        records.append(
            {
                k: row[k]
                for k in [
                    "domain",
                    "block",
                    "seed",
                    "model",
                    "phase",
                    "lr",
                    "updates",
                    "seconds",
                ]
            }
        )
    with (root / "experiments.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "domain",
                "block",
                "seed",
                "model",
                "phase",
                "lr",
                "updates",
                "seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    mappings = {
        "raw_metrics": "execution/evaluations.json",
        "logs": "run.log",
        "configs": "source/reference/protocol.json",
        "runtime_vram": "qualification",
        "metrics.csv": "analysis/metrics.csv",
        "ablations.csv": "analysis/ablations.csv",
        "baselines.csv": "analysis/baselines.csv",
        "seeds.csv": "analysis/seeds.csv",
        "figures": "analysis",
        "paper_table_data.json": "analysis/paper_table_data.json",
    }
    write(
        "artifact_contract.json",
        {
            "items": {
                name: {
                    "path": path,
                    "exists": (root / path).exists(),
                    "missing_reason": None
                    if (root / path).exists()
                    else "Stage not executed or no outcomes; returned status/logs explain the stop. Analysis products require ANALYZE_ONLY.",
                }
                for name, path in mappings.items()
            },
            "completed_fit_receipts": len(records),
            "note": "Empty inventory is absence of evidence, never a zero score. Qualification timing/VRAM is engineering-only; full-worker profiling is under execution/cost-profile.",
        },
    )
    (root / "RESULTS_RETURN.md").write_text(
        "Return the complete ripii_execution_v3_results.zip and its SHA-256. Preserve the provider output version. Do not remove raw predictions, checkpoints or failed attempts. artifact_contract.json maps requested artifacts and records absences. All scientific conclusions require independent intake.\n"
    )
