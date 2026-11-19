"""Regenerate tables from retained raw receipts. Missing results are never zeros."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

from readiness import verify_inventory


def aggregate(root, output):
    root, output = Path(root), Path(output)
    verify_inventory(root, json.loads((root / "manifest.json").read_text())["files"])
    output.mkdir(parents=True, exist_ok=False)
    execution = root / "execution"
    rows = []
    if (execution / "evaluations.json").exists():
        raw = json.loads((execution / "evaluations.json").read_text())
        import numpy as np

        control_path = root / "source/reference/control.py"
        spec = importlib.util.spec_from_file_location(
            "independent_frozen_control", control_path
        )
        control = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(control)
        protocol = json.loads((root / "source/reference/protocol.json").read_text())
        for row in raw:
            if (
                control.adequate(
                    row["curve"],
                    row["selected"]["train_rmse"],
                    row["cv_train_rmse"],
                    protocol,
                )
                != row["adequacy"]
            ):
                raise ValueError("Adequacy replay differs")
            for regime in protocol["regimes"]:
                name = f"{row['domain']}-{row['block']}-{row['seed']}-{row['model']}-{regime}.npz"
                matches = list((execution / "shards").glob("*/predictions/" + name))
                if len(matches) != 1:
                    raise ValueError("Prediction artifact missing or ambiguous")
                with np.load(matches[0], allow_pickle=False) as data:
                    for array, metric in [("prediction", row["metrics"][regime])] + (
                        [("intervention", row["intervention"][regime])]
                        if row["model"] == "radial_mixture"
                        else []
                    ):
                        predicted, target = data[array], data["target"]
                        if (
                            predicted.shape != target.shape
                            or not np.isfinite(predicted).all()
                            or not np.isfinite(target).all()
                        ):
                            raise ValueError("Invalid prediction arrays")
                        squared = (
                            predicted.astype(np.float64) - target.astype(np.float64)
                        ) ** 2
                        for key, part in [
                            ("position_rmse", slice(None, 2)),
                            ("velocity_rmse", slice(2, None)),
                        ]:
                            if float(np.sqrt(squared[..., part].mean())) != metric[key]:
                                raise ValueError(
                                    "Independent raw metric replay differs"
                                )
        replay = control.summarize(raw, protocol)
        if replay != json.loads((execution / "summary.json").read_text()):
            raise ValueError("Frozen statistical summary differs")
        (output / "independent_replay.json").write_text(
            json.dumps({"status": "PASS", "summary": replay}, indent=2) + "\n"
        )
        for row in raw:
            for regime, metrics in row["metrics"].items():
                rows.append(
                    {k: row[k] for k in ("domain", "block", "seed", "model")}
                    | {
                        "regime": regime,
                        "position_rmse": metrics["position_rmse"],
                        "velocity_rmse": metrics["velocity_rmse"],
                    }
                )
    fields = [
        "domain",
        "block",
        "seed",
        "model",
        "regime",
        "position_rmse",
        "velocity_rmse",
    ]
    with (output / "metrics.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "OBSERVED" if rows else "PENDING_EXPERIMENTAL_EVIDENCE",
        "metrics": rows,
    }
    for filename, selected in (
        ("baselines.csv", [r for r in rows if r["model"] != "radial_mixture"]),
        ("ablations.csv", [r for r in rows if r["model"] == "radial_fixed"]),
        ("seeds.csv", rows),
    ):
        with (output / filename).open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    if rows:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grouped = {}
        for row in rows:
            if row["regime"] == "standard":
                grouped.setdefault((row["domain"], row["model"]), []).append(
                    row["position_rmse"]
                )
        labels = ["/".join(k) for k in sorted(grouped)]
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        ax.bar(labels, [sum(grouped[k]) / len(grouped[k]) for k in sorted(grouped)])
        ax.set_ylabel("Descriptive mean position RMSE")
        ax.tick_params(axis="x", rotation=60)
        fig.savefig(output / "position_rmse.png", dpi=180)
        plt.close(fig)
    (output / "paper_table_data.json").write_text(json.dumps(result, indent=2) + "\n")
    tex = [
        "\\begin{tabular}{llrr}",
        "Domain & Model & Cells & Mean position RMSE \\\\",
        "\\hline",
    ]
    for domain, model in sorted({(r["domain"], r["model"]) for r in rows}):
        values = [
            r["position_rmse"]
            for r in rows
            if r["domain"] == domain
            and r["model"] == model
            and r["regime"] == "standard"
        ]
        label = model.replace("_", r"\_")
        if values:
            tex.append(
                f"{domain} & {label} & {len(values)} & {sum(values) / len(values):.8g} \\\\"
            )
    if not rows:
        tex.append("\\multicolumn{4}{l}{PENDING EXPERIMENTAL EVIDENCE} \\\\")
    tex.append("\\end{tabular}")
    (output / "results.tex").write_text("\n".join(tex) + "\n")
    (output / "README.txt").write_text(
        "Descriptive tables only. Independent blocks, nested seeds and frozen inference remain required. No missing outcome is imputed.\n"
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("output")
    a = p.parse_args()
    print(json.dumps(aggregate(a.root, a.output)))
