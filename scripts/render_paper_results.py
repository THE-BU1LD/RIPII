"""Render manuscript result text from a verified immutable result capsule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ripii.world.experiment import verify_capsule

PLACEHOLDER = "{{WORLD_V3_RESULTS}}"


def render_world_v3(capsule_path: Path) -> str:
    verification = verify_capsule(capsule_path)
    if verification["status"] != "PASS":
        raise ValueError("paper requires a content-verified v2 result capsule")
    capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
    summary = json.loads(capsule["retained"]["summary.json"]["content_text"])
    if summary.get("decision") != "no_advance":
        raise ValueError("world-v3 manuscript is bound to the no-advance decision")
    models = summary["by_model"]
    graph = models["graph_continuous"]["test"]["position_rmse_mean"]
    global_pool = models["global_pool_continuous"]["test"]["position_rmse_mean"]
    multiscale = models["multiscale_continuous"]["test"]["position_rmse_mean"]
    controls = summary["paired_controls"]
    graph_ood = 100.0 * controls["graph"]["ood_relative_improvement_mean"]
    global_ood = 100.0 * controls["global_pool"]["ood_relative_improvement_mean"]
    global_id = 100.0 * controls["global_pool"]["id_relative_improvement_mean"]
    graph_losses = controls["graph"]["seeds"] - controls["graph"]["ood_seed_wins"]
    return "\n".join(
        [
            f"Mean ID position RMSE is {graph:.4f} for graph, {global_pool:.4f} for global pool, and {multiscale:.4f} for multiscale. Graph also has the lowest mean error in every OOD regime. Multiscale's mean OOD relative improvement versus graph is {graph_ood:.2f}% and it loses all {graph_losses} paired seeds. Versus global pool, its mean OOD advantage is {global_ood:.2f}%, but paired direction is mixed and ID error is {-global_id:.2f}% worse. Exact unrounded values and provenance are in the content-verified v3 capsule cited by `EVIDENCE_LEDGER.md`.",
            "",
            "| Model | Mean ID position RMSE | OOD comparison with multiscale |",
            "|---|---:|---|",
            f"| Graph | **{graph:.4f}** | Better in every reported OOD regime |",
            f"| Global pool | {global_pool:.4f} | Multiscale mean advantage {global_ood:.2f}%; paired direction mixed |",
            f"| Multiscale | {multiscale:.4f} | {graph_ood:.2f}% relative improvement versus graph |",
        ]
    )


def render_manuscript(template: Path, capsule: Path) -> str:
    source = template.read_text(encoding="utf-8")
    if source.count(PLACEHOLDER) != 1:
        raise ValueError("manuscript must contain exactly one world-v3 placeholder")
    return source.replace(PLACEHOLDER, render_world_v3(capsule))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rendered = render_manuscript(args.template, args.capsule)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
