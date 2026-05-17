
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _metric_keys(runs: list[dict[str, Any]]) -> list[str]:
    keys = set()
    for row in runs:
        for key, value in row.items():
            if key in {"mode", "seed"}:
                continue
            if isinstance(value, (int, float)):
                keys.add(key)
    return sorted(keys)


def summarize_benchmark(data: dict[str, Any]) -> dict[str, Any]:
    runs = list(data.get("runs", []))
    by_mode = data.get("by_mode", {})
    modes = sorted(by_mode.keys())
    keys = _metric_keys(runs)
    summary = {"modes": modes, "metrics": keys, "by_mode": by_mode, "ranking": []}
    if "total_mean" in next(iter(by_mode.values()), {}):
        summary["ranking"] = sorted(
            ((mode, by_mode[mode]["total_mean"]) for mode in modes if "total_mean" in by_mode[mode]),
            key=lambda x: x[1],
        )
    base = by_mode.get("base", {})
    if base:
        deltas = {}
        for mode, stats in by_mode.items():
            if mode == "base":
                continue
            delta = {}
            for key in keys:
                mean_key = f"{key}_mean"
                if mean_key in stats and mean_key in base:
                    delta[key] = stats[mean_key] - base[mean_key]
            deltas[mode] = delta
        summary["delta_vs_base"] = deltas
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    modes = summary.get("modes", [])
    by_mode = summary.get("by_mode", {})
    metrics = [m for m in summary.get("metrics", []) if any(f"{m}_mean" in stats for stats in by_mode.values())]
    lines = ["# RIPII Benchmark Summary", ""]
    if summary.get("ranking"):
        lines.append("## Ranking by total loss")
        lines.append("")
        for mode, value in summary["ranking"]:
            lines.append(f"- {mode}: {value:.6f}")
        lines.append("")
    header = ["mode"] + metrics[:8]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for mode in modes:
        stats = by_mode.get(mode, {})
        row = [mode]
        for metric in metrics[:8]:
            key = f"{metric}_mean"
            value = stats.get(key, float('nan'))
            row.append(f"{value:.6f}")
        lines.append("| " + " | ".join(row) + " |")
    if summary.get("delta_vs_base"):
        lines.append("")
        lines.append("## Delta vs base")
        lines.append("")
        for mode, delta in summary["delta_vs_base"].items():
            items = ", ".join(f"{k}={v:+.6f}" for k, v in sorted(delta.items())[:6])
            lines.append(f"- {mode}: {items}")
    return "\n".join(lines).strip() + "\n"


def write_report(data: dict[str, Any], path: str | Path, summary_json: str | Path | None = None) -> dict[str, Any]:
    summary = summarize_benchmark(data)
    md = render_markdown(summary)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    if summary_json is not None:
        summary_path = Path(summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
