"""Measured conservative serial cost admission; never invents a concurrency gain."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "reference"), str(ROOT)]
import torch  # noqa: E402
from captured_engine import CapturedEngine  # noqa: E402


def profile(output, deadline):
    p = json.loads((ROOT / "reference/protocol.json").read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    engine = CapturedEngine(output, p, deadline)
    rows = []
    for name in p["models"]:
        engine.check()
        model = engine.model(name, 75001)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=torch.tensor(0.001, device="cuda"),
            weight_decay=p["weight_decay"],
            capturable=True,
        )
        sampler = torch.Generator().manual_seed(75003)
        data = (
            torch.randn(
                p["train_scenes"],
                p["context"] + p["forecast"],
                5,
                4,
                generator=torch.Generator().manual_seed(75007),
            )
            * 0.05
        )
        engine._graphs = {}
        samples = []
        for i in range(8):
            torch.cuda.synchronize()
            start = time.monotonic()
            engine.captured_update(model, opt, data, sampler, 64)
            torch.cuda.synchronize()
            if i >= 3:
                samples.append(time.monotonic() - start)
        start = time.monotonic()
        engine.metrics(model, data)
        engine.metrics(model, data[: p["validation_scenes"]])
        validation = time.monotonic() - start
        rows.append(
            {
                "name": name,
                "update_max_seconds": max(samples),
                "samples": samples,
                "validation_seconds": validation,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            }
        )
        engine._graphs.clear()
        del model, opt
        torch.cuda.empty_cache()
    start = time.monotonic()
    for domain in p["domains"]:
        engine.data(domain, p["calibration_blocks"][0], "train")
    data_seconds = time.monotonic() - start
    repeats = len(p["domains"]) * (
        len(p["learning_rates"]) + len(p["data_blocks"]) * len(p["model_seeds"])
    )
    train = repeats * sum(
        r["update_max_seconds"] * p["maximum_steps"]
        + r["validation_seconds"] * (p["maximum_steps"] // p["validate_every"])
        for r in rows
    )
    data_factor = (1 + len(p["data_blocks"])) * (
        1
        + (p["validation_scenes"] + len(p["regimes"]) * p["test_scenes"])
        / p["train_scenes"]
    )
    evaluation = (
        sum(r["validation_seconds"] for r in rows)
        * len(p["domains"])
        * len(p["data_blocks"])
        * len(p["model_seeds"])
        * (len(p["regimes"]) + 2)
    )
    estimate = 1.25 * (train + data_factor * data_seconds + evaluation) + 600
    result = {
        "status": "PASS" if estimate < deadline - time.monotonic() else "BLOCKED_COST",
        "projected_serial_seconds": estimate,
        "remaining_seconds": deadline - time.monotonic(),
        "safety_factor": 1.25,
        "concurrency_credit": 0,
        "rows": rows,
        "real_train_data_seconds": data_seconds,
        "test_data_opened": False,
        "limitation": "Measured estimate, not a guaranteed upper bound; parallel benefit is unqualified.",
    }
    (output / "cost.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = profile(Path(sys.argv[1]), time.monotonic() + float(sys.argv[2]))
    print(json.dumps(result), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(2)
