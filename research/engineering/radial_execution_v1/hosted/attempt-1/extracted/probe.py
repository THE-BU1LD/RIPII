"""Bounded CUDA equivalence and performance probe, not a scientific fit."""

import copy
import gc
import json
import platform
import time
import traceback
from pathlib import Path

import torch
from optimized import checked_forward, make_optimized, reference

OUT = Path("results")
OUT.mkdir(exist_ok=True)
NAMES = ("radial_mixture", "radial_fixed", "graph", "simple_pool", "transformer")


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def loss(model, x, y):
    return (model(x, y.shape[1])[0] - y).square().mean()


def compare(a, b, label, atol=2e-5, rtol=2e-4):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=label)
    return float((a - b).abs().max())


def train_step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    value = loss(model, x, y)
    value.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return value, norm


def qualify(name, width):
    torch.manual_seed(71001)
    original = reference.make_model(name, width, 8).cuda().train()
    # Nonzero trained-like heads prevent vacuous zero-initialization equivalence.
    with torch.no_grad():
        for pname, param in original.named_parameters():
            if "head" in pname or "expert" in pname:
                param.add_(torch.randn_like(param) * 0.0001)
    optimized = make_optimized(original)
    x = torch.randn(3, 8, 5, 4, device="cuda") * 0.05
    y = torch.randn(3, 64, 5, 4, device="cuda") * 0.05
    rows = []
    for horizon in (1, 8, 64):
        original.zero_grad(set_to_none=True)
        optimized.zero_grad(set_to_none=True)
        a = checked_forward(original, x, horizon)[0]
        b = checked_forward(optimized, x, horizon)[0]
        prediction_error = compare(a, b, f"{name} prediction H={horizon}")
        (a - y[:, :horizon]).square().mean().backward()
        (b - y[:, :horizon]).square().mean().backward()
        errors = []
        for (key, p), (other, q) in zip(
            original.named_parameters(), optimized.named_parameters(), strict=True
        ):
            assert key == other
            assert (p.grad is None) == (q.grad is None), key
            if p.grad is not None:
                errors.append(compare(p.grad, q.grad, f"{name} gradient {key}"))
        rows.append(
            {
                "horizon": horizon,
                "prediction_max_abs": prediction_error,
                "gradient_max_abs": max(errors),
                "active_gradients": len(errors),
            }
        )
    # Near radius-threshold and empty-edge cases expose dense-mask mistakes.
    if name in {"graph", "simple_pool"}:
        for distance in (0.0, 0.5999, 0.6001, 2.0):
            edge_x = x.clone()
            edge_x[..., :2] = 0
            edge_x[:, :, :, 0] = torch.arange(5, device="cuda") * distance
            edge_x[..., 2:] = 0
            compare(original(edge_x, 1)[0], optimized(edge_x, 1)[0], "edge boundary")
    before = copy.deepcopy(original.state_dict())
    aopt = torch.optim.AdamW(original.parameters(), lr=0.001)
    bopt = torch.optim.AdamW(optimized.parameters(), lr=0.001, capturable=True)
    for _ in range(3):
        train_step(original, aopt, x, y[:, :8])
        train_step(optimized, bopt, x, y[:, :8])
    update_errors = [
        compare(p, optimized.state_dict()[key], "optimizer " + key)
        for key, p in original.state_dict().items()
        if p.is_floating_point()
    ]
    assert any(not torch.equal(before[k], v) for k, v in original.state_dict().items())
    torch.save(optimized.state_dict(), OUT / f"{name}-checkpoint.pt")
    restored = make_optimized(reference.make_model(name, width, 8).cuda())
    restored.load_state_dict(
        torch.load(OUT / f"{name}-checkpoint.pt", weights_only=True)
    )
    compare(
        optimized(x, 64)[0], restored(x, 64)[0], "checkpoint replay", atol=0, rtol=0
    )
    # Explicit public boundary rejects invalid inputs.
    bad = x.clone()
    bad[0, 0, 0, 0] = float("nan")
    try:
        checked_forward(optimized, bad, 1)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("NaN accepted")
    return {
        "model": name,
        "checks": rows,
        "optimizer_max_abs": max(update_errors),
        "checkpoint_replay": "PASS",
        "invalid_input": "REJECTED",
        "status": "PASS",
    }


def profile(name, width):
    torch.manual_seed(72001)
    model = make_optimized(reference.make_model(name, width, 8).cuda()).train()
    x = torch.randn(16, 8, 5, 4, device="cuda") * 0.05
    y = torch.randn(16, 64, 5, 4, device="cuda") * 0.05
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, capturable=True)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            train_step(model, optimizer, x, y)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    # Capture the training step with static input buffers and retained outputs.
    graph = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        value, norm = train_step(model, optimizer, x, y)
    # Capture itself does not execute updates. Replay must match eager training
    # from the identical post-warmup parameters AND optimizer state.
    eager = make_optimized(reference.make_model(name, width, 8).cuda()).train()
    eager.load_state_dict(copy.deepcopy(model.state_dict()))
    eopt = torch.optim.AdamW(eager.parameters(), lr=0.001, capturable=True)
    eopt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    errors = []
    for _ in range(3):
        x.copy_(torch.randn_like(x) * 0.05)
        y.copy_(torch.randn_like(y) * 0.05)
        graph.replay()
        expected, _ = train_step(eager, eopt, x, y)
        torch.cuda.synchronize()
        compare(value, expected, "captured objective")
        errors.extend(
            compare(p, eager.state_dict()[key], "captured optimizer " + key)
            for key, p in model.state_dict().items()
            if p.is_floating_point()
        )
    samples = []
    for _ in range(7):
        torch.cuda.synchronize()
        start = time.monotonic()
        graph.replay()
        torch.cuda.synchronize()
        # Keep post-update finite checks in measured safe-use time.
        if not torch.isfinite(value) or not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite captured update")
        samples.append(time.monotonic() - start)
    del graph
    return {
        "model": name,
        "captured_update_seconds_max": max(samples),
        "samples": samples,
        "captured_optimizer_max_abs": max(errors),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "status": "PASS",
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no local CPU fallback")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    save(
        "environment.json",
        {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": torch.cuda.get_device_name(0),
            "devices": torch.cuda.device_count(),
        },
    )
    widths = reference.matched_hidden_sizes(8)
    rows = []
    for name in NAMES:
        row = {"model": name}
        try:
            row["equivalence"] = qualify(name, widths[name])
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            row["profile"] = profile(name, widths[name])
            row["status"] = "PASS"
        except Exception:
            row.update(status="FAIL", error=traceback.format_exc())
        rows.append(row)
        save("probe.json", rows)
        print(json.dumps(row), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    passed = all(r["status"] == "PASS" for r in rows)
    summary = {
        "engineering_status": "PASS" if passed else "FAIL",
        "scientific_result": "NOT_EVALUATED",
        "comparison_fits": 0,
        "test_data_opened": False,
    }
    if passed:
        seconds = sum(r["profile"]["captured_update_seconds_max"] for r in rows)
        summary["minimum_update_only_host_seconds"] = 28 * 6000 * seconds
        summary["maximum_update_only_host_seconds"] = 28 * 16000 * seconds
        summary["cost_caveat"] = (
            "Update-only extrapolation excludes data, validation, capture, serialization and safety reserve."
        )
    save("summary.json", summary)


if __name__ == "__main__":
    main()
