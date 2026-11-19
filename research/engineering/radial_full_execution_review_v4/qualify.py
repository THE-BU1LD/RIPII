"""CUDA qualification in isolated processes; generated fixtures are not research data."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from readiness import (
    QUALIFICATION_HORIZONS,
    identity,
    validate_reference_result,
    verify_inventory,
    worker_environment,
)

ROOT = Path(__file__).resolve().parent
NAMES = ("radial_mixture", "radial_fixed", "graph", "simple_pool", "transformer")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def numeric(mode, name, output):
    sys.path[:0] = [str(ROOT / "reference"), str(ROOT)]
    import copy

    import torch
    from captured_engine import CapturedEngine
    from optimized import reference

    protocol = json.loads((ROOT / "reference/protocol.json").read_text())
    if mode == "reference":
        # Compare to the frozen model, not a second copy of the optimized path.
        # Do not run probe.profile: capture qualification below exercises the
        # actual CapturedEngine that will execute the study.
        import probe

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        probe.OUT = output
        widths = reference.matched_hidden_sizes(
            protocol["context"], protocol["candidate_hidden"]
        )
        result = probe.qualify(name, widths[name], horizons=QUALIFICATION_HORIZONS)
        validate_reference_result(name, result)
        write(output / "result.json", result)
        return
    if mode == "capture":
        engine = CapturedEngine(output, protocol, time.monotonic() + 900)
        model = engine.model(name, 73101)
        eager = copy.deepcopy(model)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=torch.tensor(0.001, device="cuda"),
            weight_decay=0.0001,
            capturable=True,
        )
        eopt = torch.optim.AdamW(
            eager.parameters(),
            lr=torch.tensor(0.001, device="cuda"),
            weight_decay=0.0001,
            capturable=True,
        )
        g = torch.Generator().manual_seed(73102)
        eg = torch.Generator().manual_seed(73102)
        data = (
            torch.randn(32, 72, 5, 4, generator=torch.Generator().manual_seed(73103))
            * 0.05
        )
        engine._graphs = {}
        checks = []
        for _, horizon in protocol["curriculum"]:
            samples = []
            for replay_index in range(3):
                if replay_index == 2:
                    # Warmup and capture have already completed. Mutate the
                    # existing LR tensors so this tests a captured replay's
                    # response to the scheduler, without replacing/recapturing.
                    next_lr = float(opt.param_groups[0]["lr"].item()) * 0.5
                    opt.param_groups[0]["lr"].fill_(next_lr)
                    eopt.param_groups[0]["lr"].fill_(next_lr)
                torch.cuda.synchronize()
                start = time.monotonic()
                loss, norm = engine.captured_update(model, opt, data, g, horizon)
                torch.cuda.synchronize()
                samples.append(time.monotonic() - start)
                other, enorm = engine.update(eager, eopt, data, eg, horizon)
                torch.testing.assert_close(
                    loss, loss.new_tensor(other), atol=2e-5, rtol=2e-4
                )
                torch.testing.assert_close(
                    norm, norm.new_tensor(enorm), atol=2e-5, rtol=2e-4
                )
                for k, v in model.state_dict().items():
                    torch.testing.assert_close(
                        v, eager.state_dict()[k], atol=2e-5, rtol=2e-4
                    )
                for p, ep in zip(model.parameters(), eager.parameters(), strict=True):
                    for k, v in opt.state[p].items():
                        if torch.is_tensor(v):
                            torch.testing.assert_close(
                                v, eopt.state[ep][k], atol=2e-5, rtol=2e-4
                            )
            checks.append({"horizon": horizon, "seconds": samples})
        # This is the diagnostic path used by fit and evaluation, not just forward.
        metric, pred = engine.metrics(model, data[:4])
        if name == "radial_mixture":
            assert 0.05 <= metric["gate_mean"] <= 0.95
            fixed = reference.make_model("radial_fixed", engine.widths[name], 8).cuda()
            fixed.load_state_dict(model.state_dict())
            _, intervention = engine.metrics(model, data[:4], gate=0.5)
            _, target = super(CapturedEngine, engine).metrics(fixed, data[:4])
            torch.testing.assert_close(intervention, target, atol=0, rtol=0)
            _, after = engine.metrics(model, data[:4])
            torch.testing.assert_close(pred, after, atol=0, rtol=0)
        write(
            output / "result.json",
            {
                "status": "PASS",
                "name": name,
                "checks": checks,
                "capture_lr_change": "PASS",
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            },
        )
        return

    protocol.update(
        forecast=4,
        train_scenes=8,
        validation_scenes=8,
        batch_size=2,
        curriculum=[[2, 1], [8, 4]],
        minimum_steps=100,
        maximum_steps=8,
        validate_every=2,
        plateau_window=2,
    )

    class Fixture(CapturedEngine):
        count = 0

        def data(self, domain, block, split, regime="standard"):
            seed = 74201 if split == "train" else 74203
            return (
                torch.randn(8, 12, 5, 4, generator=torch.Generator().manual_seed(seed))
                * 0.03
            )

        def captured_update(self, *args, **kwargs):
            if mode == "interrupt" and self.count == 4:
                raise InterruptedError("intentional interruption after durable step 4")
            self.count += 1
            return super().captured_update(*args, **kwargs)

    engine = Fixture(output, protocol, time.monotonic() + 300)
    try:
        row = engine.fit("springs", 74201, 74209, name, 0.001, "fixture")
    except InterruptedError:
        if mode != "interrupt":
            raise
        write(output / "result.json", {"status": "INTERRUPTED_AS_DESIGNED"})
    else:
        write(output / "result.json", {"status": "PASS", "row": row})


def main():
    # Also qualify direct CLI invocation, before numeric() imports torch.
    os.environ.update(worker_environment())
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--mode",
        choices=["all", "reference", "capture", "interrupt", "resume", "continuous"],
        default="all",
    )
    p.add_argument("--name", choices=NAMES)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode != "all":
        numeric(args.mode, args.name, args.output)
        return
    source = json.loads((ROOT / "SOURCE.json").read_text())
    verify_inventory(ROOT, source)
    rows = []
    for name in NAMES:
        for mode in ["reference", "capture", "continuous", "interrupt", "resume"]:
            out = args.output / f"{name}-{mode}"
            if mode == "resume":
                shutil.copytree(
                    args.output / f"{name}-interrupt" / "fits", out / "fits"
                )
            out.mkdir(parents=True, exist_ok=True)
            with (out / "process.log").open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--mode",
                        mode,
                        "--name",
                        name,
                        "--output",
                        str(out),
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=900,
                )
            if mode == "reference":
                validate_reference_result(
                    name, json.loads((out / "result.json").read_text())
                )
        # Compare serialized optimizer, scheduler, sampler, model, and selection
        # in yet another process, so no live Python state can hide a broken resume.
        with (args.output / f"{name}-replay.log").open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "replay_checkpoint.py"),
                    str(args.output / f"{name}-continuous"),
                    str(args.output / f"{name}-resume"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
        rows.append(
            {
                "name": name,
                "frozen_reference_equivalence": "PASS",
                "capture_eager": "PASS",
                "capture_lr_change": "PASS",
                "resume_new_process": "PASS",
            }
        )
    write(
        args.output / "qualification.json",
        {
            "status": "PASS",
            "source_sha256": identity(source),
            "all_models": 5,
            "curriculum_horizons": 7,
            "resume_new_process": "PASS",
            "gate_intervention": "PASS",
            "capture_eager": "PASS",
            "capture_lr_change": "PASS",
            "frozen_reference_equivalence": "PASS",
            "test_data_opened": False,
            "scientific_result": "NOT_EVALUATED",
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
