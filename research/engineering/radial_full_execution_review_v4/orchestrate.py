"""Four-lane, two-GPU executor for the frozen radial-adaptation-v2 matrix."""

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from readiness import admit_full, identity, verify_inventory

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "reference"
TIMING = {
    "radial_mixture": 0.0535,
    "radial_fixed": 0.0470,
    "graph": 0.0999,
    "simple_pool": 0.0959,
    "transformer": 0.0754,
}
DEVICES = (0, 0, 1, 1)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def replace_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def claim_job(queue_path, worker_name):
    queue_path = Path(queue_path)
    lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = read(queue_path)
        if not queue["pending"]:
            return None
        job = queue["pending"].pop(0)
        queue["claimed"].append({"worker": worker_name, "job": job})
        replace_json(queue_path, queue)
        return job


def control_module():
    spec = importlib.util.spec_from_file_location(
        "frozen_control", REFERENCE / "control.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tuning_jobs(protocol):
    return [
        {
            "domain": domain,
            "block": protocol["calibration_blocks"][0],
            "seed": protocol["calibration_model_seed"],
            "name": name,
            "lr": lr,
            "phase": "calibration",
        }
        for domain in protocol["domains"]
        for name in protocol["models"]
        for lr in protocol["learning_rates"]
    ]


def choose_learning_rates(receipts, protocol):
    choices = {}
    for domain in protocol["domains"]:
        for name in protocol["models"]:
            rows = [
                row
                for row in receipts
                if row["domain"] == domain and row["model"] == name
            ]
            eligible = [row for row in rows if row["adequacy"]["pass"]]
            if eligible:
                choices[f"{domain}/{name}"] = min(
                    eligible,
                    key=lambda row: (row["selected"]["validation_rmse"], row["lr"]),
                )["lr"]
    return choices


def main_jobs(protocol, choices):
    jobs = []
    for domain in protocol["domains"]:
        for block in protocol["data_blocks"]:
            for seed in protocol["model_seeds"]:
                names = protocol["models"]
                offset = (block + seed) % len(names)
                for name in names[offset:] + names[:offset]:
                    jobs.append(
                        {
                            "domain": domain,
                            "block": block,
                            "seed": seed,
                            "name": name,
                            "lr": choices[f"{domain}/{name}"],
                            "phase": "main",
                        }
                    )
    return jobs


def assign(jobs, lanes=None):
    if lanes is None:
        lanes = len(DEVICES)
    groups, loads = [[] for _ in range(lanes)], [0.0] * lanes

    def model_name(row):
        return row.get("name", row.get("model"))

    for job in sorted(jobs, key=lambda row: TIMING[model_name(row)], reverse=True):
        lane = min(range(lanes), key=loads.__getitem__)
        groups[lane].append(job)
        loads[lane] += TIMING[model_name(job)]
    return groups, loads


def ensure_data_link(stage, shared):
    stage.mkdir(parents=True, exist_ok=False)
    os.symlink(shared, stage / "data", target_is_directory=True)


def worker(kind, root, stage, jobs_path, deadline):
    sys.path.insert(0, str(REFERENCE))
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from captured_engine import CapturedEngine, Plateau
    from engine import array_metrics

    protocol = read(REFERENCE / "protocol.json")
    jobs_payload = read(jobs_path)
    dynamic = isinstance(jobs_payload, dict) and "pending" in jobs_payload
    jobs = [] if dynamic else jobs_payload
    ensure_data_link(stage, root / "data")
    engine = CapturedEngine(stage, protocol, deadline)
    rows = []
    if kind == "data":
        for job in jobs:
            engine.data(job["domain"], job["block"], job["split"], job["regime"])
        write(stage / "worker-result.json", {"datasets": len(jobs)})
        return
    if kind == "fit":
        while True:
            job = claim_job(jobs_path, stage.name) if dynamic else None
            if not dynamic:
                if not jobs:
                    break
                job = jobs.pop(0)
            if job is None:
                break
            # Every session has fresh logs/receipts. Copy only the uniquely
            # identified fit from the source-verified previous session.
            fit_name = f"{job['domain']}-{job['block']}-{job['seed']}-{job['name']}-{job['lr']}"
            previous = list(
                (root / "resume" / "shards").glob(f"*/fits/{job['phase']}/{fit_name}")
            )
            if len(previous) > 1:
                raise RuntimeError("Ambiguous resumed fit")
            if previous:
                target = stage / "fits" / job["phase"] / fit_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(previous[0], target)
            row = engine.fit(**job)
            row["stage"] = stage.name
            row["checkpoint"] = str((stage / row["checkpoint"]).relative_to(root))
            rows.append(row)
        write(stage / "worker-result.json", rows)
        return
    if kind == "smoke":
        job = jobs[0]
        actual_parameter = torch.nn.Parameter(torch.ones((), device=engine.device))
        actual_optimizer = torch.optim.AdamW([actual_parameter], lr=job["lr"])
        actual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            actual_optimizer,
            factor=protocol["scheduler"]["factor"],
            patience=protocol["scheduler"]["patience"],
            min_lr=protocol["scheduler"]["minimum_lr"],
        )
        candidate_lr = torch.tensor(job["lr"], device=engine.device)
        candidate_scheduler = Plateau(candidate_lr, protocol["scheduler"])
        scheduler_trace = []
        for value in (1.0, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8):
            actual_scheduler.step(value)
            candidate_scheduler.step(value)
            actual_lr = actual_optimizer.param_groups[0]["lr"]
            rounded_actual = torch.tensor(
                actual_lr, dtype=candidate_lr.dtype, device=engine.device
            )
            if not torch.equal(candidate_lr, rounded_actual):
                raise RuntimeError("Device scheduler differs from frozen scheduler")
            scheduler_trace.append(
                {
                    "metric": value,
                    "frozen_lr": actual_lr,
                    "device_lr": float(candidate_lr),
                }
            )
        model = engine.model(job["name"], job["seed"])
        eager = copy.deepcopy(model)
        lr = torch.tensor(job["lr"], device=engine.device)
        eager_lr = lr.clone()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=protocol["weight_decay"],
            capturable=True,
        )
        eager_optimizer = torch.optim.AdamW(
            eager.parameters(),
            lr=eager_lr,
            weight_decay=protocol["weight_decay"],
            capturable=True,
        )
        sampler = torch.Generator().manual_seed(job["seed"] + 900001)
        eager_sampler = torch.Generator().manual_seed(job["seed"] + 900001)
        generator = torch.Generator().manual_seed(job["seed"] + 700001)
        train = (
            torch.randn(
                32,
                protocol["context"] + protocol["forecast"],
                protocol["objects"],
                4,
                generator=generator,
            )
            * 0.05
        )
        engine._graphs = {}
        checked = []
        maximum_parameter_error = 0.0
        for _, horizon in protocol["curriculum"]:
            for _ in range(3):
                loss, grad = engine.captured_update(
                    model, optimizer, train, sampler, horizon
                )
                eager_loss, eager_grad = engine.update(
                    eager, eager_optimizer, train, eager_sampler, horizon
                )
            torch.cuda.synchronize()
            if not torch.isfinite(loss) or not torch.isfinite(grad):
                raise FloatingPointError("Nonfinite multi-worker smoke update")
            if abs(float(loss) - eager_loss) > 2e-5 + 2e-4 * abs(eager_loss):
                raise RuntimeError("Captured objective differs from eager execution")
            if abs(float(grad) - eager_grad) > 2e-5 + 2e-4 * abs(eager_grad):
                raise RuntimeError(
                    "Captured gradient norm differs from eager execution"
                )
            for key, value in model.state_dict().items():
                reference_value = eager.state_dict()[key]
                if value.is_floating_point():
                    error = float((value - reference_value).abs().max())
                    maximum_parameter_error = max(maximum_parameter_error, error)
                    torch.testing.assert_close(
                        value,
                        reference_value,
                        atol=2e-5,
                        rtol=2e-4,
                        msg=f"captured optimizer {job['name']} {key} H={horizon}",
                    )
                elif not torch.equal(value, reference_value):
                    raise RuntimeError("Nonfloating model state differs")
            checked.append(horizon)
        write(
            stage / "worker-result.json",
            [
                {
                    "model": job["name"],
                    "horizons": checked,
                    "captured_eager_max_abs": maximum_parameter_error,
                    "scheduler_equivalence": "PASS",
                    "scheduler_trace": scheduler_trace,
                    "status": "PASS",
                }
            ],
        )
        return
    if kind != "evaluate":
        raise ValueError(kind)
    while True:
        selection = claim_job(jobs_path, stage.name) if dynamic else None
        if not dynamic:
            if not jobs:
                break
            selection = jobs.pop(0)
        if selection is None:
            break
        engine.check()
        domain, block, seed, name = (
            selection[key] for key in ("domain", "block", "seed", "model")
        )
        model = engine.model(name, seed)
        model.load_state_dict(
            torch.load(
                root / selection["checkpoint"],
                map_location=engine.device,
                weights_only=True,
            )["model"]
        )
        metrics, intervention = {}, {}
        for regime in protocol["regimes"]:
            data = engine.data(domain, block, "test", regime)
            metric, prediction = engine.metrics(model, data)
            target = data[:, protocol["context"] :].numpy()
            path = (
                stage / "predictions" / f"{domain}-{block}-{seed}-{name}-{regime}.npz"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            arrays = {"prediction": prediction.numpy(), "target": target}
            metrics[regime] = metric
            if name == "radial_mixture":
                item, pred = engine.metrics(
                    model, data, gate=selection["selected"]["train_gate_mean"]
                )
                intervention[regime] = item
                arrays["intervention"] = pred.numpy()
            np.savez_compressed(path, **arrays)
            with np.load(path, allow_pickle=False) as archive:
                replay = array_metrics(archive["prediction"], archive["target"])
                if replay["position_rmse"] != metric["position_rmse"]:
                    raise RuntimeError("Prediction metric replay failed")
                if "intervention" in archive:
                    replay = array_metrics(archive["intervention"], archive["target"])
                    if replay["position_rmse"] != intervention[regime]["position_rmse"]:
                        raise RuntimeError("Intervention metric replay failed")
            _, repeated = engine.metrics(model, data)
            if not torch.equal(prediction, repeated):
                raise RuntimeError("Held-out prediction replay failed")
        row = {**selection, "metrics": metrics, "intervention": intervention}
        write(
            stage / "evaluations" / f"{domain}-{block}-{seed}-{name}.json",
            row,
        )
        rows.append(row)
    write(stage / "worker-result.json", rows)


def run_stage(kind, root, label, jobs, deadline, devices=DEVICES):
    dynamic = kind in {"fit", "evaluate"}
    queue_path = None
    if kind in {"fit", "evaluate"}:
        _, loads = assign(jobs, len(devices))
        groups = [[{"lane": lane}] for lane in range(len(devices))]
        ordered = sorted(
            jobs,
            key=lambda row: TIMING[row.get("name", row.get("model"))],
            reverse=True,
        )
        write(root / "jobs" / f"{label}-all.json", jobs)
        queue_path = root / "queues" / f"{label}.json"
        write(queue_path, {"pending": ordered, "claimed": []})
    elif kind == "smoke":
        if len(jobs) != len(devices):
            raise ValueError("Smoke qualification requires one job per lane")
        groups, loads = [[job] for job in jobs], [0.0] * len(devices)
    else:
        groups, loads = [jobs], [0.0]
    processes = []
    for lane, group in enumerate(groups):
        if not group:
            continue
        stage = root / "shards" / f"{label}-{lane}"
        if dynamic:
            jobs_path = queue_path
        else:
            jobs_path = root / "jobs" / f"{label}-{lane}.json"
            write(jobs_path, group)
        log = root / "logs" / f"{label}-{lane}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("x")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(devices[lane]))
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "orchestrate.py"),
                "--worker",
                kind,
                "--root",
                str(root),
                "--stage",
                str(stage),
                "--jobs",
                str(jobs_path),
                "--deadline",
                str(deadline),
            ],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, handle, stage, log))
    failures = []
    timed_out = False
    for process, handle, stage, log in processes:
        left = max(1, deadline - time.monotonic())
        try:
            code = process.wait(timeout=left)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
        finally:
            handle.close()
        if code:
            failures.append({"stage": stage.name, "returncode": code, "log": str(log)})
    if failures:
        if timed_out:
            raise TimeoutError("SESSION_CAP_REACHED_WORKERS_STOPPED")
        raise RuntimeError("Worker failure: " + json.dumps(failures))
    if dynamic:
        queue = read(queue_path)
        if queue["pending"] or len(queue["claimed"]) != len(jobs):
            raise RuntimeError("Dynamic queue did not claim the complete matrix")
    rows = []
    for _, _, stage, _ in processes:
        result = read(stage / "worker-result.json")
        if isinstance(result, list):
            rows.extend(result)
    if dynamic:
        requested = {
            (
                row["domain"],
                row["block"],
                row["seed"],
                row.get("name", row.get("model")),
                row.get("lr"),
                row.get("phase"),
            )
            for row in jobs
        }
        returned = {
            (
                row["domain"],
                row["block"],
                row["seed"],
                row.get("name", row.get("model")),
                row.get("lr"),
                row.get("phase"),
            )
            for row in rows
        }
        if requested != returned or len(rows) != len(jobs):
            raise RuntimeError("Worker results do not match the requested matrix")
    write(
        root / "receipts" / f"{label}.json",
        {
            "scheduler": "dynamic_longest_estimate_first" if dynamic else "static",
            "planned_static_loads": loads,
            "workers": len(processes),
            "rows": rows,
        },
    )
    return rows


def verify_source():
    source = read(ROOT / "SOURCE.json")
    for name, digest in source.items():
        path = ROOT / name
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise RuntimeError(f"Source mismatch: {name}")


def copy_resume_snapshot(previous, destination):
    """Keep the newest fit transaction for every job across repeated timeouts."""
    shutil.copytree(previous, destination, ignore=shutil.ignore_patterns("resume"))
    present = {
        (p.parent.name, p.name)
        for p in (destination / "shards").glob("*/fits/*/*")
        if p.is_dir()
    }
    ancestor = previous / "resume"
    generation = 0
    while ancestor.is_dir():
        for fit in sorted((ancestor / "shards").glob("*/fits/*/*")):
            key = (fit.parent.name, fit.name)
            if not fit.is_dir() or key in present:
                continue
            target = (
                destination
                / "shards"
                / f"retained-{generation}-{fit.parents[2].name}"
                / "fits"
                / key[0]
                / key[1]
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(fit, target)
            present.add(key)
        ancestor = ancestor / "resume"
        generation += 1


def supervisor(output, seconds, qualification, resume=None):
    verify_source()
    admit_full(identity(read(ROOT / "SOURCE.json")), read(qualification))
    protocol = read(REFERENCE / "protocol.json")
    control = control_module()
    output.mkdir(parents=True, exist_ok=False)
    (output / "data").mkdir()
    if resume is not None:
        manifest = read(resume / "manifest.json")["files"]
        verify_inventory(resume, manifest)
        if read(resume / "execution-identity.json")["source_sha256"] != identity(
            read(ROOT / "SOURCE.json")
        ):
            raise RuntimeError("Resume source differs")
        if read(resume / "status.json").get("resume_allowed") is not True:
            raise RuntimeError(
                "Only a pre-test timeout is resumable; numerical/terminal failures are retained"
            )
        if read(resume / "status.json")["test_data_opened"]:
            raise RuntimeError(
                "Post-test interrupted execution requires independent review"
            )
        copy_resume_snapshot(resume, output / "resume")
        for path in (resume / "data").glob("*"):
            if path.is_file() and "-test-" not in path.name:
                shutil.copy2(path, output / "data" / path.name)
    write(
        output / "execution-identity.json",
        {"source_sha256": identity(read(ROOT / "SOURCE.json"))},
    )
    start = time.monotonic()
    deadline = start + seconds
    status = {
        "status": "FAILED",
        "scientific_result": "NOT_EVALUATED",
        "test_data_opened": False,
    }
    try:
        with (output / "cost-profile.log").open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "cost_profile.py"),
                    str(output / "cost-profile"),
                    str(max(1, deadline - time.monotonic())),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=max(1, deadline - time.monotonic()),
            )
        qualification = output / "qualification"
        qualification.mkdir()
        with (qualification / "probe.log").open("x") as log:
            subprocess.run(
                [sys.executable, str(ROOT / "probe.py"), "--reference-only"],
                cwd=qualification,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=min(1800, max(1, deadline - time.monotonic())),
            )
        probe = read(qualification / "results" / "summary.json")
        if (
            probe["engineering_status"] != "PASS"
            or probe.get("frozen_reference_equivalence") != "PASS"
            or probe.get("capture_profile") != "NOT_RUN"
        ):
            raise RuntimeError("Frozen-reference execution qualification failed")
        write(output / "qualification.json", probe)
        smoke = run_stage(
            "smoke",
            output,
            "four-lane-smoke",
            [
                {"name": name, "seed": 60501 + lane, "lr": 0.001}
                for lane, name in enumerate(
                    ("radial_mixture", "graph", "simple_pool", "transformer")
                )
            ],
            deadline,
        )
        if len(smoke) != 4 or not all(row["status"] == "PASS" for row in smoke):
            raise RuntimeError("Four-lane CUDA qualification failed")

        blocks = protocol["calibration_blocks"] + protocol["data_blocks"]
        datasets = [
            {"domain": domain, "block": block, "split": split, "regime": "standard"}
            for domain in protocol["domains"]
            for block in blocks
            for split in ("train", "validation")
        ]
        run_stage("data", output, "development-data", datasets, deadline, devices=(0,))

        tuning = run_stage(
            "fit", output, "calibration", tuning_jobs(protocol), deadline
        )
        if len(tuning) != 20:
            raise RuntimeError("Incomplete calibration matrix")
        choices = choose_learning_rates(tuning, protocol)
        write(output / "tuning.json", {"trials": tuning, "choices": choices})
        if len(choices) != len(protocol["domains"]) * len(protocol["models"]):
            status.update(
                status="TERMINAL", scientific_result="INCONCLUSIVE_UNDERTRAINED"
            )
            raise RuntimeError("INCONCLUSIVE_UNDERTRAINED_CALIBRATION_TEST_CLOSED")

        selections = run_stage(
            "fit", output, "main", main_jobs(protocol, choices), deadline
        )
        if len(selections) != 120:
            raise RuntimeError("Incomplete main matrix")
        for row in selections:
            replay = control.adequate(
                row["curve"],
                row["selected"]["train_rmse"],
                row["cv_train_rmse"],
                protocol,
            )
            if (
                replay != row["adequacy"]
                or sha(output / row["checkpoint"]) != row["checkpoint_sha256"]
            ):
                raise RuntimeError("Adequacy or checkpoint replay changed")
        write(output / "selections.json", selections)
        if not all(row["adequacy"]["pass"] for row in selections):
            status.update(
                status="TERMINAL", scientific_result="INCONCLUSIVE_UNDERTRAINED"
            )
            raise RuntimeError("INCONCLUSIVE_UNDERTRAINED_MAIN_TEST_CLOSED")
        write(
            output / "test-admission.json",
            {
                "status": "PASS",
                "selections_sha256": sha(output / "selections.json"),
                "protocol_sha256": sha(REFERENCE / "protocol.json"),
                "fits": len(selections),
            },
        )

        test_datasets = [
            {"domain": domain, "block": block, "split": "test", "regime": regime}
            for domain in protocol["domains"]
            for block in protocol["data_blocks"]
            for regime in protocol["regimes"]
        ]
        run_stage(
            "data", output, "protected-test-data", test_datasets, deadline, devices=(0,)
        )
        evaluations = run_stage("evaluate", output, "evaluation", selections, deadline)
        summary = control.summarize(evaluations, protocol)
        write(output / "evaluations.json", evaluations)
        write(output / "summary.json", summary)
        status.update(status="COMPLETE", scientific_result=summary["decision"])
    except Exception as exc:
        status["resume_allowed"] = isinstance(
            exc, (TimeoutError, subprocess.TimeoutExpired)
        )
        if status["status"] == "FAILED" and "INCONCLUSIVE_UNDERTRAINED" in str(exc):
            status.update(
                status="TERMINAL", scientific_result="INCONCLUSIVE_UNDERTRAINED"
            )
        status["reason"] = str(exc)
        write(output / "failure.json", {"error": str(exc)})
    finally:
        elapsed = time.monotonic() - start
        status["elapsed_host_seconds"] = elapsed
        status["test_data_opened"] = bool(list((output / "data").glob("*-test-*.npz")))
        status["resume_allowed"] = (
            bool(status.get("resume_allowed")) and not status["test_data_opened"]
        )
        write(output / "status.json", status)
        charged_host_seconds = elapsed + protocol["export_reserve_seconds"]
        write(
            output / "charge.json",
            {
                "elapsed_host_seconds": elapsed,
                "export_reserve_seconds": protocol["export_reserve_seconds"],
                "charged_host_seconds": charged_host_seconds,
                "charged_gpu_seconds": protocol["charged_device_count"]
                * charged_host_seconds,
                "allocated_gpu_hours_cap": protocol["free_gpu_hours_cap"],
                "cloud_spend_usd": 0,
            },
        )
        files = {
            str(path.relative_to(output)): sha(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
            and not path.is_symlink()
            and path != output / "manifest.json"
        }
        write(output / "manifest.json", {"files": files})
    print(json.dumps(status, indent=2), flush=True)
    if status["status"] not in {"COMPLETE", "TERMINAL"}:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds", type=float, default=30000)
    parser.add_argument("--worker", choices=("data", "fit", "evaluate", "smoke"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.root, args.stage, args.jobs, args.deadline)
    else:
        if args.qualification is None:
            parser.error(
                "--qualification is required; full execution is not yet admitted"
            )
        supervisor(args.output.resolve(), args.seconds, args.qualification, args.resume)


if __name__ == "__main__":
    main()
