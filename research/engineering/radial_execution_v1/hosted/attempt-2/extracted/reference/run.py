"""Source-bound resource/budget supervisor. No numerical import before admission."""

import argparse
import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid

from control import ROOT, read, sha, write


def verify_source():
    frozen = read(ROOT / "FROZEN.json")
    for relative, digest in frozen["files"].items():
        path = ROOT / relative
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"Frozen source mismatch: {relative}")
    return sha(ROOT / "FROZEN.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from pathlib import Path

    output = Path(args.output).resolve()
    source_sha = verify_source()
    protocol = read(ROOT / "protocol.json")
    if args.worker:
        header = read(output / "header.json")
        if os.environ.get("RADIAL_RUN_TOKEN") != header["token"]:
            raise RuntimeError("Use supervisor")
        if header["source_sha256"] != source_sha:
            raise RuntimeError("Source changed before worker")
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "unittest",
                "discover",
                "-s",
                str(ROOT),
                "-p",
                "test_control.py",
                "-v",
            ],
            cwd=ROOT,
            check=True,
        )
        from engine import Engine

        engine = Engine(
            output,
            protocol,
            time.monotonic() + float(os.environ["RADIAL_SECONDS_LEFT"]),
        )
        engine.qualify()
        engine.profile()
        result = engine.evaluate(engine.train())
        write(output / "worker-complete.json", {"decision": result["decision"]})
        return
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    lock = (ROOT / "active.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    charges = []
    for path in sorted(output.parent.glob("*/charge.json")):
        charges.append(read(path))
    charged = sum(r["charged_host_seconds"] for r in charges)
    left = (
        protocol["host_wall_seconds_cap"] - charged - protocol["export_reserve_seconds"]
    )
    token = uuid.uuid4().hex
    write(
        output / "header.json",
        {
            "token": token,
            "source_sha256": source_sha,
            "protocol_sha256": sha(ROOT / "protocol.json"),
            "prior_charges": charges,
            "seconds_available": left,
        },
    )
    status = {
        "status": "FAILED",
        "scientific_result": "NOT_EVALUATED",
        "test_data_opened": False,
    }
    process = None
    try:
        spec = importlib.util.spec_from_file_location(
            "resource_control",
            ROOT / "vendor/research/successors/simple_graph_v3/control.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        admission = module.admission()
        write(output / "admission.json", admission)
        if admission["status"] != "PASS":
            raise RuntimeError("BLOCKED_RESOURCE")
        if left <= 0:
            raise RuntimeError("BLOCKED_BUDGET")
        env = dict(
            os.environ,
            RADIAL_RUN_TOKEN=token,
            RADIAL_SECONDS_LEFT=str(left),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONHASHSEED="0",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        with (output / "worker.log").open("x") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "run.py"),
                    "--worker",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(
                    timeout=max(1, left - (time.monotonic() - start))
                )
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise TimeoutError("CUMULATIVE_WALL_CAP_REACHED") from None
        if returncode:
            lines = (output / "worker.log").read_text().splitlines()
            raise RuntimeError(lines[-1] if lines else "Worker failed without output")
        result = read(output / "worker-complete.json")
        status.update(status="COMPLETE", scientific_result=result["decision"])
    except Exception as exc:
        status["reason"] = str(exc)
        if "BLOCKED" in str(exc):
            status["status"] = "BLOCKED"
        if "INCONCLUSIVE_UNDERTRAINED" in str(exc):
            status["scientific_result"] = "INCONCLUSIVE_UNDERTRAINED"
        write(
            output / "failure.json",
            {"error": str(exc), "traceback": traceback.format_exc()},
        )
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        status["test_data_opened"] = bool(list((output / "data").glob("*-test-*.npz")))
        write(output / "status.json", status)
        elapsed = time.monotonic() - start
        write(
            output / "charge.json",
            {
                "elapsed_host_seconds": elapsed,
                "export_reserve_seconds": protocol["export_reserve_seconds"],
                "charged_host_seconds": elapsed + protocol["export_reserve_seconds"],
                "charged_gpu_seconds": protocol["charged_device_count"]
                * (elapsed + protocol["export_reserve_seconds"]),
                "cloud_spend_usd": 0,
                "source_sha256": source_sha,
            },
        )
        write(
            output / "manifest.json",
            {
                "files": {
                    str(p.relative_to(output)): sha(p)
                    for p in sorted(output.rglob("*"))
                    if p.is_file() and not p.is_symlink()
                }
            },
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
