"""Offline notebook runtime. Credentials, provider submission, and paid compute absent."""

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from contextlib import suppress
from pathlib import Path

from readiness import (
    admit_full,
    digest,
    identity,
    safe_relative,
    session_reservation,
    verify_inventory,
    worker_environment,
)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def unpack(path, destination):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive paths")
        if sum(i.file_size for i in z.infolist()) > 20 * 1024**3:
            raise ValueError("Archive exceeds 20 GiB")
        for info in z.infolist():
            safe_relative(info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlink archive member")
        z.extractall(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    verify_inventory(destination, manifest["files"])
    actual = {
        str(p.relative_to(destination))
        for p in destination.rglob("*")
        if p.is_file() and p != destination / "manifest.json"
    }
    expected = set(manifest["files"])
    if actual != expected:
        raise ValueError("Result inventory differs")


def run(source, mode, seconds, previous_archive=None):
    if mode not in {
        "PREFLIGHT",
        "DEVELOPMENT_FULL",
        "RESUME",
        "ANALYZE_ONLY",
        "CONFIRMATORY_FROZEN",
    }:
        raise ValueError("Unknown mode")
    if mode == "CONFIRMATORY_FROZEN":
        raise RuntimeError("BLOCKED: no independently frozen confirmatory study exists")
    if type(seconds) is not int or seconds < 120:
        raise ValueError("SESSION_SECONDS must be an integer >=120")
    source = Path(source).resolve()
    inventory = json.loads((source / "SOURCE.json").read_text())
    verify_inventory(source, inventory)
    fingerprint = identity(inventory)
    output = Path("/kaggle/working/ripii-execution-v3")
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    prior = 3794.580382916
    previous = None
    if previous_archive is not None:
        previous = output / "prior"
        unpack(Path(previous_archive), previous)
        old = json.loads((previous / "run_manifest.json").read_text())
        if old["source_sha256"] != fingerprint:
            raise ValueError("Previous source differs")
        prior = json.loads((previous / "wrapper-status.json").read_text())[
            "cumulative_ceiling_gpu_seconds"
        ]
    budget = (
        {
            "prior": prior,
            "reserved_gpu_seconds": 0,
            "cumulative_ceiling_gpu_seconds": prior,
            "cap": 64800,
        }
        if mode == "ANALYZE_ONLY"
        else session_reservation(prior, seconds, mode)
    )
    write(
        output / "run_manifest.json",
        {
            "source_sha256": fingerprint,
            "mode": mode,
            "budget": budget,
            "python": sys.version,
            "platform": platform.platform(),
            "attached_assets": previous_archive,
            "quota": "Not observable without credentials; launch owner must check live free quota",
            "stage": "ENGINEERING_PREFLIGHT"
            if mode == "PREFLIGHT"
            else "FROZEN_DEVELOPMENT",
            "protected_confirmation": "NOT_AUTHORIZED_OR_DEFINED",
        },
    )
    status = {
        "status": "FAILED",
        "scientific_result": "NOT_EVALUATED",
        "test_data_opened": False,
        **budget,
    }
    timed_out = False
    try:
        # Read-only host admission before any tensor import.
        samples = []
        for i in range(2):
            mem = {
                x.split(":")[0]: int(x.split()[1]) * 1024
                for x in Path("/proc/meminfo").read_text().splitlines()
            }
            vm = dict(x.split() for x in Path("/proc/vmstat").read_text().splitlines())
            samples.append(
                {
                    "available": mem["MemAvailable"],
                    "swapin": int(vm["pswpin"]),
                    "swapout": int(vm["pswpout"]),
                }
            )
            if not i:
                time.sleep(3)
        write(
            output / "admission.json",
            {"samples": samples, "disk": shutil.disk_usage(output).free},
        )
        if (
            min(x["available"] for x in samples) < 512 * 1024**2
            or samples[0]["swapin"] != samples[1]["swapin"]
            or samples[0]["swapout"] != samples[1]["swapout"]
        ):
            raise RuntimeError("BLOCKED_RESOURCE")
        if shutil.disk_usage(output).free < 10 * 1024**3:
            raise RuntimeError("BLOCKED_DISK")
        if mode != "ANALYZE_ONLY":
            gpu = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            write(
                output / "hardware.json",
                {
                    "returncode": gpu.returncode,
                    "name_total_free_vram_mib": gpu.stdout,
                    "stderr": gpu.stderr,
                },
            )
            print("GPU name,total MiB,free MiB:\n" + gpu.stdout, flush=True)
            if mode != "ANALYZE_ONLY" and (
                gpu.returncode or len(gpu.stdout.strip().splitlines()) != 2
            ):
                raise RuntimeError(
                    "This frozen four-worker executor requires two CUDA GPUs; no CPU fallback"
                )
        if mode == "ANALYZE_ONLY":
            if previous is None:
                raise ValueError("ANALYZE_ONLY needs previous archive")
            command = [
                sys.executable,
                str(source / "aggregate_results.py"),
                str(previous),
                str(output / "analysis"),
            ]
        elif mode == "PREFLIGHT":
            command = [
                sys.executable,
                str(source / "qualify.py"),
                "--output",
                str(output / "qualification"),
            ]
        else:
            if previous is None:
                raise ValueError(
                    "Full or resume requires exact-source qualification archive"
                )
            if mode == "DEVELOPMENT_FULL":
                old_manifest = json.loads((previous / "run_manifest.json").read_text())
                old_status = json.loads((previous / "wrapper-status.json").read_text())
                if (
                    old_manifest["mode"] != "PREFLIGHT"
                    or old_status["status"] != "PREFLIGHT_COMPLETE"
                ):
                    raise ValueError(
                        "A new full run requires its completed preflight; terminal research cannot be relaunched"
                    )
            qualifications = list(previous.rglob("qualification/qualification.json"))
            if len(qualifications) != 1:
                raise ValueError("Unique passed qualification is required")
            qual = qualifications[0]
            admit_full(fingerprint, json.loads(qual.read_text()))
            command = [
                sys.executable,
                str(source / "orchestrate.py"),
                "--output",
                str(output / "execution"),
                "--seconds",
                str(seconds - 60),
                "--qualification",
                str(qual),
            ]
            if mode == "RESUME":
                if not (previous / "execution").is_dir():
                    raise ValueError("Missing interrupted execution")
                command += ["--resume", str(previous / "execution")]
        print(
            json.dumps(
                {
                    "stage": mode,
                    "command": command,
                    "budget": budget,
                    "output": str(output),
                },
                indent=2,
            ),
            flush=True,
        )
        with (output / "run.log").open("x") as log:
            proc = subprocess.Popen(
                command,
                cwd=source,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=worker_environment(),
            )
            try:
                code = proc.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                raise TimeoutError("SESSION_CAP_REACHED_ALL_CHILDREN_STOPPED") from None
        status["exit_code"] = code
        if code:
            raise RuntimeError(
                "FAILED: see run.log and qualification process.log files"
            )
        final = output / "execution/status.json"
        if final.exists():
            status.update(json.loads(final.read_text()))
        else:
            status["status"] = (
                "PREFLIGHT_COMPLETE" if mode == "PREFLIGHT" else "ANALYSIS_COMPLETE"
            )
    except Exception as exc:
        status["reason"] = str(exc)
        write(output / "failures.json", {"reason": str(exc)})
    finally:
        status["test_data_opened"] = bool(
            list(output.glob("execution/data/*-test-*.npz"))
        )
        execution = output / "execution"
        terminal = execution / "status.json"
        if terminal.exists():
            status["execution_status"] = json.loads(terminal.read_text())
        elif execution.is_dir() and timed_out:
            write(
                terminal,
                {
                    "status": "INTERRUPTED",
                    "scientific_result": "NOT_EVALUATED",
                    "test_data_opened": status["test_data_opened"],
                    "resume_allowed": not status["test_data_opened"],
                    "reason": "SESSION_CAP_REACHED_ALL_CHILDREN_STOPPED",
                },
            )
        if execution.is_dir() and not (execution / "manifest.json").exists():
            write(
                execution / "manifest.json",
                {
                    "files": {
                        str(p.relative_to(execution)): digest(p)
                        for p in sorted(execution.rglob("*"))
                        if p.is_file() and not p.is_symlink()
                    }
                },
            )
        status["elapsed_seconds"] = time.monotonic() - started
        status["charge_policy"] = (
            "Worst-case full reservation retained, including interrupted attempts; no automatic refund"
        )
        write(output / "wrapper-status.json", status)
        shutil.copytree(
            source, output / "source", ignore=shutil.ignore_patterns("__pycache__")
        )
        from export_contract import emit

        emit(output, status)
        files = {
            str(p.relative_to(output)): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and not p.is_symlink()
        }
        write(output / "manifest.json", {"files": files})
        archive = Path("/kaggle/working/ripii_execution_v3_results.zip")
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            for p in sorted(output.rglob("*")):
                if p.is_file() and not p.is_symlink():
                    z.write(p, p.relative_to(output))
        print(json.dumps(status, indent=2), flush=True)
        print("RESULT_ARCHIVE", archive, "SHA256", digest(archive), flush=True)
    if status["status"] == "FAILED":
        raise RuntimeError(status.get("reason", "Failed"))
