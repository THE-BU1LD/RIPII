"""Rebuild this complete recovered-source derivative without its old workstation.

Packaging and syntax validation are not numerical qualification. No provider API,
dependency installation, simulator, model fit, or protected evaluation is invoked.
"""

import argparse
import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

from readiness import DEFAULT_SESSION_SECONDS, identity, verify_inventory

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def payload_files():
    baseline = json.loads((ROOT / "BASELINE_SOURCE_V3.json").read_text())
    scientific = {k: v for k, v in baseline.items() if k.startswith("reference/")}
    verify_inventory(ROOT, scientific)
    actual_scientific = {
        str(p.relative_to(ROOT))
        for p in (ROOT / "reference").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    if actual_scientific != set(scientific):
        raise ValueError("Frozen-reference source membership changed")
    selected = sorted(ROOT.glob("*.py")) + sorted(ROOT.glob("*.md"))
    selected += [p for p in sorted(ROOT.glob("*.json")) if p.name != "SOURCE.json"]
    selected += [ROOT / name for name in sorted(scientific)]
    files = {}
    for path in selected:
        if path.is_symlink():
            raise ValueError("Source symlink refused: " + path.name)
        name = str(path.relative_to(ROOT))
        data = path.read_bytes()
        if path.suffix == ".py":
            ast.parse(data.decode("utf-8"), name)
        files[name] = data
    manifest = {k: sha(v) for k, v in sorted(files.items())}
    files["SOURCE.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    return files, manifest


def build(output):
    files, manifest = payload_files()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name, (2026, 11, 19, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
    payload = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if {n: archive.read(n) for n in archive.namelist()} != files:
            raise ValueError("Source archive roundtrip differs")
    configuration = f'''# Private free Kaggle T4 x2 only. Confirm account, quota and job state first.
MODE = "PREFLIGHT"  # DEVELOPMENT_FULL, RESUME, ANALYZE_ONLY
SESSION_SECONDS = {DEFAULT_SESSION_SECONDS!r}[MODE]
PREVIOUS_RESULTS_ARCHIVE = None  # actual exact-source returned archive, when required
'''
    execution = f'''import base64, hashlib, io, pathlib, sys, zipfile
payload = base64.b64decode({base64.b64encode(payload).decode()!r}, validate=True)
if hashlib.sha256(payload).hexdigest() != {sha(payload)!r}:
    raise ValueError("Embedded source identity differs")
source = pathlib.Path('/kaggle/working/ripii-reference-qualification-v4-source')
source.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for member in archive.infolist():
        path = pathlib.PurePosixPath(member.filename)
        if path.is_absolute() or '..' in path.parts or ((member.external_attr >> 16) & 0o170000 == 0o120000):
            raise ValueError("Unsafe source archive")
    archive.extractall(source)
sys.path.insert(0, str(source))
from notebook_runtime import run
run(source, MODE, SESSION_SECONDS, PREVIOUS_RESULTS_ARCHIVE)
'''
    for code in [configuration, execution]:
        ast.parse(code)

    def cell(kind, name, content):
        result = {"cell_type": kind, "id": name, "metadata": {}, "source": content.splitlines(keepends=True)}
        if kind == "code":
            result.update(execution_count=None, outputs=[])
        return result

    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
        "cells": [
            cell("markdown", "contract", "# RIPII reference qualification review v4\nProspective engineering derivative of the recovered v3 executor. GPU equivalence, resume, dependencies and whole-matrix feasibility remain UNVERIFIED. No scientific fits or positive efficacy result are supplied by this package. Keep PREFLIGHT until its exact-source numerical receipt passes.\n"),
            cell("code", "configuration", configuration),
            cell("code", "execution", execution),
        ],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    notebook_bytes = (json.dumps(notebook, indent=1) + "\n").encode()
    (output / "research_source.zip").write_bytes(payload)
    (output / "KAGGLE_FULL_RESEARCH_RUN.ipynb").write_bytes(notebook_bytes)
    (output / "SOURCE.json").write_bytes(files["SOURCE.json"])
    (ROOT / "SOURCE.json").write_bytes(files["SOURCE.json"])
    receipt = {
        "status": "PASS_STATIC_PORTABILITY",
        "source_files": len(files),
        "source_sha256": identity(manifest),
        "source_zip_sha256": sha(payload),
        "notebook_sha256": sha(notebook_bytes),
        "baseline_source_sha256": "f43e71379037622523200775366ed758de708e6fef2890d2147c0a8dc3606a20",
        "frozen_reference_files_unchanged": len([k for k in manifest if k.startswith("reference/")]),
        "numerical_qualification": "NOT_RUN",
        "kaggle_execution": "NOT_RUN",
        "scientific_result": "NOT_EVALUATED",
        "launch_ready": False,
    }
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    from validate_package import validate

    validation = validate(output)
    (output / "package_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
