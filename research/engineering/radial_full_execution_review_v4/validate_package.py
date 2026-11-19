"""Static portable-package validation. Does not claim CUDA or dependency qualification."""

import argparse
import ast
import base64
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

from readiness import identity, safe_relative


def validate(folder):
    folder = Path(folder)
    payload = (folder / "research_source.zip").read_bytes()
    notebook = json.loads((folder / "KAGGLE_FULL_RESEARCH_RUN.ipynb").read_text())
    if (
        notebook["nbformat"] != 4
        or notebook["metadata"]["kernelspec"]["name"] != "python3"
    ):
        raise ValueError("Notebook format")
    if [c["id"] for c in notebook["cells"]] != [
        "contract",
        "configuration",
        "execution",
    ]:
        raise ValueError("Cell order")
    trees = []
    for c in notebook["cells"]:
        if c["cell_type"] == "code":
            trees.append(ast.parse("".join(c["source"])))
    embedded = []
    for tree in trees:
        for n in ast.walk(tree):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "b64decode"
            ):
                embedded.append(
                    base64.b64decode(ast.literal_eval(n.args[0]), validate=True)
                )
    if embedded != [payload]:
        raise ValueError("Embedded source differs from archive")
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicates")
        for name in names:
            safe_relative(name)
        manifest = json.loads(z.read("SOURCE.json"))
        if set(names) != set(manifest) | {"SOURCE.json"}:
            raise ValueError("Unmanifested source")
        imports = set()
        parsed = 0
        for name, expected in manifest.items():
            data = z.read(name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("Member mismatch " + name)
            if name.endswith(".py"):
                tree = ast.parse(data.decode(), name)
                parsed += 1
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.update(n.name.split(".")[0] for n in node.names)
                    if (
                        isinstance(node, ast.ImportFrom)
                        and not node.level
                        and node.module
                    ):
                        imports.add(node.module.split(".")[0])
        external = sorted(
            imports
            - set(sys.stdlib_module_names)
            - {Path(n).stem for n in names if n.endswith(".py")}
            - {"ripii"}
        )
    receipt = json.loads((folder / "build.json").read_text())
    if receipt["source_zip_sha256"] != hashlib.sha256(payload).hexdigest():
        raise ValueError("Build receipt ZIP mismatch")
    if (
        receipt["notebook_sha256"]
        != hashlib.sha256(
            (folder / "KAGGLE_FULL_RESEARCH_RUN.ipynb").read_bytes()
        ).hexdigest()
    ):
        raise ValueError("Build receipt notebook mismatch")
    return {
        "status": "PASS_STATIC_PORTABILITY",
        "python_files_parsed": parsed,
        "archive_members": len(names),
        "source_sha256": identity(manifest),
        "external_import_roots": external,
        "numerical_execution": "NOT_RUN",
        "dependencies_resolved": "NOT_VERIFIED_IN_CURRENT_KAGGLE_ENVIRONMENT",
        "manual_full_launch_ready": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = validate(a.folder)
    if a.output:
        a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
