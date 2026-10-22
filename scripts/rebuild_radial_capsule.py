"""Reproduce the published capsule without requiring the large predecessor archive.

The original builder additionally audits the locally retained predecessor. This
reproducer checks the frozen successor source and byte-identical portable payload;
it makes no new assertion about access to the historical raw archive.
"""

import argparse
import ast
import base64
import hashlib
import io
import json
import lzma
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/successors/radial_adaptation_v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists")
    frozen = json.loads((STUDY / "FROZEN.json").read_text())
    files = {}
    for name, expected in frozen["files"].items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe source path")
        data = (STUDY / rel).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Frozen source changed: " + name)
        files[name] = data
    files["FROZEN.json"] = (STUDY / "FROZEN.json").read_bytes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    payload = lzma.compress(buffer.getvalue())
    sha = hashlib.sha256(payload).hexdigest()
    notebook = STUDY / "portable" / f"radial-adaptation-v1-{sha[:12]}.ipynb"
    original = json.loads(notebook.read_text())
    code = "".join(
        next(c["source"] for c in original["cells"] if c["cell_type"] == "code")
    )
    tree = ast.parse(code)
    assignment = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "payload" for t in n.targets)
    )
    embedded = base64.b64decode(ast.literal_eval(assignment.value.args[0]))
    if embedded != payload:
        raise ValueError("Published payload differs from reconstructed frozen source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(notebook.read_bytes())
    print(
        json.dumps(
            {
                "status": "PASS",
                "source_files": len(files),
                "payload_sha256": sha,
                "notebook_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "historical_raw_archive_reverified": False,
            }
        )
    )


if __name__ == "__main__":
    main()
