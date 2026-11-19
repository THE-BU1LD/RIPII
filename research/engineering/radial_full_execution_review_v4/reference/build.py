"""Freeze and export only allowlisted study source to a private offline notebook."""

import base64
import hashlib
import io
import json
import lzma
import zipfile

from control import ROOT, read, sha, write


def main():
    predecessor = read(ROOT / "PREDECESSOR.json")
    repo = ROOT.parents[2]
    for rel, digest in predecessor["protected_files"].items():
        if sha(repo / rel) != digest:
            raise RuntimeError(f"Predecessor changed: {rel}")
    for rel, digest in predecessor["vendor_files"].items():
        if sha(ROOT / "vendor" / rel) != digest:
            raise RuntimeError(f"Vendor changed: {rel}")
    paths = list(ROOT.glob("*.py")) + [
        ROOT / "protocol.json",
        ROOT / "PROTOCOL.md",
        ROOT / "PREDECESSOR.json",
    ]
    paths += [ROOT / "vendor" / rel for rel in predecessor["vendor_files"]]
    files = {str(p.relative_to(ROOT)): p.read_bytes() for p in paths}
    frozen = {
        "files": {p: hashlib.sha256(b).hexdigest() for p, b in sorted(files.items())},
        "predecessor_preservation": "PASS",
        "historical_archives_exported": False,
    }
    if (ROOT / "FROZEN.json").exists():
        if read(ROOT / "FROZEN.json") != frozen:
            raise RuntimeError("Already frozen; use explicit prelaunch amendment")
    else:
        write(ROOT / "FROZEN.json", frozen)
    files["FROZEN.json"] = (ROOT / "FROZEN.json").read_bytes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    compressed = lzma.compress(buffer.getvalue())
    digest = hashlib.sha256(compressed).hexdigest()
    code = f"""import base64, hashlib, io, json, lzma, os, pathlib, shutil, subprocess, sys, zipfile
payload=base64.b64decode({base64.b64encode(compressed).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={digest!r}
root=pathlib.Path('/kaggle/working/radial-adaptation-v1-source')
root.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(lzma.decompress(payload))) as archive:
    for info in archive.infolist():
        rel=pathlib.PurePosixPath(info.filename)
        assert not rel.is_absolute() and '..' not in rel.parts
        dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(archive.read(info))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
try:
    subprocess.run([sys.executable,str(root/'run.py'),'--output',str(root/'execution'/'attempt-1')],cwd=root,env=env,check=True)
    with (root/'independent-verification.json').open('x') as verification:
        subprocess.run([sys.executable,str(root/'verify.py'),str(root/'execution'/'attempt-1'),'--numeric'],cwd=root,env=env,stdout=verification,check=True)
finally:
    shutil.make_archive('/kaggle/working/radial-adaptation-v1-results','zip',root)
    result=root/'execution/attempt-1/status.json'
    if result.exists():print(result.read_text())
    profile=root/'execution/attempt-1/profile.json'
    if profile.exists():print(profile.read_text())
    log=root/'execution/attempt-1/worker.log'
    if log.exists():print(log.read_text()[-12000:])
    print('All artifacts retained in radial-adaptation-v1-results.zip')
"""
    compile(code, "notebook", "exec")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [
            {
                "id": "protocol",
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# RIPII Radial Adaptation v1\nPrivate free GPU comparison. Frozen v3 preserved. Qualification and full-budget feasibility precede training. Test data stays closed unless all fits qualify."
                ],
            },
            {
                "id": "run",
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
        ],
    }
    out = ROOT / "portable"
    out.mkdir(exist_ok=True)
    target = out / f"radial-adaptation-v1-{digest[:12]}.ipynb"
    if not target.exists():
        write(target, notebook)
    with zipfile.ZipFile(io.BytesIO(lzma.decompress(compressed))) as archive:
        assert set(archive.namelist()) == set(files)
        for name, data in files.items():
            assert archive.read(name) == data
    receipt = {
        "notebook": str(target),
        "notebook_sha256": sha(target),
        "payload_sha256": digest,
        "source_sha256": sha(ROOT / "FROZEN.json"),
        "bytes": target.stat().st_size,
        "files": len(files),
        "roundtrip": "PASS",
        "secrets_included": False,
    }
    if not (out / "receipt.json").exists():
        write(out / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
