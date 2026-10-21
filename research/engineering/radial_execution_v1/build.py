"""Package only reviewed source; carry prior charge into the bounded probe."""

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parents[1] / "successors/radial_adaptation_v1"


def main():
    state = json.loads((ROOT / "STATE.json").read_text())
    budget = state["budget"]
    assert budget["prior_charged_gpu_seconds"] + 2 * (1200 + 180) < 4 * 3600
    frozen = json.loads((REFERENCE / "FROZEN.json").read_text())
    files = {}
    for name, digest in frozen["files"].items():
        data = (REFERENCE / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
        files["reference/" + name] = data
    files["reference/FROZEN.json"] = (REFERENCE / "FROZEN.json").read_bytes()
    for name in ("optimized.py", "probe.py", "STATE.json"):
        files[name] = (ROOT / name).read_bytes()
    manifest = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())}
    files["SOURCE.json"] = json.dumps(manifest, indent=2).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_LZMA) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_LZMA
            archive.writestr(info, data)
    payload = buf.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    wrapper = f"""import base64, hashlib, io, json, os, subprocess, sys, time, zipfile
from pathlib import Path
started = time.monotonic()
root = Path('/kaggle/working/radial-execution-v1')
root.mkdir(exist_ok=False)
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(root)
for name, digest in json.loads((root/'SOURCE.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
os.chdir(root)
status = {{'status':'RUNNING'}}
try:
    # Linux resource admission before importing the numerical runtime.
    samples=[]
    for i in range(2):
        memory={{line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}}
        vm=dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
        samples.append({{'available':memory['MemAvailable'],'pswpin':int(vm['pswpin']),'pswpout':int(vm['pswpout'])}})
        if i==0:time.sleep(1)
    (root/'admission.json').write_text(json.dumps(samples,indent=2))
    assert min(s['available'] for s in samples)>=512*1024**2
    assert samples[1]['pswpin']==samples[0]['pswpin'] and samples[1]['pswpout']==samples[0]['pswpout']
    with (root/'worker.log').open('w') as log:
        run=subprocess.run([sys.executable,'probe.py'],stdout=log,stderr=subprocess.STDOUT,timeout=max(1,1200-(time.monotonic()-started)))
    status={{'status':'COMPLETE' if run.returncode==0 else 'FAILED','returncode':run.returncode}}
except Exception as exc:
    status={{'status':'FAILED','error':str(exc)}}
finally:
    elapsed=time.monotonic()-started
    charge={{'elapsed_host_seconds':elapsed,'reserved_export_seconds':180,'charged_devices':2,'charged_gpu_seconds':2*(elapsed+180),'prior_charged_gpu_seconds':{budget["prior_charged_gpu_seconds"]},'cumulative_charged_gpu_seconds':{budget["prior_charged_gpu_seconds"]}+2*(elapsed+180),'cloud_spend_usd':0}}
    (root/'charge.json').write_text(json.dumps(charge,indent=2))
    (root/'status.json').write_text(json.dumps(status,indent=2))
    manifest={{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}}
    (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile('/kaggle/working/radial-execution-v1-results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted([*manifest,'MANIFEST.json']):archive.write(root/name,name)
    print(json.dumps(status));print(json.dumps(charge))
    if (root/'results/summary.json').exists():print((root/'results/summary.json').read_text())
    if (root/'worker.log').exists():print((root/'worker.log').read_text()[-16000:])
"""
    compile(wrapper, "kaggle-wrapper", "exec")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": wrapper.splitlines(keepends=True),
            }
        ],
    }
    out = ROOT / "portable"
    out.mkdir(exist_ok=True)
    path = out / f"radial-execution-v1-{digest[:12]}.ipynb"
    path.write_text(json.dumps(notebook))
    receipt = {
        "payload_sha256": digest,
        "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "notebook": str(path),
        "source_files": len(files),
        "budget": budget,
    }
    (out / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (out / "SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
