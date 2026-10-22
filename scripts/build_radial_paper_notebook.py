"""Build the evidence report on free Kaggle CPU; pin the TeX binary by digest."""

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/radial_adaptation_20261021"


def main():
    names = [
        "main.tex",
        "arxiv.sty",
        "observed.tex",
        "optimization.tex",
        "STYLE_PROVENANCE.json",
    ]
    files = {name: (PAPER / name).read_bytes() for name in names}
    initial = json.loads(
        (
            ROOT
            / "research/successors/radial_adaptation_v1/hosted/kaggle/v1/extracted/execution/attempt-1/profile.json"
        ).read_text()
    )
    final = json.loads(
        (
            ROOT
            / "research/engineering/radial_execution_v1/hosted/attempt-2/extracted/results/probe.json"
        ).read_text()
    )
    plot = {
        "models": [r["model"] for r in initial["rows"]],
        "initial": [r["step_seconds_max"] for r in initial["rows"]],
        "optimized": [r["profile"]["captured_update_seconds_max"] for r in final],
        "boundary": "Descriptive timings from separate Kaggle T4 runs; no efficacy or paired speedup inference.",
    }
    files["timings.json"] = json.dumps(plot, indent=2).encode()
    source = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    files["SOURCE.json"] = json.dumps(source, indent=2).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    payload = buffer.getvalue()
    sha = hashlib.sha256(payload).hexdigest()
    wrapper = f"""import base64, hashlib, io, json, os, subprocess, sys, tarfile, time, urllib.request, zipfile, shutil
from pathlib import Path
start=time.monotonic()
root=Path('/kaggle/working/radial-paper');root.mkdir(exist_ok=False);os.chdir(root)
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={sha!r}
with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)
status={{'status':'STARTED','gpu_used':False,'cloud_spend_usd':0}}
try:
    for name,digest in json.loads((root/'SOURCE.json').read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    samples=[]
    for i in range(2):
        mem={{x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()}}
        vm=dict(x.split() for x in Path('/proc/vmstat').read_text().splitlines())
        samples.append({{'available':mem['MemAvailable'],'swapin':vm['pswpin'],'swapout':vm['pswpout']}})
        if i==0:time.sleep(1)
    (root/'admission.json').write_text(json.dumps(samples))
    assert min(s['available'] for s in samples)>=512*1024**2
    assert samples[0]['swapin']==samples[1]['swapin'] and samples[0]['swapout']==samples[1]['swapout']
    url='https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz'
    with urllib.request.urlopen(url,timeout=60) as response: binary=response.read()
    assert hashlib.sha256(binary).hexdigest()=='8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7'
    with tarfile.open(fileobj=io.BytesIO(binary),mode='r:gz') as archive:
        member=next(x for x in archive.getmembers() if Path(x.name).name=='tectonic' and x.isfile())
        (root/'tectonic').write_bytes(archive.extractfile(member).read())
    (root/'tectonic').chmod(0o700)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=json.loads((root/'timings.json').read_text())
    fig,ax=plt.subplots(figsize=(6.5,2.65),layout='constrained')
    x=list(range(len(d['models'])))
    ax.bar([i-.18 for i in x],d['initial'],width=.36,label='Initial eager profile',color='#63758b')
    ax.bar([i+.18 for i in x],d['optimized'],width=.36,label='Qualified capture fixture',color='#007e80')
    ax.set_xticks(x,['Adaptive radial','Fixed radial','Graph','Pooling','Transformer'])
    ax.tick_params(axis='x',labelsize=8)
    ax.set_ylabel('Seconds per update')
    ax.legend(frameon=False,fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(root/'timing.pdf');fig.savefig(root/'timing.png',dpi=180);plt.close(fig)
    with (root/'build.log').open('w') as log:
        result=subprocess.run([str(root/'tectonic'),'main.tex','--keep-logs','--keep-intermediates'],stdout=log,stderr=subprocess.STDOUT,timeout=700)
    assert result.returncode==0, (root/'build.log').read_text()[-4000:]
    if shutil.which('pdftoppm'):
        subprocess.run(['pdftoppm','-r','65','-png','main.pdf','page'],check=True,timeout=120)
        if shutil.which('pdftotext'):subprocess.run(['pdftotext','-layout','main.pdf','paper-text.txt'],check=True,timeout=30)
    else:
        try:import fitz
        except ImportError:
            subprocess.run([sys.executable,'-m','pip','install','pymupdf==1.26.4'],check=True,timeout=180)
            import fitz
        doc=fitz.open('main.pdf')
        (root/'paper-text.txt').write_text('\\n'.join(p.get_text() for p in doc))
        for i,page in enumerate(doc):page.get_pixmap(matrix=fitz.Matrix(.9,.9)).save(root/f'page-{{i+1:02}}.png')
    status.update(status='COMPILED_AND_RENDERED',pages=len(list(root.glob('page-*.png'))))
except Exception as exc:
    status.update(status='FAILED',error=str(exc))
finally:
    status['elapsed_cpu_host_seconds']=time.monotonic()-start
    (root/'status.json').write_text(json.dumps(status,indent=2))
    manifest={{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file() and p.name!='tectonic'}}
    (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile('/kaggle/working/radial-paper-results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in [*manifest,'MANIFEST.json']:archive.write(root/name,name)
    print(json.dumps(status))
    if (root/'build.log').exists():print((root/'build.log').read_text()[-3000:])
"""
    compile(wrapper, "paper-wrapper", "exec")
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
                "id": "radial-paper-build",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": wrapper.splitlines(keepends=True),
            }
        ],
    }
    out = PAPER / "portable"
    out.mkdir(exist_ok=True)
    path = out / f"paper-{sha[:12]}.ipynb"
    path.write_text(json.dumps(notebook))
    (out / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
    receipt = {
        "payload_sha256": sha,
        "notebook": str(path),
        "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "gpu_used": False,
        "cloud_spend_usd": 0,
    }
    (out / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
