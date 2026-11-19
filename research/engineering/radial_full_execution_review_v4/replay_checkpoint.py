"""Compare two isolated-process qualification checkpoints, not research outcomes."""

import sys
from pathlib import Path

import torch


def compare(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4, msg=path)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            compare(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            compare(x, y, path + "/" + str(i))
    elif isinstance(a, float):
        assert abs(a - b) <= 2e-5 + 2e-4 * abs(b), path
    else:
        assert a == b, path


if __name__ == "__main__":
    files = []
    for root in sys.argv[1:]:
        matches = list(Path(root).glob("fits/*/*/latest.pt"))
        assert len(matches) == 1
        files.append(torch.load(matches[0], weights_only=True, map_location="cpu"))
    # Exact generator state is important even where floating arithmetic admits
    # the predeclared equivalence tolerance.
    for key in ("rng", "sampler"):
        assert torch.equal(files[0][key], files[1][key]), key
    compare(*files)
    print("PASS: model, optimizer, scheduler, RNG, sampler, curve and selection replay")
