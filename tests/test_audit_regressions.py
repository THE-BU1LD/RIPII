from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ripii.data.synthetic import SyntheticStructuralDataset
from ripii.models.factory import build_model
from ripii.utils.ablation import apply_mode
from ripii.utils.config import load_config
from ripii.utils.metrics import effective_rank
from ripii.utils.training import load_checkpoint


def test_disabled_geometry_has_no_uncertainty_gradient():
    cfg = apply_mode(load_config("configs/mechanism_smoke.yaml"), "no_renorm")
    model = build_model(cfg)
    batch = {
        "x": torch.randn(4, cfg.input_dim),
        "x_view": torch.randn(4, cfg.input_dim),
        "transform": torch.randn(4, 4),
    }
    losses = model.losses(batch, vars(cfg.loss_weights))
    losses["total"].backward()
    assert (
        model.loss_balancer.log_vars.grad[model.loss_balancer.keys.index("geom")] == 0
    )


def test_zero_variance_effective_rank_and_identity_transform():
    assert effective_rank(torch.ones(8, 4)) == 0
    ds = SyntheticStructuralDataset(8, 24, 0, 0.2, 3, num_classes=4)
    x = ds[0].x
    assert torch.equal(ds._transform(x, torch.zeros(4)), x)


def test_legacy_training_resume_and_ablation_pipeline(tmp_path: Path):
    def run(*args, check=True):
        return subprocess.run(
            [sys.executable, *args], capture_output=True, text=True, check=check
        )

    config = "configs/mechanism_smoke.yaml"
    full, split = tmp_path / "full", tmp_path / "split"
    run(
        "scripts/train.py",
        "--config",
        config,
        "--steps",
        "4",
        "--output-dir",
        str(full),
    )
    run(
        "scripts/train.py",
        "--config",
        config,
        "--steps",
        "2",
        "--output-dir",
        str(split),
    )
    run(
        "scripts/train.py",
        "--config",
        config,
        "--steps",
        "4",
        "--output-dir",
        str(split),
        "--resume",
        str(split / "latest.pt"),
    )
    a = torch.load(full / "final.pt", weights_only=True)
    b = torch.load(split / "final.pt", weights_only=True)
    assert a["step"] == b["step"] == 4
    assert all(torch.equal(value, b["model"][key]) for key, value in a["model"].items())
    done = run(
        "scripts/train.py",
        "--config",
        config,
        "--steps",
        "4",
        "--output-dir",
        str(split),
        "--resume",
        str(split / "final.pt"),
        check=False,
    )
    assert "already completed" in done.stderr
    run(
        "scripts/run_pipeline.py",
        "--config",
        config,
        "--mode",
        "no_vq",
        "--output-dir",
        str(tmp_path / "pipeline"),
    )
    assert (tmp_path / "pipeline/eval.json").is_file()


def test_epoch_metrics_weight_samples_including_short_batch():
    from ripii.utils.training import run_epoch

    class MeanModel(torch.nn.Module):
        def losses(self, batch, weights, warmup=False):
            return {"total": batch["x"].mean(), "recon": batch["x"].mean()}

    cfg = load_config("configs/mechanism_smoke.yaml")
    batches = [{"x": torch.zeros(4, 1)}, {"x": torch.ones(1, 1)}]
    metrics = run_epoch(MeanModel(), batches, None, cfg, "cpu", False)
    assert abs(metrics["recon"] - 0.2) < 1e-7


def test_pre_correction_checkpoint_is_not_reinterpreted(tmp_path: Path):
    cfg = load_config("configs/mechanism_smoke.yaml")
    model = build_model(cfg)
    path = tmp_path / "old.pt"
    torch.save({"model": model.state_dict(), "checkpoint_version": 2}, path)
    with pytest.raises(ValueError, match="predates the 0.2 mechanism correction"):
        load_checkpoint(path, model)
