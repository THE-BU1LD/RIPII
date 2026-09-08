from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ripii.utils.config import load_config, runtime_profile
from ripii.utils.metrics import heldout_ridge_probe_accuracy, mse
from ripii.utils.training import build_data_splits, load_compatible_initial_state

ROOT = Path(__file__).resolve().parents[1]


def test_unknown_config_key_fails_closed(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("steps: 1\nstepz: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown configuration keys: stepz"):
        load_config(config)


def test_invalid_transform_contract_fails_closed(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("transform_dim: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="transform_dim"):
        load_config(config)


def test_eval_latent_is_deterministic_but_training_latent_is_stochastic() -> None:
    from ripii.models.layers import StochasticLatentHead

    head = StochasticLatentHead(4, 4)
    inputs = torch.ones(2, 4)
    head.eval()
    first = head(inputs)[0]
    second = head(inputs)[0]
    assert torch.equal(first, second)
    head.train()
    first = head(inputs)[0]
    second = head(inputs)[0]
    assert not torch.equal(first, second)


def test_no_equiv_is_distinct_from_no_action() -> None:
    from ripii.utils.ablation import apply_mode
    from ripii.utils.config import Config

    cfg = Config()
    no_equiv = apply_mode(cfg, "no_equiv")
    no_action = apply_mode(cfg, "no_action")
    assert no_equiv.use_action
    assert no_equiv.loss_weights.equiv == 0.0
    assert not no_action.use_action


def test_no_structured_disables_all_claimed_mechanisms() -> None:
    from ripii.utils.ablation import apply_mode
    from ripii.utils.config import Config

    cfg = apply_mode(Config(), "no_structured")
    assert not any(
        (cfg.use_projective, cfg.use_graph, cfg.use_quantizer, cfg.use_action)
    )
    assert cfg.num_levels == 0
    assert cfg.graph_steps == 0


def test_profiles_make_disabled_mechanisms_explicit() -> None:
    plumbing = runtime_profile(load_config(ROOT / "configs" / "smoke.yaml"))
    mechanism = runtime_profile(load_config(ROOT / "configs" / "mechanism_smoke.yaml"))
    assert plumbing.profile == "plumbing_smoke"
    assert not any(
        (
            plumbing.use_projective,
            plumbing.use_graph,
            plumbing.use_quantizer,
            plumbing.use_action,
        )
    )
    assert mechanism.profile == "mechanism_smoke"
    assert all(
        (
            mechanism.use_projective,
            mechanism.use_graph,
            mechanism.use_quantizer,
            mechanism.use_action,
        )
    )
    assert mechanism.num_levels > 0
    assert mechanism.graph_steps > 0


def test_nonfinite_metrics_fail_instead_of_being_hidden() -> None:
    with pytest.raises(FloatingPointError, match="non-finite"):
        mse(torch.tensor([float("nan")]), torch.zeros(1))


def test_train_validation_test_indices_are_disjoint() -> None:
    cfg = load_config(ROOT / "configs" / "mechanism_smoke.yaml")
    train, validation, test = build_data_splits(cfg)
    train_ids = set(train.dataset.indices)
    validation_ids = set(validation.dataset.indices)
    test_ids = set(test.dataset.indices)
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert train_ids | validation_ids | test_ids == set(range(cfg.dataset_size))


def test_heldout_probe_scores_only_test_examples() -> None:
    train_x = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    train_y = torch.tensor([0, 0, 1, 1])
    test_x = torch.tensor([[-3.0], [3.0]])
    test_y = torch.tensor([0, 1])
    assert heldout_ridge_probe_accuracy(train_x, train_y, test_x, test_y, 2) == 1.0


def test_compatible_initial_state_reuses_shared_tensors(tmp_path: Path) -> None:
    from ripii.models.layers import StochasticLatentHead

    source = StochasticLatentHead(4, 4)
    target = StochasticLatentHead(4, 4)
    initial_state = tmp_path / "initial.pt"
    torch.save({"model": source.state_dict()}, initial_state)
    receipt = load_compatible_initial_state(initial_state, target)
    assert receipt["compatible_tensor_count"] == len(source.state_dict())
    assert all(
        torch.equal(source.state_dict()[key], target.state_dict()[key])
        for key in source.state_dict()
    )


def test_legacy_reconstruction_weight_is_mapped_explicitly(tmp_path: Path) -> None:
    config = tmp_path / "legacy.yaml"
    config.write_text("loss_weights:\n  rec: 0.25\n", encoding="utf-8")
    cfg = load_config(config)
    assert cfg.loss_weights.recon == 0.25
    assert "rec" not in cfg.as_dict()["loss_weights"]


def test_conflicting_reconstruction_weight_names_fail_closed(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        "loss_weights:\n  rec: 0.25\n  recon: 0.5\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="both rec and recon"):
        load_config(config)


def test_zero_weight_loss_has_no_balancer_gradient() -> None:
    from ripii.utils.loss_balancer import AdaptiveLossBalancer

    balancer = AdaptiveLossBalancer(["active", "disabled"])
    losses = {
        "active": torch.tensor(2.0, requires_grad=True),
        "disabled": torch.tensor(3.0, requires_grad=True),
    }
    total, terms = balancer(losses, {"active": 1.0, "disabled": 0.0})
    total.backward()
    assert terms["balanced_disabled"].item() == 0.0
    assert balancer.log_vars.grad is not None
    assert balancer.log_vars.grad[1].item() == 0.0
    assert losses["disabled"].grad is None


def test_warmup_disables_balancer_offsets_for_inactive_losses() -> None:
    from ripii.utils.config import Config

    cfg = Config(
        input_dim=16,
        latent_dim=8,
        hidden_dim=16,
        node_dim=4,
        num_nodes=3,
        num_levels=1,
        num_projectors=1,
        codebook_size=4,
        fine_codebook_size=4,
        codebook_dim=4,
        graph_steps=1,
        graph_topk=1,
        dataset_size=8,
        num_classes=3,
    )
    model = __import__("ripii.models.factory", fromlist=["build_model"]).build_model(cfg)
    batch = {
        "x": torch.randn(4, cfg.input_dim),
        "x_view": torch.randn(4, cfg.input_dim),
        "transform": torch.randn(4, cfg.transform_dim),
    }
    losses = model.losses(batch, vars(cfg.loss_weights), warmup=True)
    for key in (
        "equiv",
        "inv",
        "scale",
        "proj",
        "spectral",
        "geom",
        "vq",
        "node",
        "moment",
        "identity",
        "depth",
    ):
        assert losses[f"balanced_{key}"].item() == 0.0


def test_disabled_modules_are_not_reported_as_trainable() -> None:
    from ripii.models.factory import build_model
    from ripii.utils.ablation import apply_mode
    from ripii.utils.config import Config

    model = build_model(apply_mode(Config(), "no_structured"))
    assert not any(parameter.requires_grad for parameter in model.action.parameters())
    assert not any(parameter.requires_grad for parameter in model.graph.parameters())
    assert not any(parameter.requires_grad for parameter in model.quantizer.parameters())
