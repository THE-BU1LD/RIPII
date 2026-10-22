"""Numerical engine; imported only after resource and source admission."""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from control import ROOT, adequate, read, sha, summarize, write

VENDOR = ROOT / "vendor"
sys.path.insert(0, str(VENDOR))
MODEL_PATH = VENDOR / "research/successors/simple_graph_v3"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


models = load("models", MODEL_PATH / "models.py")
old_control = load("frozen_v3_control", MODEL_PATH / "control.py")
simulators = load(
    "pinned_nri",
    VENDOR / "research/successors/simple_graph_v1/upstream/data/synthetic_sim.py",
)


class Engine:
    def __init__(self, output, protocol, deadline):
        self.out, self.p, self.deadline = Path(output), protocol, deadline
        self.device = torch.device("cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_REQUIRED_NO_CPU_FALLBACK")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.widths = models.matched_hidden_sizes(
            protocol["context"], protocol["candidate_hidden"]
        )
        self.last_guard = 0.0
        write(
            self.out / "environment.json",
            {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": sys.version,
                "cuda": torch.version.cuda,
                "devices": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "widths": self.widths,
                "deterministic": True,
                "tf32": False,
                "threads": 1,
            },
        )

    def check(self):
        if time.monotonic() > self.deadline:
            raise TimeoutError("CUMULATIVE_WALL_CAP_REACHED")
        if time.monotonic() - self.last_guard > 120:
            sample = old_control.sample()
            prior = getattr(self, "last_sample", None)
            if prior:
                assessed = old_control.assess([prior, sample])
                if assessed["status"] != "PASS":
                    write(
                        self.out / f"resource-refusal-{time.time_ns()}.json", assessed
                    )
                    raise RuntimeError("RESOURCE_PRESSURE_DURING_RUN")
            elif sample.get("errors") or sample.get("free", 0) < 512 * 1024**2:
                raise RuntimeError("RESOURCE_ADMISSION_FAILED")
            self.last_sample, self.last_guard = sample, time.monotonic()

    def model(self, name, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return models.make_model(name, self.widths[name], self.p["context"]).to(
            self.device
        )

    def data(self, domain, block, split, regime="standard"):
        self.check()
        target = self.out / "data" / f"{domain}-{block}-{split}-{regime}.npz"
        if target.exists():
            meta = read(target.with_suffix(".json"))
            if meta["sha256"] != sha(target):
                raise RuntimeError("Dataset changed")
            with np.load(target, allow_pickle=False) as archive:
                return torch.from_numpy(archive["states"].copy())
        seed = old_control.dataset_seed(domain, block, split, regime)
        np.random.seed(seed)
        cls = (
            simulators.SpringSim
            if domain == "springs"
            else simulators.ChargedParticlesSim
        )
        params = old_control.simulator_parameters(domain, regime, 5)
        simulator = cls(**params)
        count = self.p[
            {
                "train": "train_scenes",
                "validation": "validation_scenes",
                "test": "test_scenes",
            }[split]
        ]
        length = self.p["context"] + self.p["forecast"]
        values, edges = [], []
        started = time.monotonic()
        for _ in range(count):
            self.check()
            loc, vel, edge = simulator.sample_trajectory(
                T=(length + 1) * 50, sample_freq=50
            )
            value = np.concatenate((loc, vel), axis=1).transpose(0, 2, 1) / 5.0
            if value.shape != (length, 5, 4) or not np.isfinite(value).all():
                raise ValueError("Invalid simulator output")
            values.append(value.astype(np.float32))
            edges.append(edge)
        target.parent.mkdir(parents=True, exist_ok=True)
        array = np.stack(values)
        np.savez_compressed(target, states=array, edges=np.stack(edges), seed=seed)
        write(
            target.with_suffix(".json"),
            {
                "sha256": sha(target),
                "seed": seed,
                "domain": domain,
                "block": block,
                "split": split,
                "regime": regime,
                "parameters": params,
                "shape": list(array.shape),
                "scale": 5,
                "sample_frequency": 50,
                "edges_used_as_input": False,
                "simulator_sha256": sha(Path(simulators.__file__)),
                "seconds": time.monotonic() - started,
            },
        )
        return torch.from_numpy(array)

    @torch.no_grad()
    def metrics(self, model, data, gate=None):
        self.check()
        model.eval()
        hook = None
        if gate is not None:
            hook = model.scene_gate.register_forward_hook(
                lambda module, args, output: torch.full_like(
                    output, (gate - 0.05) / 0.9
                )
            )
        outputs, gates = [], []
        try:
            for chunk in data.split(self.p["batch_size"]):
                self.check()
                pred, _ = model(
                    chunk[:, : self.p["context"]].to(self.device), self.p["forecast"]
                )
                outputs.append(pred.cpu())
                if hasattr(model, "last_scene_gates"):
                    gates.append(torch.cat(model.last_scene_gates).cpu().flatten())
        finally:
            if hook is not None:
                hook.remove()
        prediction = torch.cat(outputs)
        truth = data[:, self.p["context"] :]
        metrics = array_metrics(prediction.numpy(), truth.numpy())
        if gates:
            values = torch.cat(gates)
            metrics["gate_mean"] = float(values.mean())
            metrics["gate_std"] = float(values.std(unbiased=False))
        return metrics, prediction

    def cv(self, data):
        pred = data[:, self.p["context"] - 1 : self.p["context"]].repeat(
            1, self.p["forecast"], 1, 1
        )
        elapsed = torch.arange(1, self.p["forecast"] + 1) * 0.05
        pred[..., :2] += elapsed[None, :, None, None] * pred[..., 2:]
        return array_metrics(pred.numpy(), data[:, self.p["context"] :].numpy())[
            "position_rmse"
        ]

    def update(self, model, optimizer, train, sampler, horizon):
        indices = torch.randint(len(train), (self.p["batch_size"],), generator=sampler)
        start = int(
            torch.randint(
                train.shape[1] - self.p["context"] - horizon + 1, (), generator=sampler
            )
        )
        batch = train[indices].to(self.device)
        model.train()
        pred, _ = model(batch[:, start : start + self.p["context"]], horizon)
        truth = batch[
            :, start + self.p["context"] : start + self.p["context"] + horizon
        ]
        loss = (pred - truth).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        return float(loss.detach()), float(grad)

    def qualify(self):
        import unittest

        suite_module = load("frozen_tests", MODEL_PATH / "test_numeric.py")
        suite = unittest.defaultTestLoader.loadTestsFromModule(suite_module)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        write(
            self.out / "qualification-cpu.json",
            {
                "tests": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped),
            },
        )
        if not result.wasSuccessful() or result.skipped or result.testsRun != 10:
            raise RuntimeError("Frozen qualification failed")
        # Real CUDA full-horizon gradients and checkpoint replay for every comparator.
        checks = []
        for name in self.p["models"]:
            self.check()
            model = self.model(name, 1337)
            data = torch.randn(2, self.p["context"] + self.p["forecast"], 5, 4) * 0.1
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            sampler = torch.Generator().manual_seed(101)
            self.update(model, optimizer, data, sampler, self.p["forecast"])
            _, original = self.metrics(model, data)
            target = self.out / f"qualification-{name}.pt"
            torch.save(model.state_dict(), target)
            clone = self.model(name, 1338)
            clone.load_state_dict(
                torch.load(target, map_location=self.device, weights_only=True)
            )
            _, replay = self.metrics(clone, data)
            if not torch.equal(original, replay):
                raise RuntimeError("CUDA serialized replay failed")
            if name == "radial_mixture":
                fixed = self.model("radial_fixed", 1339)
                fixed.load_state_dict(model.state_dict())
                _, intervention = self.metrics(model, data, gate=0.5)
                _, expected = self.metrics(fixed, data)
                if not torch.equal(intervention, expected):
                    raise RuntimeError(
                        "Frozen gate intervention differs from fixed model"
                    )
                _, restored = self.metrics(model, data)
                if not torch.equal(original, restored):
                    raise RuntimeError("Intervention leaked into adaptive evaluation")
            checks.append(
                {
                    "model": name,
                    "full_horizon_backward_and_replay": "PASS",
                    "parameters": models.parameter_count(model),
                    "gradient_active_parameters": sum(
                        p.numel() for p in model.parameters() if p.grad is not None
                    ),
                }
            )
        write(
            self.out / "qualification-cuda.json", {"status": "PASS", "checks": checks}
        )

    def profile(self):
        rows = []
        for name in self.p["models"]:
            self.check()
            model = self.model(name, 7001)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            sampler = torch.Generator().manual_seed(7002)
            data = (
                torch.randn(
                    self.p["train_scenes"], self.p["context"] + self.p["forecast"], 5, 4
                )
                * 0.05
            )
            timings = []
            for i in range(8):
                torch.cuda.synchronize()
                start = time.monotonic()
                self.update(model, optimizer, data, sampler, self.p["forecast"])
                torch.cuda.synchronize()
                if i >= 3:
                    timings.append(time.monotonic() - start)
            start = time.monotonic()
            self.metrics(model, data)
            self.metrics(model, data[: self.p["validation_scenes"]])
            seconds = time.monotonic() - start
            rows.append(
                {
                    "model": name,
                    "step_seconds_max": max(timings),
                    "validation_and_train_seconds": seconds,
                }
            )
        # Time real simulator creation without exposing any protected test split.
        start = time.monotonic()
        self.data("springs", self.p["calibration_blocks"][0], "train")
        self.data("charged", self.p["calibration_blocks"][0], "train")
        data_seconds = time.monotonic() - start
        repetitions = len(self.p["domains"]) * (
            len(self.p["learning_rates"]) * len(self.p["calibration_blocks"])
            + len(self.p["data_blocks"]) * len(self.p["model_seeds"])
        )
        train_seconds = repetitions * sum(
            r["step_seconds_max"] * self.p["maximum_steps"]
            + r["validation_and_train_seconds"]
            * (self.p["maximum_steps"] // self.p["validate_every"])
            for r in rows
        )
        dataset_factor = (
            len(self.p["calibration_blocks"]) + len(self.p["data_blocks"])
        ) * (
            1
            + (
                self.p["validation_scenes"]
                + len(self.p["regimes"]) * self.p["test_scenes"]
            )
            / self.p["train_scenes"]
        )
        evaluation_seconds = (
            sum(r["validation_and_train_seconds"] for r in rows)
            * len(self.p["domains"])
            * len(self.p["data_blocks"])
            * len(self.p["model_seeds"])
            * (len(self.p["regimes"]) + 2)
        )
        projected = (
            self.p["profile_safety_factor"]
            * (train_seconds + dataset_factor * data_seconds + evaluation_seconds)
            + 600
        )
        result = {
            "rows": rows,
            "real_data_seconds": data_seconds,
            "projected_maximum_seconds": projected,
            "remaining_seconds": self.deadline - time.monotonic(),
            "gate": "PASS"
            if projected < self.deadline - time.monotonic()
            else "BLOCKED_COST",
            "minimum_steps_projection_seconds": train_seconds
            * self.p["minimum_steps"]
            / self.p["maximum_steps"],
        }
        write(self.out / "profile.json", result)
        print(json.dumps(result), flush=True)
        if result["gate"] != "PASS":
            raise RuntimeError("BLOCKED_COST_FULL_CONVERGENCE_MATRIX")

    def fit(self, domain, block, seed, name, lr, phase):
        self.check()
        folder = self.out / "fits" / phase / f"{domain}-{block}-{seed}-{name}-{lr}"
        folder.mkdir(parents=True, exist_ok=False)
        train, validation = (
            self.data(domain, block, "train"),
            self.data(domain, block, "validation"),
        )
        cv = self.cv(train)
        model = self.model(name, seed)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=self.p["weight_decay"]
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.p["scheduler"]["factor"],
            patience=self.p["scheduler"]["patience"],
            min_lr=self.p["scheduler"]["minimum_lr"],
        )
        sampler = torch.Generator().manual_seed(seed + 900001)
        best, curve, selected = float("inf"), [], None
        started = time.monotonic()
        for step in range(1, self.p["maximum_steps"] + 1):
            self.check()
            horizon = next(h for end, h in self.p["curriculum"] if step <= end)
            loss, grad = self.update(model, optimizer, train, sampler, horizon)
            if step % self.p["validate_every"]:
                continue
            tm, _ = self.metrics(model, train)
            vm, _ = self.metrics(model, validation)
            row = {
                "step": step,
                "horizon": horizon,
                "train_rmse": tm["position_rmse"],
                "validation_rmse": vm["position_rmse"],
                "loss": loss,
                "gradient_norm": grad,
                "lr": optimizer.param_groups[0]["lr"],
            }
            curve.append(row)
            write(folder / f"curve-{step:06d}.json", row)
            # Scheduler sees the same full-horizon validation objective throughout.
            sched.step(vm["position_rmse"])
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": sched.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "sampler": sampler.get_state(),
                "step": step,
                "name": name,
                "hidden": self.widths[name],
            }
            torch.save(checkpoint, folder / "latest.pt")
            # Select only checkpoints trained on the full forecast objective.
            if horizon == self.p["forecast"] and vm["position_rmse"] < best:
                best = vm["position_rmse"]
                selected = {
                    "step": step,
                    "train_rmse": tm["position_rmse"],
                    "validation_rmse": best,
                    "train_gate_mean": tm.get("gate_mean"),
                }
                torch.save(checkpoint, folder / "best.pt")
            status = adequate(
                curve, selected["train_rmse"] if selected else cv, cv, self.p
            )
            if status["pass"]:
                break
        if selected is None:
            raise RuntimeError("No eligible full-horizon checkpoint")
        status = adequate(curve, selected["train_rmse"], cv, self.p)
        model.load_state_dict(
            torch.load(folder / "best.pt", map_location=self.device, weights_only=True)[
                "model"
            ]
        )
        tm, _ = self.metrics(model, train)
        vm, _ = self.metrics(model, validation)
        if (
            abs(tm["position_rmse"] - selected["train_rmse"]) > 1e-8
            or abs(vm["position_rmse"] - best) > 1e-8
        ):
            raise RuntimeError("Selected checkpoint metric replay failed")
        receipt = {
            "domain": domain,
            "block": block,
            "seed": seed,
            "model": name,
            "lr": lr,
            "phase": phase,
            "updates": step,
            "seconds": time.monotonic() - started,
            "selected": selected,
            "cv_train_rmse": cv,
            "adequacy": status,
            "checkpoint": str((folder / "best.pt").relative_to(self.out)),
            "checkpoint_sha256": sha(folder / "best.pt"),
            "curve": curve,
            "parameters": models.parameter_count(model),
        }
        write(folder / "receipt.json", receipt)
        print(
            json.dumps({k: v for k, v in receipt.items() if k != "curve"}), flush=True
        )
        return receipt

    def train(self):
        tuning, choices = [], {}
        for domain in self.p["domains"]:
            for name in self.p["models"]:
                trials = []
                for lr in self.p["learning_rates"]:
                    trials.append(
                        self.fit(
                            domain,
                            self.p["calibration_blocks"][0],
                            self.p["calibration_model_seed"],
                            name,
                            lr,
                            "calibration",
                        )
                    )
                tuning.extend(trials)
                eligible = [r for r in trials if r["adequacy"]["pass"]]
                if eligible:
                    choices[f"{domain}/{name}"] = min(
                        eligible,
                        key=lambda r: (r["selected"]["validation_rmse"], r["lr"]),
                    )["lr"]
        write(self.out / "tuning.json", {"trials": tuning, "choices": choices})
        if len(choices) != len(self.p["domains"]) * len(self.p["models"]):
            raise RuntimeError("INCONCLUSIVE_UNDERTRAINED_CALIBRATION_TEST_CLOSED")
        selections = []
        for domain in self.p["domains"]:
            for block in self.p["data_blocks"]:
                for seed in self.p["model_seeds"]:
                    names = self.p["models"]
                    offset = (block + seed) % len(names)
                    for name in names[offset:] + names[:offset]:
                        selections.append(
                            self.fit(
                                domain,
                                block,
                                seed,
                                name,
                                choices[f"{domain}/{name}"],
                                "main",
                            )
                        )
        write(self.out / "selections.json", selections)
        # Recompute gates from saved curves and rehashed checkpoints, not cached booleans.
        for row in selections:
            replay = adequate(
                row["curve"],
                row["selected"]["train_rmse"],
                row["cv_train_rmse"],
                self.p,
            )
            if (
                replay != row["adequacy"]
                or sha(self.out / row["checkpoint"]) != row["checkpoint_sha256"]
            ):
                raise RuntimeError("Adequacy or checkpoint changed")
        if not all(r["adequacy"]["pass"] for r in selections):
            raise RuntimeError("INCONCLUSIVE_UNDERTRAINED_MAIN_TEST_CLOSED")
        write(
            self.out / "test-admission.json",
            {
                "status": "PASS",
                "selections_sha256": sha(self.out / "selections.json"),
                "protocol_sha256": sha(ROOT / "protocol.json"),
                "fits": len(selections),
            },
        )
        return selections

    def evaluate(self, selections):
        admitted = read(self.out / "test-admission.json")
        if admitted["selections_sha256"] != sha(self.out / "selections.json"):
            raise RuntimeError("Selection lock changed")
        rows = []
        for selection in selections:
            self.check()
            d, b, s, n = (selection[k] for k in ("domain", "block", "seed", "model"))
            model = self.model(n, s)
            model.load_state_dict(
                torch.load(
                    self.out / selection["checkpoint"],
                    map_location=self.device,
                    weights_only=True,
                )["model"]
            )
            metrics, intervention = {}, {}
            for regime in self.p["regimes"]:
                data = self.data(d, b, "test", regime)
                metric, prediction = self.metrics(model, data)
                target = data[:, self.p["context"] :].numpy()
                path = self.out / "predictions" / f"{d}-{b}-{s}-{n}-{regime}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                arrays = {"prediction": prediction.numpy(), "target": target}
                metrics[regime] = metric
                if n == "radial_mixture":
                    m, pred = self.metrics(
                        model, data, gate=selection["selected"]["train_gate_mean"]
                    )
                    intervention[regime] = m
                    arrays["intervention"] = pred.numpy()
                np.savez_compressed(path, **arrays)
                with np.load(path, allow_pickle=False) as archive:
                    if (
                        array_metrics(archive["prediction"], archive["target"])[
                            "position_rmse"
                        ]
                        != metric["position_rmse"]
                    ):
                        raise RuntimeError("Prediction metric replay failed")
                    if (
                        "intervention" in archive
                        and array_metrics(archive["intervention"], archive["target"])[
                            "position_rmse"
                        ]
                        != intervention[regime]["position_rmse"]
                    ):
                        raise RuntimeError("Intervention replay failed")
                # Fresh reconstructed checkpoint already used above; repeat prediction to verify determinism.
                _, repeated = self.metrics(model, data)
                if not torch.equal(prediction, repeated):
                    raise RuntimeError("Held-out prediction replay failed")
            row = {**selection, "metrics": metrics, "intervention": intervention}
            write(self.out / "evaluations" / f"{d}-{b}-{s}-{n}.json", row)
            rows.append(row)
        result = summarize(rows, self.p)
        write(self.out / "summary.json", result)
        return result


def array_metrics(prediction, target):
    if (
        prediction.shape != target.shape
        or not np.isfinite(prediction).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Invalid prediction arrays")
    squared = (prediction.astype(np.float64) - target.astype(np.float64)) ** 2
    return {
        "position_rmse": float(np.sqrt(squared[..., :2].mean())),
        "velocity_rmse": float(np.sqrt(squared[..., 2:].mean())),
        "scene_position_mse": squared[..., :2].mean(axis=(1, 2, 3)).tolist(),
        "horizon_position_rmse": np.sqrt(
            squared[..., :2].mean(axis=(0, 2, 3))
        ).tolist(),
    }
