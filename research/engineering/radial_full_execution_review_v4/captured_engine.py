"""Prospective execution repair for the unchanged radial-adaptation-v2 matrix.

Only execution changes: model state, objective, optimizer, curriculum, validation,
selection, adequacy, and evaluation semantics remain those of the frozen reference.
"""

import copy
import gc
import json
import math
import os
import time

import torch
from control import adequate, sha, write
from engine import Engine
from optimized import make_optimized, reference


class Plateau:
    """Protocol-exact ReduceLROnPlateau(min) state with a device LR tensor."""

    def __init__(self, lr, protocol):
        self.lr = lr
        self.factor = protocol["factor"]
        self.patience = protocol["patience"]
        self.minimum = protocol["minimum_lr"]
        self.best = math.inf
        self.bad_epochs = 0
        self.last_epoch = 0

    def step(self, value):
        self.last_epoch += 1
        if value < self.best * (1.0 - 1e-4):
            self.best = value
            self.bad_epochs = 0
            return
        self.bad_epochs += 1
        if self.bad_epochs > self.patience:
            old = float(self.lr.item())
            new = max(old * self.factor, self.minimum)
            if old - new > 1e-8:
                self.lr.fill_(new)
            self.bad_epochs = 0

    def state_dict(self):
        return {
            "best": self.best,
            "num_bad_epochs": self.bad_epochs,
            "last_epoch": self.last_epoch,
            "factor": self.factor,
            "patience": self.patience,
            "min_lrs": [self.minimum],
            "threshold": 1e-4,
            "threshold_mode": "rel",
            "mode": "min",
        }


class CapturedEngine(Engine):
    def metrics(self, model, data, gate=None):
        # Optimized rollout omits diagnostics. Evaluate with the frozen reference
        # so the train-mean gate intervention uses real, current gate values.
        diagnostic = reference.make_model(
            model._execution_name, self.widths[model._execution_name], self.p["context"]
        ).to(self.device)
        diagnostic.load_state_dict(model.state_dict())
        return super().metrics(diagnostic, data, gate)

    def model(self, name, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = make_optimized(
            reference.make_model(name, self.widths[name], self.p["context"])
        ).to(self.device)
        model._execution_name = name
        return model

    def captured_update(self, model, optimizer, train, sampler, horizon):
        indices = torch.randint(len(train), (self.p["batch_size"],), generator=sampler)
        start = int(
            torch.randint(
                train.shape[1] - self.p["context"] - horizon + 1,
                (),
                generator=sampler,
            )
        )
        batch = train[indices]
        x = batch[:, start : start + self.p["context"]].to(self.device)
        y = batch[
            :, start + self.p["context"] : start + self.p["context"] + horizon
        ].to(self.device)
        model.train()
        # One non-default stream owns warmup, capture and replay. Retained
        # autograd nodes must never cross from a default-stream warmup into
        # capture (the failure observed in attempt 2).
        if not hasattr(self, "_training_stream"):
            self._training_stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        self._training_stream.wait_stream(current)
        with torch.cuda.stream(self._training_stream):
            entry = self._graphs.get(horizon)
            if entry is None:
                # Curriculum horizons only increase. Release obsolete graphs
                # and their activation pools rather than retaining all seven.
                self._graphs.clear()
                optimizer.zero_grad(set_to_none=True)
                prediction, _ = model(x, horizon)
                loss = (prediction - y).square().mean()
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                self._graphs[horizon] = {"warm": True}
            else:
                if "graph" not in entry:
                    static_x, static_y = torch.empty_like(x), torch.empty_like(y)
                    static_x.copy_(x)
                    static_y.copy_(y)
                    graph = torch.cuda.CUDAGraph()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.graph(graph, stream=self._training_stream):
                        prediction, _ = model(static_x, horizon)
                        loss = (prediction - static_y).square().mean()
                        loss.backward()
                        # error_if_nonfinite=True reads a GPU scalar on the CPU
                        # and is illegal inside capture. Check after every replay.
                        grad = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0, error_if_nonfinite=False
                        )
                        optimizer.step()
                    entry.update(
                        graph=graph,
                        x=static_x,
                        y=static_y,
                        loss=loss.detach(),
                        grad=grad.detach(),
                    )
                else:
                    entry["x"].copy_(x)
                    entry["y"].copy_(y)
                entry["graph"].replay()
                loss, grad = entry["loss"], entry["grad"]
        current.wait_stream(self._training_stream)
        # A failed finite check makes this attempt invalid; it cannot proceed
        # to validation, checkpoint selection or protected evaluation.
        if not torch.isfinite(loss) or not torch.isfinite(grad):
            raise FloatingPointError("Nonfinite captured update; attempt invalid")
        return loss.detach(), grad.detach()

    def fit(self, domain, block, seed, name, lr, phase):
        self.check()
        folder = self.out / "fits" / phase / f"{domain}-{block}-{seed}-{name}-{lr}"
        folder.mkdir(parents=True, exist_ok=True)
        identity = {
            "domain": domain,
            "block": block,
            "seed": seed,
            "name": name,
            "lr": lr,
            "phase": phase,
            "protocol": self.p,
        }
        if (folder / "receipt.json").exists():
            receipt = json.loads((folder / "receipt.json").read_text())
            if (
                receipt.get("identity") != identity
                or sha(folder / "best.pt") != receipt["checkpoint_sha256"]
            ):
                raise RuntimeError("Completed fit resume identity differs")
            receipt["checkpoint"] = str((folder / "best.pt").relative_to(self.out))
            return receipt
        train, validation = (
            self.data(domain, block, "train"),
            self.data(domain, block, "validation"),
        )
        cv = self.cv(train)
        model = self.model(name, seed)
        lr_tensor = torch.tensor(lr, device=self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr_tensor,
            weight_decay=self.p["weight_decay"],
            capturable=True,
        )
        scheduler = Plateau(lr_tensor, self.p["scheduler"])
        sampler = torch.Generator().manual_seed(seed + 900001)
        best, curve, selected, best_checkpoint = float("inf"), [], None, None
        start_step = 0
        latest = folder / "latest.pt"
        if latest.exists():
            checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
            if checkpoint["identity"] != identity:
                raise RuntimeError("Resume identity differs")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"].to(
                self.device
            )
            # The optimizer's LR tensor must also be the scheduler's tensor.
            lr_tensor = optimizer.param_groups[0]["lr"]
            scheduler = Plateau(lr_tensor, self.p["scheduler"])
            saved = checkpoint["scheduler"]
            scheduler.best = saved["best"]
            scheduler.bad_epochs = saved["num_bad_epochs"]
            scheduler.last_epoch = saved["last_epoch"]
            torch.set_rng_state(checkpoint["rng"])
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
            sampler.set_state(checkpoint["sampler"])
            start_step = checkpoint["step"]
            curve, selected, best = (
                checkpoint["curve"],
                checkpoint["selected"],
                checkpoint["best"],
            )
            best_checkpoint = checkpoint["best_checkpoint"]
        self._graphs = {}
        started = time.monotonic()
        step = start_step
        already_adequate = adequate(
            curve, selected["train_rmse"] if selected else cv, cv, self.p
        )["pass"]
        for step in range(start_step + 1, self.p["maximum_steps"] + 1):
            if already_adequate:
                step = start_step
                break
            if step % 25 == 1:
                self.check()
            horizon = next(h for end, h in self.p["curriculum"] if step <= end)
            loss, grad = self.captured_update(model, optimizer, train, sampler, horizon)
            if step % self.p["validate_every"]:
                continue
            self.check()
            torch.cuda.synchronize()
            loss_value, grad_value = float(loss), float(grad)
            if not math.isfinite(loss_value) or not math.isfinite(grad_value):
                raise FloatingPointError("Nonfinite captured update")
            tm, _ = self.metrics(model, train)
            vm, _ = self.metrics(model, validation)
            row = {
                "step": step,
                "horizon": horizon,
                "train_rmse": tm["position_rmse"],
                "validation_rmse": vm["position_rmse"],
                "loss": loss_value,
                "gradient_norm": grad_value,
                "lr": float(lr_tensor.item()),
            }
            curve.append(row)
            scheduler.step(vm["position_rmse"])
            improved = horizon == self.p["forecast"] and vm["position_rmse"] < best
            if improved:
                best = vm["position_rmse"]
                selected = {
                    "step": step,
                    "train_rmse": tm["position_rmse"],
                    "validation_rmse": best,
                    "train_gate_mean": tm.get("gate_mean"),
                }
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "sampler": sampler.get_state(),
                "step": step,
                "name": name,
                "hidden": self.widths[name],
                "identity": identity,
                "curve": curve,
                "selected": selected,
                "best": best,
                "execution": "CUDA_GRAPH_REPAIR_V3_REQUIRES_QUALIFICATION",
            }
            if improved:
                best_checkpoint = {
                    "model": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    "step": step,
                    "name": name,
                    "hidden": self.widths[name],
                    "identity": identity,
                }
            # One atomic file is the recovery transaction, including the selected
            # model. A crash cannot leave best.pt ahead of the saved optimizer.
            checkpoint["best_checkpoint"] = copy.deepcopy(best_checkpoint)
            torch.save(checkpoint, folder / ".latest.pt.tmp")
            os.replace(folder / ".latest.pt.tmp", latest)
            print(
                json.dumps({"fit": folder.name, **row, "checkpoint": str(latest)}),
                flush=True,
            )
            status = adequate(
                curve, selected["train_rmse"] if selected else cv, cv, self.p
            )
            if status["pass"]:
                break
        if selected is None:
            raise RuntimeError("No eligible full-horizon checkpoint")
        torch.save(best_checkpoint, folder / ".best.pt.tmp")
        os.replace(folder / ".best.pt.tmp", folder / "best.pt")
        status = adequate(curve, selected["train_rmse"], cv, self.p)
        self._graphs.clear()
        torch.cuda.synchronize()
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
            "identity": identity,
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
            "parameters": reference.parameter_count(model),
            "execution": "CUDA_GRAPH_REPAIR_V3_REQUIRES_QUALIFICATION",
        }
        write(folder / "receipt.json", receipt)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        return receipt
