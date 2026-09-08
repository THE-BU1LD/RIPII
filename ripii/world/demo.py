from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from .experiment import load_model
from .models import rollout
from .physics import sample_scene, simulate

_MATPLOTLIB_TEMP_CACHE: tempfile.TemporaryDirectory[str] | None = None


def _configure_matplotlib_cache() -> None:
    """Give headless/restricted runs a writable cache before importing Matplotlib."""
    global _MATPLOTLIB_TEMP_CACHE
    if "MPLCONFIGDIR" not in os.environ:
        _MATPLOTLIB_TEMP_CACHE = tempfile.TemporaryDirectory(
            prefix="ripii-matplotlib-"
        )
        os.environ["MPLCONFIGDIR"] = _MATPLOTLIB_TEMP_CACHE.name


class WorldDemo:
    """Native interactive workbench; predictions always come from the loaded weights."""

    def __init__(self, checkpoint: Path, seed=42):
        _configure_matplotlib_cache()
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import Button, RadioButtons, Slider

        self.plt = plt
        self.model, self.checkpoint = load_model(checkpoint)
        self.generator = torch.Generator().manual_seed(seed)
        self.objects, self.horizon, self.selected = 4, 48, 0
        self.force = torch.zeros(2)
        self.playing, self.frame, self.dragging = False, 0, None
        self.palette = "Original"
        self.fig = plt.figure(figsize=(13, 8), facecolor="#101820")
        self.axes = [self.fig.add_axes([x, 0.28, 0.40, 0.58]) for x in (0.06, 0.54)]
        self.fig.text(
            0.06,
            0.95,
            "RIPII / INTERVENTION LAB",
            color="#f1f5ed",
            size=19,
            weight="bold",
        )
        self.fig.text(
            0.06,
            0.915,
            f"{self.model.variant} · trained {self.checkpoint['completed_steps']} updates · known object states",
            color="#9cacb8",
            size=10,
        )
        self.status = self.fig.text(0.06, 0.24, "", color="#f1f5ed", size=11)
        self.fig.text(
            0.06,
            0.025,
            "Drag left: move object. Drag right: set velocity. Select an object before applying force. Space: play/pause.",
            color="#9cacb8",
            size=9,
        )
        self.buttons = []
        for x, label, callback in (
            (0.06, "Play / pause", self.toggle),
            (0.20, "New scene", self.reset),
            (0.34, "Rewind", self.rewind),
        ):
            button = Button(
                self.fig.add_axes([x, 0.17, 0.12, 0.045]),
                label,
                color="#c6e98b",
                hovercolor="#e0ffb0",
            )
            button.on_clicked(callback)
            self.buttons.append(button)
        self.object_slider = Slider(
            self.fig.add_axes([0.64, 0.19, 0.26, 0.025]),
            "Objects",
            2,
            min(12, self.model.max_objects),
            valinit=4,
            valstep=1,
            color="#c6e98b",
        )
        self.object_slider.on_changed(self.change_objects)
        self.force_x = Slider(
            self.fig.add_axes([0.12, 0.115, 0.30, 0.025]),
            "Force x",
            -1.0,
            1.0,
            valinit=0,
            color="#c6e98b",
        )
        self.force_y = Slider(
            self.fig.add_axes([0.12, 0.065, 0.30, 0.025]),
            "Force y",
            -1.0,
            1.0,
            valinit=0,
            color="#c6e98b",
        )
        self.force_x.on_changed(self.change_force)
        self.force_y.on_changed(self.change_force)
        self.palette_control = RadioButtons(
            self.fig.add_axes([0.60, 0.05, 0.15, 0.10], facecolor="#dce5dd"),
            ("Original", "Recolored"),
        )
        self.palette_control.on_clicked(self.change_palette)
        for slider in (self.object_slider, self.force_x, self.force_y):
            slider.label.set_color("#f1f5ed")
            slider.valtext.set_color("#f1f5ed")
        self.fig.canvas.mpl_connect("button_press_event", self.press)
        self.fig.canvas.mpl_connect("button_release_event", self.release)
        self.fig.canvas.mpl_connect(
            "key_press_event", lambda e: self.toggle() if e.key == " " else None
        )
        self.reset()
        self.animation = FuncAnimation(
            self.fig, self.tick, interval=50, cache_frame_data=False
        )

    def reset(self, _=None):
        self.initial, self.mask = sample_scene(
            self.generator, self.objects, self.model.max_objects
        )
        self.selected = min(self.selected, self.objects - 1)
        self.recompute()

    def change_objects(self, value):
        self.objects = int(value)
        self.reset()

    def change_palette(self, label):
        self.palette = label
        # Appearance is renderer-only; the model explicitly consumes object states.
        self.draw()

    def change_force(self, _=None):
        self.force = torch.tensor([self.force_x.val, self.force_y.val])
        self.recompute()

    @torch.no_grad()
    def recompute(self):
        self.frame = 0
        actions = torch.zeros(1, self.horizon, self.model.max_objects, 2)
        actions[0, :, self.selected] = self.force
        self.prediction = rollout(
            self.model, self.initial[None], actions, self.mask[None]
        )[0]
        truth = [self.initial]
        for action in actions[0]:
            truth.append(simulate(truth[-1][None], action[None], self.mask[None])[0])
        self.truth = torch.stack(truth)
        self.error = (
            (
                (self.prediction[:, self.mask, :2] - self.truth[:, self.mask, :2])
                .square()
                .mean()
            )
            .sqrt()
            .item()
        )
        self.draw()

    def toggle(self, _=None):
        self.playing = not self.playing

    def rewind(self, _=None):
        self.frame = 0
        self.draw()

    def tick(self, _=None):
        if self.playing:
            self.frame = (self.frame + 1) % (self.horizon + 1)
            self.draw()

    def press(self, event):
        if event.inaxes is self.axes[0] and event.xdata is not None:
            point = torch.tensor([event.xdata, event.ydata])
            distances = (self.truth[self.frame, : self.objects, :2] - point).norm(
                dim=-1
            )
            i = int(distances.argmin())
            if distances[i] < self.initial[i, 4] + 0.07:
                self.selected = i
                self.dragging = event.button
                self.drag_start = point
                self.drag_origin = self.initial[i, :2].clone()
                self.playing = False
                self.frame = 0
                self.draw()

    def release(self, event):
        if self.dragging is None:
            return
        if event.inaxes is self.axes[0] and event.xdata is not None:
            point = torch.tensor([event.xdata, event.ydata])
            displacement = point - self.drag_start
            if self.dragging == 1:
                radius = float(self.initial[self.selected, 4])
                point = (self.drag_origin + displacement).clamp(-1 + radius, 1 - radius)
                others = torch.arange(self.objects) != self.selected
                separation = (self.initial[: self.objects, :2][others] - point).norm(
                    dim=-1
                )
                if torch.all(
                    separation > self.initial[: self.objects, 4][others] + radius
                ):
                    self.initial[self.selected, :2] = point
            elif self.dragging == 3:
                self.initial[self.selected, 2:4] = displacement.clamp(-1.5, 1.5) * 2
            self.recompute()
        self.dragging = None

    def draw(self):
        from matplotlib.patches import Circle

        palette = self.plt.get_cmap("Set2" if self.palette == "Original" else "plasma")
        for ax, trajectory, title in zip(
            self.axes,
            (self.truth, self.prediction),
            ("SIMULATOR", "LEARNED PREDICTION"),
        ):
            ax.clear()
            ax.set_facecolor("#182630")
            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-1.15, 1.15)
            ax.set_aspect("equal")
            ax.tick_params(colors="#6e8291", labelsize=8)
            ax.set_title(title, color="#b9cec8", fontsize=10, loc="left", pad=10)
            ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color="#638071", lw=1)
            for i in range(self.objects):
                color = palette(i / max(1, self.objects - 1))
                trail = trajectory[: self.frame + 1, i].numpy()
                state = trajectory[self.frame, i].numpy()
                ax.plot(trail[:, 0], trail[:, 1], color=color, alpha=0.6, lw=1.5)
                ax.add_patch(
                    Circle(
                        state[:2],
                        float(state[4]),
                        facecolor=color,
                        edgecolor="white" if i == self.selected else "none",
                        linewidth=2,
                    )
                )
                ax.text(
                    state[0],
                    state[1],
                    str(i + 1),
                    color="#101820",
                    ha="center",
                    va="center",
                    size=8,
                )
                if self.frame == 0:
                    ax.arrow(
                        state[0],
                        state[1],
                        state[2] * 0.15,
                        state[3] * 0.15,
                        width=0.004,
                        color=color,
                    )
        regime = (
            "unseen object count" if self.objects > 4 else "training-range object count"
        )
        self.status.set_text(
            f"t = {self.frame * 0.05:.2f}s   |   rollout position RMSE {self.error:.4f}   |   object {self.selected + 1} selected   |   {regime}"
        )
        self.fig.canvas.draw_idle()

    def export(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.playing = False
        self.frame = self.horizon
        self.draw()
        self.fig.savefig(path, dpi=150, facecolor=self.fig.get_facecolor())


def main(checkpoint: Path, export: Path | None = None, seed=42):
    _configure_matplotlib_cache()
    if export:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    demo = WorldDemo(checkpoint, seed)
    if export:
        demo.export(export)
        plt.close(demo.fig)
    else:
        plt.show()
    return demo
