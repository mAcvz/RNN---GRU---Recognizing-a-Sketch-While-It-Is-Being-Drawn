"""
Plotting helpers for Quick, Draw! sketches.

- draw(ax, seq): draw a sketch (stroke-3 format) on a matplotlib axis
- animate(seq, probs, classes, path): GIF of the sketch being drawn point by point,
  next to the probability the network assigns to each class at that moment
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def _strokes(seq):
    """stroke-3 (dx, dy, pen_lifted) -> list of strokes in absolute coordinates."""
    xy = np.cumsum(seq[:, :2], axis=0)
    xy[:, 1] *= -1                               # y points up (in the data it points down)
    stroke_ends = np.where(seq[:, 2] == 1)[0] + 1
    return np.split(xy, stroke_ends)


def draw(ax, seq, n_points=None, color="black"):
    """Draw the first n_points of the sketch (all of them if None)."""
    seq = seq if n_points is None else seq[:n_points]
    for stroke in _strokes(seq):
        if len(stroke) > 0:
            ax.plot(stroke[:, 0], stroke[:, 1], color=color, lw=2)
    ax.set_aspect("equal")
    ax.axis("off")


def animate(seq, probs, classes, path, fps=12, final_pause=15):
    """
    seq:   (T, 3) sketch in stroke-3 format (real points only, no padding)
    probs: (T, n_classes) network probabilities after seeing 1, 2, ..., T points
    Saves a GIF at `path`.
    """
    T = len(seq)
    xy = np.concatenate(_strokes(seq))
    margin = 0.1 * (xy.max(0) - xy.min(0)).max()

    fig, (ax_d, ax_p) = plt.subplots(1, 2, figsize=(9, 4), dpi=80,
                                     gridspec_kw={"width_ratios": [1, 1.2]})
    bars = ax_p.barh(classes, probs[0], color="lightgray")
    ax_p.set_xlim(0, 1)
    ax_p.invert_yaxis()
    ax_p.set_xlabel("probability")

    def frame(i):
        t = min(i, T - 1) + 1                    # points seen so far
        ax_d.clear()
        draw(ax_d, seq, t)
        ax_d.set_xlim(xy[:, 0].min() - margin, xy[:, 0].max() + margin)
        ax_d.set_ylim(xy[:, 1].min() - margin, xy[:, 1].max() + margin)
        ax_d.set_title(f"points seen: {t}/{T}  ({100 * t / T:.0f}%)")

        p = probs[t - 1]
        best = p.argmax()
        for k, b in enumerate(bars):
            b.set_width(p[k])
            b.set_color("crimson" if k == best else "lightgray")
        ax_p.set_title(f"the network says: {classes[best]} ({p[best]:.0%})")
        return bars

    anim = FuncAnimation(fig, frame, frames=T + final_pause)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
